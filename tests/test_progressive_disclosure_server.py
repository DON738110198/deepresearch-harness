from __future__ import annotations

from collections.abc import Sequence

import pytest

from deepresearch_harness.progressive_disclosure import (
    EvidenceCandidate,
    ProgressiveDisclosurePolicy,
)
from deepresearch_harness.progressive_disclosure_server import (
    ProgressiveDisclosureRuntime,
)


class WordTokenizer:
    def __init__(self) -> None:
        self._tokens: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str) -> Sequence[object]:
        output = []
        for word in text.split():
            token = self._tokens.setdefault(word, len(self._tokens) + 1)
            self._words[token] = word
            output.append(token)
        return output

    def decode(self, tokens: Sequence[object]) -> str:
        return " ".join(self._words[int(token)] for token in tokens)


def _candidate(docid: str, score: float = 1.0) -> EvidenceCandidate:
    return EvidenceCandidate(
        docid=docid,
        score=score,
        text=" ".join(f"{docid}-{index}" for index in range(20)),
    )


def _runtime() -> ProgressiveDisclosureRuntime:
    documents = {"dense-1": _candidate("dense-1").text}
    return ProgressiveDisclosureRuntime(
        retriever_id="test-hybrid",
        policy=ProgressiveDisclosurePolicy(
            anchor_count=1,
            dense_lead_count=1,
            anchor_token_cap=16,
            lead_token_cap=4,
            open_token_cap=16,
            maximum_open_calls=1,
            total_evidence_ingress_token_budget=512,
            open_evidence_ingress_token_budget=16,
        ),
        tokenizer=WordTokenizer(),
        bm25_search=lambda _query: [_candidate("anchor")],
        dense_search=lambda _query: [_candidate("anchor"), _candidate("dense-1")],
        document_loader=documents.get,
    )


def test_runtime_exposes_anchor_and_lead_without_duplicate_content() -> None:
    runtime = _runtime()

    response = runtime.search(run_id="run-1", query="rare clue")

    assert [row.docid for row in response.results] == ["anchor", "dense-1"]
    assert response.disclosure.anchors_returned == 1
    assert response.disclosure.leads_returned == 1
    assert response.disclosure.cross_channel_duplicate_slots == 1
    assert response.disclosure.new_ingress_tokens == 20
    assert response.state.eligible_open_docids == ("dense-1",)
    assert runtime.health().session_count == 1


def test_runtime_opens_only_a_lead_from_the_same_run() -> None:
    runtime = _runtime()
    runtime.search(run_id="run-1", query="rare clue")

    denied = runtime.open_evidence(run_id="run-1", docid="anchor")
    opened = runtime.open_evidence(run_id="run-1", docid="dense-1")

    assert denied.result.outcome == "already_opened"
    assert opened.result.outcome == "opened"
    assert opened.result.ingress_tokens == 16
    assert opened.state.successful_open_calls == 1
    assert opened.state.cumulative_ingress_tokens == 36


def test_runtime_rejects_state_or_open_for_unknown_run() -> None:
    runtime = _runtime()

    with pytest.raises(KeyError, match="unknown run_id"):
        runtime.state(run_id="missing")
    with pytest.raises(KeyError, match="unknown run_id"):
        runtime.open_evidence(run_id="missing", docid="dense-1")
