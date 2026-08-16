from __future__ import annotations

from collections.abc import Sequence

import pytest

from deepresearch_harness.progressive_disclosure import (
    EvidenceCandidate,
    ProgressiveDisclosurePolicy,
    ProgressiveDisclosureSession,
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


def _candidate(docid: str, words: int = 20) -> EvidenceCandidate:
    return EvidenceCandidate(
        docid=docid,
        score=1.0,
        text=" ".join(f"{docid}-{index}" for index in range(words)),
    )


def _session(*, maximum_open_calls: int = 8) -> ProgressiveDisclosureSession:
    documents = {
        "c": _candidate("c").text,
        "d": _candidate("d").text,
    }
    return ProgressiveDisclosureSession(
        run_id="run-1",
        policy=ProgressiveDisclosurePolicy(
            anchor_count=2,
            dense_lead_count=2,
            anchor_token_cap=16,
            lead_token_cap=4,
            open_token_cap=16,
            maximum_open_calls=maximum_open_calls,
            total_evidence_ingress_token_budget=512,
            open_evidence_ingress_token_budget=16,
        ),
        tokenizer=WordTokenizer(),
        document_loader=documents.get,
    )


def test_search_preserves_anchors_and_deduplicates_dense_leads() -> None:
    session = _session()
    result = session.search(
        bm25_candidates=[_candidate("a"), _candidate("a"), _candidate("b")],
        dense_candidates=[
            _candidate("b"),
            _candidate("c"),
            _candidate("d"),
            _candidate("c"),
        ],
    )

    assert [row.docid for row in result.results] == ["a", "b", "c", "d"]
    assert [row.channel for row in result.results] == [
        "bm25_anchor",
        "bm25_anchor",
        "dense_lead",
        "dense_lead",
    ]
    assert result.within_channel_duplicate_slots == 2
    assert result.cross_channel_duplicate_slots == 1
    assert result.new_ingress_tokens == 40

    repeated = session.search(
        bm25_candidates=[_candidate("a"), _candidate("b")],
        dense_candidates=[_candidate("b"), _candidate("c"), _candidate("d")],
    )
    assert repeated.results == []
    assert repeated.prior_context_duplicate_slots == 5


def test_open_evidence_is_eligible_bounded_and_idempotent() -> None:
    session = _session(maximum_open_calls=1)
    session.search(
        bm25_candidates=[_candidate("a"), _candidate("b")],
        dense_candidates=[_candidate("c"), _candidate("d")],
    )

    denied = session.open_evidence("never-disclosed")
    opened = session.open_evidence("c")
    repeated = session.open_evidence("c")
    limited = session.open_evidence("d")

    assert denied.outcome == "not_disclosed"
    assert opened.outcome == "opened"
    assert opened.ingress_tokens == 16
    assert repeated.outcome == "already_opened"
    assert limited.outcome == "open_limit_reached"
    snapshot = session.snapshot()
    assert snapshot.open_attempts == 4
    assert snapshot.successful_open_calls == 1
    assert snapshot.cumulative_ingress_tokens == 56
    assert snapshot.search_ingress_tokens == 40
    assert snapshot.open_ingress_tokens == 16


def test_total_ingress_budget_stops_search_and_open() -> None:
    session = ProgressiveDisclosureSession(
        run_id="run-budget",
        policy=ProgressiveDisclosurePolicy(
            anchor_count=1,
            dense_lead_count=1,
            anchor_token_cap=512,
            lead_token_cap=4,
            open_token_cap=16,
            maximum_open_calls=1,
            total_evidence_ingress_token_budget=512,
            open_evidence_ingress_token_budget=0,
        ),
        tokenizer=WordTokenizer(),
        document_loader=lambda _docid: _candidate("c").text,
    )
    result = session.search(
        bm25_candidates=[_candidate("a", words=600)],
        dense_candidates=[_candidate("c")],
    )

    assert [row.docid for row in result.results] == ["a"]
    assert result.ingress_budget_exhausted is True
    assert session.open_evidence("c").outcome == "not_disclosed"
    assert session.snapshot().remaining_ingress_tokens == 0


def test_obligation_policy_can_reopen_a_disclosed_anchor_preview() -> None:
    calls: list[tuple[str, str]] = []

    def load_span(docid: str, obligation_query: str) -> str:
        calls.append((docid, obligation_query))
        return "target answer from a far-away section"

    session = ProgressiveDisclosureSession(
        run_id="run-obligation",
        policy=ProgressiveDisclosurePolicy(
            anchor_count=1,
            dense_lead_count=1,
            anchor_token_cap=16,
            lead_token_cap=4,
            open_token_cap=16,
            maximum_open_calls=1,
            total_evidence_ingress_token_budget=512,
            open_evidence_ingress_token_budget=16,
            anchor_open_policy="reopen_with_obligation",
            open_content_policy="answer_obligation_window_v0",
        ),
        tokenizer=WordTokenizer(),
        document_loader=lambda _docid: "unused",
        obligation_document_loader=load_span,
    )
    session.search(
        bm25_candidates=[_candidate("anchor")],
        dense_candidates=[_candidate("dense")],
    )

    assert session.snapshot().eligible_open_docids == ("anchor", "dense")
    opened = session.open_evidence(
        "anchor", obligation_query="grant code in acknowledgments"
    )
    assert opened.outcome == "opened"
    assert "target answer" in (opened.content or "")
    assert calls == [("anchor", "grant code in acknowledgments")]


def test_obligation_policy_requires_a_nonblank_obligation() -> None:
    session = ProgressiveDisclosureSession(
        run_id="run-obligation",
        policy=ProgressiveDisclosurePolicy(
            anchor_count=1,
            dense_lead_count=1,
            anchor_token_cap=16,
            lead_token_cap=4,
            open_token_cap=16,
            maximum_open_calls=1,
            total_evidence_ingress_token_budget=512,
            open_evidence_ingress_token_budget=16,
            anchor_open_policy="reopen_with_obligation",
            open_content_policy="answer_obligation_window_v0",
        ),
        tokenizer=WordTokenizer(),
        document_loader=lambda _docid: "unused",
        obligation_document_loader=lambda _docid, _query: "target",
    )
    session.search(
        bm25_candidates=[_candidate("anchor")],
        dense_candidates=[_candidate("dense")],
    )
    with pytest.raises(ValueError, match="obligation_query"):
        session.open_evidence("anchor")
