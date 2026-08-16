from __future__ import annotations

import json

import pytest

from deepresearch_harness.atomic_clue_frontier import (
    AtomicClueCaseResult,
    AtomicClueSearch,
    RetrievedDocument,
)
from deepresearch_harness.corpus_bridge_induction import (
    CorpusBridge,
    CorpusBridgeCase,
    CorpusBridgeSlate,
    NearMissDocument,
    NearMissGroup,
    build_corpus_bridge_prompt,
    build_near_miss_packet,
    validate_slate_for_case,
)


def _case() -> CorpusBridgeCase:
    return CorpusBridgeCase(
        query_id="q1",
        question="Who was adopted, dismissed, and later knighted?",
        unresolved_obligation="Identify the exact person.",
        unverified_prior_subject="a lawyer",
        near_miss_packet=(
            NearMissGroup(
                clue_query="dismissed own practice",
                documents=(
                    NearMissDocument(
                        docid="d1",
                        score=2.0,
                        snippet="An architecture practice and its partners are discussed here.",
                    ),
                ),
            ),
        ),
        gold_docids=("gold",),
    )


def _slate() -> CorpusBridgeSlate:
    return CorpusBridgeSlate(
        bridges=(
            CorpusBridge(
                type="domain",
                term="architect",
                source_docids=("d1",),
                query="architect adopted partner dismissed practice knighthood",
            ),
            CorpusBridge(
                type="geography",
                term="New Zealand",
                source_docids=("d1",),
                query="New Zealand adopted partner practice knighthood",
            ),
            CorpusBridge(
                type="entity",
                term="design firm",
                source_docids=("d1",),
                query="design firm adopted partner dismissed practice knighthood",
            ),
        )
    )


def test_near_miss_packet_selects_top_two_and_caps_snippets() -> None:
    documents = tuple(
        RetrievedDocument(docid=f"d{i}", score=3 - i, snippet="x" * 1300)
        for i in range(3)
    )
    frontier = AtomicClueCaseResult(
        query_id="q1",
        status="succeeded",
        searches=(
            AtomicClueSearch(
                query_index=0,
                query="one clue",
                documents=documents,
                gold_hits=(),
                gold_hit_ranks=(),
                latency_ms=1,
            ),
        ),
        unique_docids=("d0", "d1", "d2"),
        gold_hits=(),
        error=None,
    )
    packet = build_near_miss_packet(
        frontier_item=frontier,
        documents_per_clue=2,
        character_cap=1200,
    )
    assert tuple(doc.docid for doc in packet[0].documents) == ("d0", "d1")
    assert all(len(doc.snippet) == 1200 for doc in packet[0].documents)


def test_prompt_contains_only_typed_context_and_cited_slate_validates() -> None:
    case = _case()
    prompt = build_corpus_bridge_prompt(case)
    payload = json.loads(prompt)
    assert set(payload["input"]) == {
        "question",
        "unresolved_obligation",
        "unverified_prior_subject",
        "near_miss_packet",
    }
    assert "JSON" in prompt
    for forbidden in ("prior_exact_answer", "draft_answer", "prior_query", "gold_docids"):
        assert forbidden not in prompt
    validate_slate_for_case(case=case, slate=_slate())


def test_slate_rejects_docid_outside_packet() -> None:
    bad = _slate().model_copy(
        update={
            "bridges": (
                _slate().bridges[0].model_copy(update={"source_docids": ("unknown",)}),
                *_slate().bridges[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="outside the near-miss packet"):
        validate_slate_for_case(case=_case(), slate=bad)
