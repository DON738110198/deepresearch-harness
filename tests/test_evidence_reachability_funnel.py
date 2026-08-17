from __future__ import annotations

from deepresearch_harness.evidence_reachability_funnel import (
    _query_ids_sha256,
    classify_reachability,
    extract_cited_docids,
    route_reachability,
)


def test_reachability_classification_is_mutually_exclusive() -> None:
    assert classify_reachability(
        literal_answer_visible=False, literal_answer_cited=False
    ) == "answer_hidden_after_reference_arrival"
    assert classify_reachability(
        literal_answer_visible=True, literal_answer_cited=False
    ) == "answer_visible_reference_uncited"
    assert classify_reachability(
        literal_answer_visible=True, literal_answer_cited=True
    ) == "answer_visible_reference_cited_wrong"


def test_routing_uses_registered_denominators() -> None:
    assert route_reachability(
        answer_hidden=41,
        answer_visible_uncited=20,
        answer_visible_cited=6,
        dominance_percent=60,
    ) == "evidence_exposure_or_opening"
    assert route_reachability(
        answer_hidden=20,
        answer_visible_uncited=30,
        answer_visible_cited=17,
        dominance_percent=60,
    ) == "evidence_selection"
    assert route_reachability(
        answer_hidden=20,
        answer_visible_uncited=17,
        answer_visible_cited=30,
        dominance_percent=60,
    ) == "synthesis_verification"
    assert route_reachability(
        answer_hidden=20,
        answer_visible_uncited=23,
        answer_visible_cited=24,
        dominance_percent=60,
    ) == "mixed_stratified_audit"


def test_citation_parser_accepts_only_known_bracketed_docids() -> None:
    answer = "Supported by [123, 456] and [doc-7]; not by bare 999 or [unknown]."
    assert extract_cited_docids(answer, {"123", "456", "doc-7", "999"}) == (
        "123",
        "456",
        "doc-7",
    )


def test_query_id_hash_uses_real_lf_and_a_final_lf() -> None:
    assert _query_ids_sha256(["10", "2"]) == (
        "eba7437651bd2dabe00aba8388b552da5557f5f7b0fbe2ea2248902e7ffc9cfd"
    )
