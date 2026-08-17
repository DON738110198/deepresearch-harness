from __future__ import annotations

from deepresearch_harness.gold_evidence_causal_funnel import (
    classify_causal_failure,
    queue_counts,
)


def test_causal_categories_are_mutually_exclusive_with_frozen_precedence() -> None:
    assert classify_causal_failure(
        answer_contract_failure=True,
        gold_arrived=False,
        supporting_arrived=False,
        gold_span_complete=False,
        cited_answer_complete=False,
    ) == "answer_contract_failure"
    assert classify_causal_failure(
        answer_contract_failure=False,
        gold_arrived=False,
        supporting_arrived=False,
        gold_span_complete=False,
        cited_answer_complete=False,
    ) == "no_reference_arrival"
    assert classify_causal_failure(
        answer_contract_failure=False,
        gold_arrived=False,
        supporting_arrived=True,
        gold_span_complete=False,
        cited_answer_complete=False,
    ) == "supporting_evidence_only_arrived"
    assert classify_causal_failure(
        answer_contract_failure=False,
        gold_arrived=True,
        supporting_arrived=True,
        gold_span_complete=False,
        cited_answer_complete=False,
    ) == "gold_arrived_gold_span_incomplete"
    assert classify_causal_failure(
        answer_contract_failure=False,
        gold_arrived=True,
        supporting_arrived=False,
        gold_span_complete=True,
        cited_answer_complete=False,
    ) == "gold_arrived_answer_visible_uncited"
    assert classify_causal_failure(
        answer_contract_failure=False,
        gold_arrived=True,
        supporting_arrived=False,
        gold_span_complete=True,
        cited_answer_complete=True,
    ) == "gold_arrived_answer_visible_cited_wrong"


def test_queues_do_not_treat_supporting_only_arrival_as_exposure() -> None:
    queues = queue_counts(
        {
            "answer_contract_failure": 3,
            "no_reference_arrival": 39,
            "supporting_evidence_only_arrived": 18,
            "gold_arrived_gold_span_incomplete": 30,
            "gold_arrived_answer_visible_uncited": 11,
            "gold_arrived_answer_visible_cited_wrong": 9,
        }
    )
    assert queues.retrieval_or_evidence_frontier == 57
    assert queues.evidence_exposure == 30
    assert queues.evidence_selection == 11
    assert queues.synthesis_verification == 9
    assert queues.answer_contract == 3
