from __future__ import annotations

import pytest

from deepresearch_harness.development_failure_profile import (
    FailureRouting,
    classify_failure,
    select_next_layer,
)


def _routing() -> FailureRouting:
    return FailureRouting(
        minimum_scored_wrong_cases=30,
        answer_contract_failure_share_all_percent=10,
        reference_document_not_retrieved_share_wrong_percent=60,
        reference_document_retrieved_answer_wrong_share_wrong_percent=60,
    )


def test_failure_classification_uses_registered_precedence() -> None:
    assert classify_failure(
        judge_correct=True,
        answer_schema_complete=False,
        exact_answer_extracted=True,
        evidence_recall=1,
        gold_recall=1,
    ) == "answer_contract_failure"
    assert classify_failure(
        judge_correct=True,
        answer_schema_complete=True,
        exact_answer_extracted=True,
        evidence_recall=0,
        gold_recall=0,
    ) == "judge_correct"
    assert classify_failure(
        judge_correct=False,
        answer_schema_complete=True,
        exact_answer_extracted=True,
        evidence_recall=0,
        gold_recall=0,
    ) == "reference_document_not_retrieved"
    assert classify_failure(
        judge_correct=False,
        answer_schema_complete=True,
        exact_answer_extracted=True,
        evidence_recall=0.5,
        gold_recall=0,
    ) == "reference_document_retrieved_answer_wrong"


def test_next_layer_routes_dominant_retrieval_failure() -> None:
    assert select_next_layer(
        query_count=175,
        judge_wrong=100,
        category_counts={"reference_document_not_retrieved": 65},
        wrong_category_counts={"reference_document_not_retrieved": 65},
        routing=_routing(),
    ) == "retrieval_visibility"


def test_answer_contract_route_has_precedence() -> None:
    assert select_next_layer(
        query_count=175,
        judge_wrong=100,
        category_counts={
            "answer_contract_failure": 18,
            "reference_document_not_retrieved": 70,
        },
        wrong_category_counts={
            "answer_contract_failure": 18,
            "reference_document_not_retrieved": 70,
        },
        routing=_routing(),
    ) == "answer_contract"


def test_route_rejects_too_few_wrong_cases() -> None:
    with pytest.raises(ValueError, match="not enough scored wrong"):
        select_next_layer(
            query_count=175,
            judge_wrong=29,
            category_counts={},
            wrong_category_counts={},
            routing=_routing(),
        )
