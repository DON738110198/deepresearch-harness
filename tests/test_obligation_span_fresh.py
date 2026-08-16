from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.obligation_span_fresh import (
    FreshCaseResult,
    FreshGate,
    ObligationSpanFreshDecision,
)


def _case(query_id: str, outcome: str) -> FreshCaseResult:
    return FreshCaseResult(
        query_id=query_id,
        baseline_judge_correct=outcome in {"both_correct", "regression"},
        candidate_judge_correct=outcome in {"both_correct", "improvement"},
        paired_outcome=outcome,
        baseline_normalized_exact=False,
        candidate_normalized_exact=True,
        baseline_evidence_recall=0.5,
        candidate_evidence_recall=0.6,
        candidate_open_attempts=1,
        candidate_successful_opens=1,
    )


def test_decision_contract_binds_gate_result_and_next_action() -> None:
    payload = {
        "created_at": "2026-08-16T00:00:00Z",
        "decision": "accept",
        "registration_sha256": "a" * 64,
        "query_count": 2,
        "baseline_judge_correct": 0,
        "candidate_judge_correct": 1,
        "judge_correct_delta": 1,
        "judge_accuracy_delta_pp": 50.0,
        "paired_improvements": 1,
        "paired_regressions": 0,
        "baseline_normalized_exact": 0,
        "candidate_normalized_exact": 1,
        "normalized_exact_delta": 1,
        "baseline_evidence_recall_percent": 50.0,
        "candidate_evidence_recall_percent": 60.0,
        "evidence_recall_delta_pp": 10.0,
        "baseline_search_calls": 10,
        "candidate_search_calls": 9,
        "candidate_open_attempts": 2,
        "candidate_successful_open_calls": 2,
        "candidate_successful_open_cases": 2,
        "baseline_total_tokens": 100,
        "candidate_total_tokens": 90,
        "total_token_ratio": 0.9,
        "baseline_cost_usd": 0.1,
        "candidate_cost_usd": 0.09,
        "provider_cost_ratio": 0.9,
        "combined_recorded_provider_cost_usd": 0.19,
        "known_unobservable_provider_attempts": 0,
        "cost_observability": "complete",
        "cases": [_case("q1", "improvement"), _case("q2", "both_wrong")],
        "gates": [
            FreshGate(
                gate_id="judge_correct_delta",
                observed=1,
                operator="ge",
                threshold=1,
                passed=True,
            )
        ],
        "sources": {"registration": {"path": "registration.json", "sha256": "b" * 64}},
        "next_action": "repeat_on_second_fresh_slice",
        "claim_boundary": "development only",
    }
    assert ObligationSpanFreshDecision.model_validate(payload).decision == "accept"
    payload["next_action"] = "freeze_span_layer_and_rediagnose"
    with pytest.raises(ValidationError, match="next action"):
        ObligationSpanFreshDecision.model_validate(payload)
