from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.query_aware_confirmation_decision import (
    EXPECTED_GATE_IDS,
    QueryAwareConfirmationDecision,
    _failure_mode,
    _paired_outcome,
)


def _payload() -> dict[str, object]:
    return {
        "created_at": "2026-08-15T00:00:00+00:00",
        "decision": "promote",
        "query_count": 1,
        "baseline_judge_correct": 0,
        "candidate_judge_correct": 1,
        "baseline_judge_accuracy_percent": 0.0,
        "candidate_judge_accuracy_percent": 100.0,
        "judge_accuracy_delta_pp": 100.0,
        "paired_judge_improvements": 1,
        "paired_judge_regressions": 0,
        "baseline_exact_match_percent": 0.0,
        "candidate_exact_match_percent": 100.0,
        "exact_match_delta_pp": 100.0,
        "baseline_evidence_recall_percent": 20.0,
        "candidate_evidence_recall_percent": 80.0,
        "evidence_recall_delta_pp": 60.0,
        "baseline_gold_recall_percent": 20.0,
        "candidate_gold_recall_percent": 80.0,
        "gold_recall_delta_pp": 60.0,
        "baseline_search_calls": 10,
        "candidate_search_calls": 8,
        "candidate_to_baseline_search_call_ratio": 0.8,
        "baseline_total_tokens": 100,
        "candidate_total_tokens": 50,
        "candidate_to_baseline_total_token_ratio": 0.5,
        "baseline_cost_usd": 0.1,
        "candidate_cost_usd": 0.05,
        "candidate_to_baseline_provider_cost_ratio": 0.5,
        "combined_generation_cost_usd": 0.15,
        "gates": [
            {
                "gate_id": gate_id,
                "observed": 1,
                "operator": "ge",
                "threshold": 1,
                "passed": True,
            }
            for gate_id in sorted(EXPECTED_GATE_IDS)
        ],
        "query_traces": [
            {
                "query_id": "q1",
                "paired_judge_outcome": "candidate_improvement",
                "baseline_judge_correct": False,
                "candidate_judge_correct": True,
                "baseline_exact_match": False,
                "candidate_exact_match": True,
                "baseline_evidence_recall": 0.2,
                "candidate_evidence_recall": 0.8,
                "baseline_search_calls": 10,
                "candidate_search_calls": 8,
                "candidate_search_limit_reached": True,
                "candidate_blocked_search_calls": 0,
                "candidate_successful_open_calls": 1,
                "baseline_total_tokens": 100,
                "candidate_total_tokens": 50,
                "baseline_cost_usd": 0.1,
                "candidate_cost_usd": 0.05,
            }
        ],
        "failure_mode": "none",
        "next_action": "preregister_three_trial_stability_confirmation",
        "sources": {},
        "claim_boundary": "Development diagnostic only.",
    }


def test_confirmation_decision_can_promote() -> None:
    decision = QueryAwareConfirmationDecision.model_validate(_payload())

    assert decision.decision == "promote"
    assert len(decision.gates) == 24


def test_confirmation_decision_rejects_inconsistent_paired_counts() -> None:
    payload = _payload()
    payload["paired_judge_improvements"] = 0

    with pytest.raises(ValidationError, match="outcome counts differ"):
        QueryAwareConfirmationDecision.model_validate(payload)


def test_confirmation_decision_rejects_promotion_with_failed_gate() -> None:
    payload = _payload()
    payload["gates"][0]["passed"] = False

    with pytest.raises(ValidationError, match="decision differs from its gates"):
        QueryAwareConfirmationDecision.model_validate(payload)


def test_failure_routing_and_paired_outcomes_are_deterministic() -> None:
    assert _paired_outcome(False, True) == "candidate_improvement"
    assert _paired_outcome(True, False) == "candidate_regression"
    assert _paired_outcome(None, True) == "unscored"
    assert _failure_mode(set()) == "none"
    assert _failure_mode({"judge_accuracy_delta_pp"}) == "quality_gate_failure"
    assert _failure_mode({"total_token_ratio"}) == "resource_gate_failure"
    assert _failure_mode({"candidate_failed"}) == "execution_contract_failure"
    assert _failure_mode(
        {"candidate_failed", "total_token_ratio"}
    ) == "mixed_gate_failure"
