from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from deepresearch_harness.evidence_debt_experiment import (
    EvidenceDebtFreshComparisonDecision,
    EvidenceDebtLiveCalibrationDecision,
    ExperimentGate,
)


def _source() -> dict[str, str]:
    return {"path": "runs/result.json", "sha256": "a" * 64}


def _live_payload() -> dict[str, object]:
    return {
        "created_at": "2026-08-15T00:00:00+00:00",
        "decision": "pass",
        "query_count": 1,
        "succeeded": 1,
        "failed": 0,
        "budget_exhausted": 0,
        "schema_complete": 1,
        "audit_trace_count": 1,
        "audit_parse_failures": 0,
        "repair_triggered_regressions": 1,
        "repair_triggered_improvements": 0,
        "repaired_regression_corrections": 1,
        "preserved_improvements": 0,
        "candidate_judge_correct": 1,
        "candidate_judge_accuracy_percent": 100.0,
        "candidate_exact_match_percent": 100.0,
        "candidate_evidence_recall_percent": 100.0,
        "total_search_calls": 8,
        "total_tokens": 100,
        "total_cost_usd": 0.01,
        "rows": [
            {
                "query_id": "1",
                "prior_outcome": "candidate_regression",
                "prior_judge_correct": False,
                "audit_status": "repair_requested",
                "repair_call_count": 1,
                "candidate_judge_correct": True,
                "transition": "regression_repaired",
                "search_calls": 8,
                "total_tokens": 100,
                "cost_usd": 0.01,
            }
        ],
        "gates": [
            {
                "gate_id": "fixture",
                "observed": 1,
                "operator": "eq",
                "threshold": 1,
                "passed": True,
            }
        ],
        "sources": {"fixture": _source()},
        "next_action": "preregister_fresh_development_comparison",
    }


def _fresh_payload() -> dict[str, object]:
    return {
        "created_at": "2026-08-15T00:00:00+00:00",
        "decision": "promote",
        "query_count": 1,
        "baseline_judge_correct": 0,
        "candidate_judge_correct": 1,
        "baseline_judge_accuracy_percent": 0.0,
        "candidate_judge_accuracy_percent": 100.0,
        "judge_accuracy_delta_pp": 100.0,
        "paired_improvements": 1,
        "paired_regressions": 0,
        "baseline_exact_match_percent": 0.0,
        "candidate_exact_match_percent": 100.0,
        "baseline_evidence_recall_percent": 50.0,
        "candidate_evidence_recall_percent": 75.0,
        "evidence_recall_delta_pp": 25.0,
        "baseline_search_calls": 8,
        "candidate_search_calls": 8,
        "search_call_ratio": 1.0,
        "baseline_total_tokens": 100,
        "candidate_total_tokens": 100,
        "total_token_ratio": 1.0,
        "baseline_cost_usd": 0.01,
        "candidate_cost_usd": 0.01,
        "provider_cost_ratio": 1.0,
        "combined_generation_cost_usd": 0.02,
        "candidate_audit_trace_count": 1,
        "candidate_audit_parse_failures": 0,
        "candidate_repair_call_count": 1,
        "rows": [
            {
                "query_id": "1",
                "baseline_judge_correct": False,
                "candidate_judge_correct": True,
                "paired_outcome": "candidate_improvement",
                "baseline_evidence_recall": 0.5,
                "candidate_evidence_recall": 0.75,
                "baseline_search_calls": 8,
                "candidate_search_calls": 8,
                "candidate_audit_status": "repair_requested",
                "candidate_repair_call_count": 1,
            }
        ],
        "gates": [
            {
                "gate_id": "fixture",
                "observed": 1,
                "operator": "eq",
                "threshold": 1,
                "passed": True,
            }
        ],
        "sources": {"fixture": _source()},
        "next_action": "preregister_fresh25_confirmation",
    }


def test_gate_rejects_a_forged_pass_flag() -> None:
    with pytest.raises(ValidationError, match="gate result differs"):
        ExperimentGate(
            gate_id="fixture",
            observed=0,
            operator="ge",
            threshold=1,
            passed=True,
        )


def test_live_calibration_decision_must_match_gates() -> None:
    result = EvidenceDebtLiveCalibrationDecision.model_validate(_live_payload())
    assert result.decision == "pass"

    payload = deepcopy(_live_payload())
    payload["decision"] = "reject"
    payload["next_action"] = "diagnose_outcome_selected_calibration"
    with pytest.raises(ValidationError, match="differs from its gates"):
        EvidenceDebtLiveCalibrationDecision.model_validate(payload)


def test_fresh_comparison_decision_must_match_gates() -> None:
    result = EvidenceDebtFreshComparisonDecision.model_validate(_fresh_payload())
    assert result.decision == "promote"

    payload = deepcopy(_fresh_payload())
    payload["decision"] = "reject"
    payload["next_action"] = "diagnose_fresh_saved_traces"
    with pytest.raises(ValidationError, match="differs from its gates"):
        EvidenceDebtFreshComparisonDecision.model_validate(payload)
