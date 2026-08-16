from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.evidence_debt_audit_calibration import (
    EvidenceDebtAuditCalibration,
)


def _payload() -> dict[str, object]:
    return {
        "created_at": "2026-08-15T00:00:00+00:00",
        "decision": "pass",
        "regression_count": 1,
        "regression_trigger_count": 1,
        "regression_trigger_recall": 1.0,
        "improvement_count": 1,
        "improvement_false_trigger_count": 0,
        "provider_calls": 0,
        "search_calls": 0,
        "maximum_repair_queries": 2,
        "rows": [
            {
                "query_id": "q1",
                "paired_outcome": "candidate_regression",
                "expected_repair_trigger": True,
                "observed_status": "open",
                "observed_repair_trigger": True,
                "matched_expectation": True,
                "reasons": ["explicit_unresolved_evidence"],
                "repair_queries": ["query"],
                "provider_calls": 0,
                "search_calls": 0,
                "source": {"path": "runs/q1.json", "sha256": "a" * 64},
            },
            {
                "query_id": "q2",
                "paired_outcome": "candidate_improvement",
                "expected_repair_trigger": False,
                "observed_status": "supported",
                "observed_repair_trigger": False,
                "matched_expectation": True,
                "reasons": [],
                "repair_queries": [],
                "provider_calls": 0,
                "search_calls": 0,
                "source": {"path": "runs/q2.json", "sha256": "b" * 64},
            },
        ],
        "gates": [
            {
                "gate_id": "regression_trigger_recall",
                "observed": 1.0,
                "operator": "eq",
                "threshold": 1.0,
                "passed": True,
            },
            {
                "gate_id": "improvement_false_trigger_count",
                "observed": 0,
                "operator": "eq",
                "threshold": 0,
                "passed": True,
            },
            {
                "gate_id": "provider_calls",
                "observed": 0,
                "operator": "eq",
                "threshold": 0,
                "passed": True,
            },
            {
                "gate_id": "search_calls",
                "observed": 0,
                "operator": "eq",
                "threshold": 0,
                "passed": True,
            },
            {
                "gate_id": "maximum_repair_queries",
                "observed": 2,
                "operator": "le",
                "threshold": 2,
                "passed": True,
            },
        ],
        "sources": {},
        "next_action": "implement_typed_pi_v11_checkpoint",
    }


def test_calibration_accepts_consistent_pass() -> None:
    result = EvidenceDebtAuditCalibration.model_validate(_payload())
    assert result.decision == "pass"


def test_calibration_rejects_decision_that_disagrees_with_gates() -> None:
    payload = _payload()
    payload["gates"][0]["passed"] = False
    with pytest.raises(ValidationError, match="decision differs"):
        EvidenceDebtAuditCalibration.model_validate(payload)
