from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.repeat_execution_audit import (
    RepeatExecutionAuditRegistration,
    RepeatExecutionAuditResult,
)


def _artifact(name: str) -> dict[str, str]:
    return {"path": name, "sha256": "a" * 64}


def test_registration_rejects_more_failures_than_calls() -> None:
    payload = {
        "status": "registered_post_failure_for_integrity_audit",
        "registered_at": "2026-08-16T00:00:00Z",
        "purpose": "fixture",
        "original_repeat_registration": _artifact("original"),
        "recovery_registration": _artifact("recovery"),
        "invalid_arm_summary": _artifact("summary"),
        "invalid_arm_diagnostic": _artifact("diagnostic"),
        "failed_baseline_attempt_summary": _artifact("failed"),
        "invalid_arm_run_root": "runs/invalid",
        "search_url": "http://127.0.0.1:8769/search",
        "expected_retriever_id": "candidate",
        "expected_query_count": 25,
        "expected_total_search_calls": 8,
        "expected_transport_failure_calls": 9,
        "expected_invalid_arm_recorded_cost_usd": 0.1,
        "known_unobservable_provider_attempts": 2,
        "provider_attempt_policy_violations": 1,
        "frozen_artifacts": [_artifact("source")],
        "claim_boundary": "execution only",
    }
    with pytest.raises(ValidationError, match="cannot exceed"):
        RepeatExecutionAuditRegistration.model_validate(payload)


def test_result_cannot_turn_all_passed_gates_into_execution_rejection() -> None:
    payload = {
        "created_at": "2026-08-16T00:00:00Z",
        "registration_sha256": "a" * 64,
        "query_count": 1,
        "summary_reported_succeeded": 1,
        "raw_run_reported_succeeded": 1,
        "total_search_calls": 1,
        "transport_failure_calls": 1,
        "successful_search_calls": 0,
        "transport_failure_details": {"fetch failed": 1},
        "recorded_invalid_arm_provider_cost_usd": 0.1,
        "known_unobservable_provider_attempts": 2,
        "provider_attempt_policy_violations": 1,
        "search_service_health_error": "unavailable",
        "gates": [
            {
                "gate_id": "fixture",
                "observed": 0,
                "threshold": 0,
                "passed": True,
            }
        ],
        "claim_boundary": "execution only",
    }
    with pytest.raises(ValidationError, match="at least one failed gate"):
        RepeatExecutionAuditResult.model_validate(payload)
