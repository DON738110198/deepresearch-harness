from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.tool_loop_governor_decision import (
    ToolLoopGovernorDecision,
)


GATE_IDS = (
    "succeeded",
    "failed",
    "budget_exhausted",
    "output_budget_overshoot",
    "research_budget_trace_count",
    "maximum_search_calls_per_query",
    "total_search_calls",
    "judge_parse_failures",
    "judge_request_failures",
    "judge_correct",
    "evidence_recall_percent",
    "total_token_ratio",
    "provider_cost_ratio",
    "generation_cost",
    "evidence_ingress_tokens",
    "successful_open_calls",
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "browsecomp-plus-query-aware-preview-fresh5-decision-v0",
        "created_at": "2026-08-15T00:00:00+00:00",
        "decision": "promote",
        "candidate_layer": "query_aware_preview_plus_tool_loop_governor_v0",
        "query_count": 1,
        "baseline_judge_correct": 1,
        "candidate_judge_correct": 1,
        "baseline_evidence_recall_percent": 50.0,
        "candidate_evidence_recall_percent": 75.0,
        "evidence_recall_delta_pp": 25.0,
        "baseline_search_calls": 10,
        "candidate_search_calls": 8,
        "baseline_total_tokens": 100,
        "candidate_total_tokens": 50,
        "candidate_to_baseline_total_token_ratio": 0.5,
        "baseline_cost_usd": 0.1,
        "candidate_cost_usd": 0.05,
        "candidate_to_baseline_provider_cost_ratio": 0.5,
        "gates": [
            {
                "gate_id": gate_id,
                "observed": 1,
                "operator": "ge",
                "threshold": 1,
                "passed": True,
            }
            for gate_id in GATE_IDS
        ],
        "query_traces": [
            {
                "query_id": "q1",
                "judge_correct": True,
                "evidence_recall": 0.75,
                "search_calls": 8,
                "search_budget_exhausted": True,
                "blocked_search_calls": 0,
                "successful_open_calls": 1,
                "total_tokens": 50,
                "cost_usd": 0.05,
            }
        ],
        "failure_mode": "none",
        "next_action": "preregister_paired_25_query_confirmation",
        "sources": {},
        "claim_boundary": "Development diagnostic only.",
    }


def test_query_aware_preview_decision_can_promote() -> None:
    decision = ToolLoopGovernorDecision.model_validate(_payload())

    assert decision.decision == "promote"
    assert decision.candidate_layer == (
        "query_aware_preview_plus_tool_loop_governor_v0"
    )


def test_decision_cannot_promote_with_a_failed_gate() -> None:
    payload = _payload()
    payload["gates"][0]["passed"] = False

    with pytest.raises(ValidationError, match="decision differs from its gates"):
        ToolLoopGovernorDecision.model_validate(payload)
