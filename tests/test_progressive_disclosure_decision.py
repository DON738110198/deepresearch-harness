from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.progressive_disclosure_decision import (
    ProgressiveDisclosureDecision,
)


GATE_IDS = (
    "succeeded",
    "failed",
    "budget_exhausted",
    "output_budget_overshoot",
    "judge_parse_failures",
    "judge_request_failures",
    "judge_correct",
    "total_token_ratio",
    "provider_cost_ratio",
    "generation_cost",
    "evidence_ingress_tokens",
    "successful_open_calls",
)


def _payload() -> dict[str, object]:
    sources = {
        name: {"path": f"runs/{name}.json", "sha256": "0" * 64}
        for name in (
            "preregistration",
            "candidate_summary",
            "candidate_diagnostic",
            "judge_execution_result",
            "judge_result",
            "baseline_summary",
        )
    }
    gates = [
        {
            "gate_id": gate_id,
            "observed": 0,
            "operator": "le",
            "threshold": 1,
            "passed": gate_id != "total_token_ratio",
        }
        for gate_id in GATE_IDS
    ]
    return {
        "created_at": "2026-08-15T00:00:00+00:00",
        "decision": "reject",
        "query_count": 1,
        "baseline_judge_correct": 0,
        "candidate_judge_correct": 0,
        "judge_correct_delta": 0,
        "normalized_exact_match": 0,
        "evidence_recall_percent": 0.0,
        "baseline_search_calls": 1,
        "candidate_search_calls": 2,
        "candidate_to_baseline_search_call_ratio": 2.0,
        "baseline_total_tokens": 10,
        "candidate_total_tokens": 20,
        "candidate_to_baseline_total_token_ratio": 2.0,
        "baseline_cost_usd": 0.01,
        "candidate_cost_usd": 0.02,
        "candidate_to_baseline_provider_cost_ratio": 2.0,
        "candidate_evidence_ingress_tokens": 2,
        "candidate_open_attempts": 1,
        "candidate_successful_open_calls": 1,
        "gates": gates,
        "query_traces": [
            {
                "query_id": "q1",
                "judge_correct": False,
                "normalized_exact_match": False,
                "evidence_recall": 0.0,
                "search_calls": 2,
                "open_attempts": 1,
                "successful_open_calls": 1,
                "evidence_ingress_tokens": 2,
                "model_requests": 3,
                "input_tokens": 4,
                "cache_read_tokens": 5,
                "output_tokens": 6,
                "total_tokens": 15,
                "cost_usd": 0.02,
            }
        ],
        "failure_mode": "tool_loop_context_replay_without_answer_gain",
        "next_action": "preregister_fresh_slice_tool_loop_governor",
        "sources": sources,
        "claim_boundary": "Development diagnostic only.",
    }


def test_rejected_decision_requires_diagnosed_route() -> None:
    decision = ProgressiveDisclosureDecision.model_validate(_payload())

    assert decision.decision == "reject"
    assert decision.next_action == "preregister_fresh_slice_tool_loop_governor"


def test_decision_cannot_promote_with_a_failed_gate() -> None:
    payload = _payload()
    payload["decision"] = "promote"

    with pytest.raises(ValidationError, match="decision differs from its gates"):
        ProgressiveDisclosureDecision.model_validate(payload)
