from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from deepresearch_harness.research_loop import ResearchLoopCheckpoint


def _payload() -> dict[str, object]:
    return {
        "loop_id": "edr-v0",
        "status": "implementing",
        "problem": "Relevant evidence is present but the answer is not verified.",
        "observed_evidence": [{"path": "runs/decision.json", "sha256": "a" * 64}],
        "simplest_baseline": "Pi v10 with eight unpartitioned searches",
        "hypothesis": "Reserve two searches for unresolved answer-critical evidence.",
        "changed_mechanisms": ["evidence-debt search reserve"],
        "anti_claim": "This does not change model parameters or prove capability gain.",
        "framework_comparisons": [
            {
                "framework": "DeerFlow",
                "official_source_url": "https://github.com/bytedance/deer-flow",
                "inspected_on": "2026-08-15",
                "relevant_mechanisms": ["sub-agent context isolation"],
                "current_gap": "No debt-linked stopping contract.",
                "disposition": "defer",
            },
            {
                "framework": "Open Deep Research",
                "official_source_url": "https://github.com/langchain-ai/open_deep_research",
                "inspected_on": "2026-08-15",
                "relevant_mechanisms": ["research compression"],
                "current_gap": "Higher-complexity supervisor is not justified by this failure.",
                "disposition": "borrow_boundary",
            },
            {
                "framework": "GPT Researcher",
                "official_source_url": "https://github.com/assafelovic/gpt-researcher",
                "inspected_on": "2026-08-15",
                "relevant_mechanisms": ["planner executor publisher"],
                "current_gap": "Parallel breadth does not verify the final answer relation.",
                "disposition": "benchmark_later",
            },
        ],
        "frozen_controls": ["model", "retriever", "search cap", "output cap"],
        "budget": {
            "maximum_development_queries": 5,
            "maximum_provider_cost_usd": 0.5,
            "maximum_search_calls_per_query": 8,
            "maximum_output_tokens_per_query": 10000,
            "sealed_holdout_access": "forbidden",
        },
        "multi_agent_status": "deferred",
        "next_action": "Run offline debt calibration.",
        "active_paid_calls": 0,
        "retained_services": ["qwen3-32b-bf16-judge@127.0.0.1:18015"],
        "task_owned_processes": "named_and_retained",
        "claim_boundary": "Outcome-selected development calibration only.",
    }


def test_active_loop_is_valid_but_not_pause_ready() -> None:
    checkpoint = ResearchLoopCheckpoint.model_validate(_payload())

    audit = checkpoint.pause_audit()

    assert not audit.ready_to_pause
    assert audit.blockers == ("loop_not_closed", "measured_result_missing", "decision_missing")


def test_closed_loop_requires_measured_decision() -> None:
    payload = _payload()
    payload["status"] = "closed"

    with pytest.raises(ValidationError, match="measured result and decision"):
        ResearchLoopCheckpoint.model_validate(payload)


def test_pause_ready_requires_closed_loop_and_no_paid_calls() -> None:
    payload = _payload()
    payload.update(
        {
            "status": "closed",
            "measured_result": {"path": "runs/result.json", "sha256": "b" * 64},
            "decision": "reject",
        }
    )
    checkpoint = ResearchLoopCheckpoint.model_validate(payload)
    assert checkpoint.pause_audit().ready_to_pause

    payload["active_paid_calls"] = 1
    checkpoint = ResearchLoopCheckpoint.model_validate(payload)
    assert checkpoint.pause_audit().blockers == ("paid_calls_still_active",)


def test_loop_rejects_duplicate_frameworks_and_unbounded_multi_agent() -> None:
    payload = _payload()
    duplicate = deepcopy(payload["framework_comparisons"][0])
    duplicate["official_source_url"] = "https://github.com/bytedance/deer-flow/tree/main"
    payload["framework_comparisons"][2] = duplicate
    with pytest.raises(ValidationError, match="distinct frameworks"):
        ResearchLoopCheckpoint.model_validate(payload)

    payload = _payload()
    payload["multi_agent_status"] = "eligible"
    with pytest.raises(ValidationError, match="observed trigger"):
        ResearchLoopCheckpoint.model_validate(payload)


def test_zero_call_offline_loop_has_an_exact_zero_budget() -> None:
    payload = _payload()
    payload["budget"] = {
        "maximum_development_queries": 5,
        "maximum_provider_cost_usd": 0,
        "maximum_search_calls_per_query": 0,
        "maximum_output_tokens_per_query": 0,
        "sealed_holdout_access": "forbidden",
    }
    checkpoint = ResearchLoopCheckpoint.model_validate(payload)
    assert checkpoint.budget.maximum_provider_cost_usd == 0
