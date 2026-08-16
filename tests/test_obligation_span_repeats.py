from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.obligation_span_repeats import (
    ObligationSpanRepeatRegistration,
)


def _arm(prefix: str) -> dict[str, str]:
    return {
        "run_root": f"runs/{prefix}",
        "summary": f"runs/{prefix}/summary.json",
        "diagnostic": f"runs/{prefix}.diagnostic.json",
        "judge": f"runs/{prefix}-judge/result.json",
    }


def test_repeat_registration_rejects_non_alternating_order() -> None:
    payload = {
        "status": "preregistered_before_trials_2_and_3",
        "registered_at": "2026-08-16T00:00:00Z",
        "purpose": "fixture",
        "reference_fresh_registration": {"path": "a", "sha256": "a" * 64},
        "first_trial_decision": {"path": "b", "sha256": "b" * 64},
        "first_trial_bad_case_audit": {"path": "c", "sha256": "c" * 64},
        "query_ids": ["q1"],
        "variants": [
            {
                "name": "baseline",
                "adapter_version": "pi-browsecomp-v10",
                "adapter_runner": {"path": "v10", "sha256": "d" * 64},
                "search_url": "http://127.0.0.1:8768/search",
                "retriever_id": "baseline",
            },
            {
                "name": "candidate",
                "adapter_version": "pi-browsecomp-v14",
                "adapter_runner": {"path": "v14", "sha256": "e" * 64},
                "search_url": "http://127.0.0.1:8769/search",
                "retriever_id": "candidate",
            },
        ],
        "trials": [
            {
                "trial_id": f"trial-{index}",
                "execution_order": "baseline_first",
                "observed_before_registration": index == 1,
                "baseline": _arm(f"t{index}-b"),
                "candidate": _arm(f"t{index}-c"),
            }
            for index in range(1, 4)
        ],
        "acceptance": {
            "trial_count_must_equal": 3,
            "total_generation_failures_must_equal": 0,
            "total_judge_failures_must_equal": 0,
            "maximum_provider_attempt_policy_violations": 0,
            "minimum_candidate_minus_baseline_schema_complete": 0,
            "minimum_aggregate_judge_correct_delta": 3,
            "minimum_noninferior_judge_trials": 2,
            "minimum_aggregate_normalized_exact_delta": 0,
            "minimum_mean_evidence_recall_delta_pp": -1.0,
            "maximum_total_token_ratio": 1.15,
            "maximum_recorded_provider_cost_ratio": 1.25,
            "maximum_new_recorded_provider_cost_usd": 1.0,
        },
        "frozen_artifacts": [{"path": "source", "sha256": "f" * 64}],
        "claim_boundary": "development only",
    }
    with pytest.raises(ValidationError, match="execution order"):
        ObligationSpanRepeatRegistration.model_validate(payload)
