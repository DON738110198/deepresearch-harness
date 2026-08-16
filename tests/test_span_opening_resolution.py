from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.span_opening_resolution import SpanOpeningResolution


def test_resolution_cannot_continue_when_an_effectiveness_gate_failed() -> None:
    trial = {
        "trial_id": "trial-1",
        "baseline_judge_correct": 1,
        "candidate_judge_correct": 0,
        "judge_correct_delta": -1,
        "baseline_normalized_exact": 1,
        "candidate_normalized_exact": 0,
        "normalized_exact_delta": -1,
        "baseline_evidence_recall_percent": 50.0,
        "candidate_evidence_recall_percent": 40.0,
        "evidence_recall_delta_pp": -10.0,
        "baseline_search_calls": 8,
        "candidate_search_calls": 8,
        "baseline_total_tokens": 100,
        "candidate_total_tokens": 90,
        "baseline_recorded_cost_usd": 0.1,
        "candidate_recorded_cost_usd": 0.09,
    }
    trial2 = {**trial, "trial_id": "trial-2"}
    payload = {
        "created_at": "2026-08-16T00:00:00Z",
        "decision": "continue_clean_repetition",
        "query_count_per_trial": 1,
        "trials": [trial, trial2],
        "aggregate_baseline_judge_correct": 2,
        "aggregate_candidate_judge_correct": 0,
        "aggregate_judge_correct_delta": -2,
        "aggregate_baseline_normalized_exact": 2,
        "aggregate_candidate_normalized_exact": 0,
        "aggregate_normalized_exact_delta": -2,
        "baseline_mean_evidence_recall_percent": 50.0,
        "candidate_mean_evidence_recall_percent": 40.0,
        "mean_evidence_recall_delta_pp": -10.0,
        "total_token_ratio": 0.9,
        "recorded_provider_cost_ratio": 0.9,
        "combined_recorded_provider_cost_usd": 0.38,
        "known_unobservable_provider_attempts": 2,
        "gates": [
            {
                "gate_id": "judge",
                "observed": -2,
                "operator": "ge",
                "threshold": 0,
                "passed": False,
            }
        ],
        "sources": {"fixture": {"path": "runs/a.json", "sha256": "a" * 64}},
        "claim_boundary": "development only",
    }
    with pytest.raises(ValidationError, match="resolution differs"):
        SpanOpeningResolution.model_validate(payload)
