from __future__ import annotations

import json
from pathlib import Path

from deepresearch_harness.evidence_selectivity_probe import (
    _paired_outcome,
    _run_metrics,
)


def test_run_metrics_tracks_duplicate_ingress_and_first_relevant_call(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "schema_version": "pi-browsecomp-run-v0",
                "adapter_version": "pi-browsecomp-v8",
                "pi_version": "0.84.1",
                "run_id": "run-1",
                "query_id": "q1",
                "model": "deepseek-v4-flash",
                "thinking_level": "high",
                "max_search_results": 20,
                "control_policy": "standard",
                "system_prompt": "",
                "prompt_sha256": "0" * 64,
                "started_at": "2026-08-14T00:00:00+00:00",
                "latency_ms": 123,
                "status": "succeeded",
                "stop_reason": "completed",
                "answer_text": "answer",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 110,
                    "cost_usd": 0.01,
                },
                "search_calls": [
                    {
                        "query": "Alpha  Beta",
                        "outcome": "ok",
                        "latency_ms": 1,
                        "results": [
                            {"docid": "d1", "score": 1.0, "snippet": "one"},
                            {"docid": "d2", "score": 0.9, "snippet": "two"},
                        ],
                    },
                    {
                        "query": "alpha beta",
                        "outcome": "ok",
                        "latency_ms": 1,
                        "results": [
                            {"docid": "d2", "score": 1.0, "snippet": "two"},
                            {"docid": "d3", "score": 0.8, "snippet": "three"},
                        ],
                    },
                ],
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    metrics = _run_metrics(run_path, {"d3"}, {"d2"})

    assert metrics.search_calls == 2
    assert metrics.unique_search_queries == 1
    assert metrics.repeated_query_rate_percent == 50.0
    assert metrics.result_slots == 4
    assert metrics.unique_docids == 3
    assert metrics.duplicate_result_rate_percent == 25.0
    assert metrics.first_relevant_call == 2
    assert metrics.minimum_relevant_rank == 2
    assert metrics.first_gold_call == 1


def test_paired_outcome_is_directional() -> None:
    assert _paired_outcome(False, True) == "candidate_improvement"
    assert _paired_outcome(True, False) == "candidate_regression"
    assert _paired_outcome(True, True) == "both_correct"
    assert _paired_outcome(False, False) == "both_incorrect"
