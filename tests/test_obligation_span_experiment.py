from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.obligation_span_experiment import (
    decide_obligation_span_calibration,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_decision_requires_opened_answer_span_and_judge_repair(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    benchmark = root / "benchmarks" / "probe"
    source = root / "runs" / "source"
    candidate = root / "runs" / "candidate"
    judge_dir = root / "runs" / "judge" / "evaluation"
    benchmark.mkdir(parents=True)
    source.mkdir(parents=True)
    (candidate / "q1").mkdir(parents=True)
    (judge_dir / "results").mkdir(parents=True)
    artifacts: dict[str, Path] = {}
    for name, payload in {
        "queries.json": {"queries": [{"query_id": "q1"}]},
        "oracle.json": {"decision": "pass"},
        "baseline_summary.json": {"query_count": 1},
        "baseline_judge.json": {
            "observations": [{"query_id": "q1", "correct": False}]
        },
        "frozen.txt": "fixed",
    }.items():
        path = source / name
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload),
            encoding="utf-8",
        )
        artifacts[name] = path
    run = candidate / "q1" / "run.json"
    run.write_text(
        json.dumps(
            {
                "evidence_open_calls": [
                    {
                        "docid": "doc-1",
                        "outcome": "ok",
                        "result": {"outcome": "opened", "content": "grant CODE-123"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = candidate / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "query_id": "q1",
                        "run_path": "runs/candidate/q1/run.json",
                    }
                ],
                "schema_complete": 1,
                "failed": 0,
                "budget_exhausted": 0,
                "total_cost_usd": 0.01,
            }
        ),
        encoding="utf-8",
    )
    candidate_judge = judge_dir / "summary.json"
    candidate_judge.write_text(
        json.dumps(
            {
                "observations": [{"query_id": "q1", "correct": True}],
                "parse_failures": 0,
                "request_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    (judge_dir / "results" / "q1_eval.json").write_text(
        json.dumps({"correct_answer": "CODE-123"}), encoding="utf-8"
    )
    registration = benchmark / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "obligation-span-calibration-registration-v0",
                "status": "registered_pre_provider",
                "registered_at": "2026-08-16T00:00:00+00:00",
                "purpose": "test",
                "query_artifact": {"path": "runs/source/queries.json", "sha256": _hash(artifacts["queries.json"])},
                "query_ids": ["q1"],
                "oracle_result": {"path": "runs/source/oracle.json", "sha256": _hash(artifacts["oracle.json"])},
                "baseline_summary": {"path": "runs/source/baseline_summary.json", "sha256": _hash(artifacts["baseline_summary.json"])},
                "baseline_judge": {"path": "runs/source/baseline_judge.json", "sha256": _hash(artifacts["baseline_judge.json"])},
                "fixed_contract": {"model": "fixed"},
                "frozen_artifacts": [{"path": "runs/source/frozen.txt", "sha256": _hash(artifacts["frozen.txt"])}],
                "acceptance": {
                    "minimum_opened_span_cases": 1,
                    "minimum_answer_bearing_span_cases": 1,
                    "minimum_judge_correct_cases": 1,
                    "minimum_judge_correct_delta": 1,
                    "schema_complete_must_equal": 1,
                    "generation_failures_must_equal": 0,
                    "judge_parse_failures_must_equal": 0,
                    "judge_request_failures_must_equal": 0,
                    "maximum_provider_cost_usd": 0.1
                },
                "sealed_holdout_access": "forbidden",
                "claim_boundary": "test",
            }
        ),
        encoding="utf-8",
    )
    result = decide_obligation_span_calibration(
        registration_path=registration,
        candidate_summary_path=summary,
        candidate_judge_path=candidate_judge,
        output_path=root / "runs" / "decision" / "result.json",
    )
    assert result.decision == "advance_to_fresh_slice"
    assert result.opened_span_cases == 1
    assert result.answer_bearing_span_cases == 1
    assert result.judge_correct_delta == 1
