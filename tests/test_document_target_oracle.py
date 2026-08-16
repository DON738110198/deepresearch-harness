from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.document_target_oracle import (
    load_document_target_oracle_registration,
    run_document_target_oracle,
    select_obligation_target_slate,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run_trace() -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Question: I am seeking the project named in the "
                            "acknowledgments section of a Purdue thesis. The author "
                            "later spoke at a symposium.\n\n"
                            "Your response should be cited."
                        ),
                    }
                ],
            }
        ],
        "search_calls": [
            {
                "query": "symposium speaker biography",
                "outcome": "ok",
                "results": [
                    {
                        "docid": "noise-anchor",
                        "snippet": "[BM25 anchor: full evidence]\nNoise",
                    },
                    {
                        "docid": "noise-lead",
                        "snippet": "[Dense lead preview; docid=noise-lead]\nNoise",
                    },
                ],
            },
            {
                "query": "Purdue thesis acknowledgments project",
                "outcome": "ok",
                "results": [
                    {
                        "docid": "answer-anchor",
                        "snippet": "[BM25 anchor: full evidence]\nCandidate",
                    },
                    {
                        "docid": "answer-lead",
                        "snippet": "[Dense lead preview; docid=answer-lead]\nCandidate",
                    },
                ],
            },
        ],
    }


def test_selector_conditions_on_obligation_and_balances_channels() -> None:
    slate = select_obligation_target_slate(
        _run_trace(), maximum_search_calls=1, slots_per_channel=1
    )
    assert slate.selected_search_calls[0].search_call_index == 2
    assert [target.docid for target in slate.targets] == [
        "answer-anchor",
        "answer-lead",
    ]
    assert [target.channel for target in slate.targets] == [
        "bm25_anchor",
        "dense_lead",
    ]


def _write_registration(root: Path) -> Path:
    benchmark = root / "benchmarks" / "probe"
    run_root = root / "runs" / "candidate" / "q1"
    benchmark.mkdir(parents=True)
    run_root.mkdir(parents=True)
    gold = root / "runs" / "gold.json"
    frozen = benchmark / "frozen.txt"
    gold.write_text(
        json.dumps({"rows": [{"query_id": "q1", "gold_docids": ["answer-lead"]}]}),
        encoding="utf-8",
    )
    frozen.write_text("frozen", encoding="utf-8")
    (run_root / "run.json").write_text(json.dumps(_run_trace()), encoding="utf-8")
    registration = benchmark / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "document-target-oracle-registration-v0",
                "status": "posthoc_registered_for_new_failure_cluster",
                "registered_at": "2026-08-16T00:00:00+00:00",
                "purpose": "test",
                "candidate_run_root": "runs/candidate",
                "gold_slice": {"path": "runs/gold.json", "sha256": _hash(gold)},
                "query_ids": ["q1"],
                "selector_id": "obligation_channel_slate_v0",
                "maximum_search_calls": 1,
                "slots_per_channel": 1,
                "frozen_artifacts": [
                    {"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}
                ],
                "acceptance": {
                    "minimum_gold_target_hit_cases": 1,
                    "maximum_targets_per_case": 2,
                },
                "sealed_holdout_access": "forbidden",
                "claim_boundary": "test only",
            }
        ),
        encoding="utf-8",
    )
    return registration


def test_oracle_uses_recorded_results_without_new_calls(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    registration = _write_registration(root)
    result = run_document_target_oracle(
        registration_path=registration,
        output_path=root / "runs" / "oracle" / "result.json",
    )
    assert result.decision == "pass"
    assert result.gold_target_hit_cases == 1
    assert result.provider_calls == 0
    assert result.new_search_calls == 0
    assert result.document_open_calls == 0
    assert result.judge_calls == 0


def test_registration_rejects_changed_gold(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    registration = _write_registration(root)
    (root / "runs" / "gold.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        load_document_target_oracle_registration(registration)
