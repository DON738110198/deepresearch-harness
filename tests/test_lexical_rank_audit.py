from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.lexical_rank_audit import run_lexical_rank_audit


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_registration(root: Path) -> Path:
    benchmark = root / "benchmarks" / "probe"
    index = root / "runs" / "index"
    run_root = root / "runs" / "baseline"
    benchmark.mkdir(parents=True)
    index.mkdir(parents=True)
    stability = root / "runs" / "stability.json"
    answerability = root / "runs" / "answerability.json"
    gold = root / "runs" / "gold.json"
    frozen = benchmark / "frozen.txt"
    stability.write_text(
        json.dumps(
            {
                "cases": [
                    {"query_id": "q1", "category": "persistent_retrieval_miss"},
                    {"query_id": "q2", "category": "persistent_retrieval_miss"},
                ]
            }
        ),
        encoding="utf-8",
    )
    answerability.write_text(
        json.dumps(
            {
                "decision": "retrieval_layer_confirmed",
                "answerable_cases": 2,
            }
        ),
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "query_id": "q1",
                        "question": "Raw question one?",
                        "answer": "A",
                        "gold_docids": ["gold-1"],
                    },
                    {
                        "query_id": "q2",
                        "question": "Raw question two?",
                        "answer": "B",
                        "gold_docids": ["gold-2"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    for query_id, question, generated in (
        ("q1", "Raw question one?", "generated one"),
        ("q2", "Raw question two?", "generated two"),
    ):
        query_root = run_root / query_id
        query_root.mkdir(parents=True)
        (query_root / "run.json").write_text(
            json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Question: {question}\n\n"
                                        "Your response should be cited."
                                    ),
                                }
                            ],
                        }
                    ],
                    "search_calls": [
                        {"query": generated, "outcome": "ok", "results": [{}]}
                    ],
                }
            ),
            encoding="utf-8",
        )
    frozen.write_text("frozen", encoding="utf-8")
    registration = benchmark / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "persistent-miss-lexical-rank-registration-v0",
                "status": "posthoc_registered_failure_cluster",
                "registered_at": "2026-08-16T00:00:00+00:00",
                "purpose": "test",
                "stability_audit": {
                    "path": "runs/stability.json",
                    "sha256": _hash(stability),
                },
                "corpus_answerability": {
                    "path": "runs/answerability.json",
                    "sha256": _hash(answerability),
                },
                "gold_slice": {"path": "runs/gold.json", "sha256": _hash(gold)},
                "baseline_run_root": "runs/baseline",
                "query_ids": ["q1", "q2"],
                "document_index_path": "runs/index",
                "maximum_rank": 100,
                "frozen_artifacts": [
                    {"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}
                ],
                "acceptance": {
                    "minimum_raw_question_top5_cases": 1,
                    "minimum_generated_query_top100_cases_for_pool_diagnosis": 1,
                },
                "sealed_holdout_access": "forbidden",
                "claim_boundary": "test only",
            }
        ),
        encoding="utf-8",
    )
    return registration


def test_rank_audit_routes_to_raw_question_anchor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    registration = _write_registration(root)
    rankings = {
        "Raw question one?": ["noise", "gold-1"],
        "generated one": ["noise"],
        "Raw question two?": ["noise"],
        "generated two": [*[f"noise-{index}" for index in range(49)], "gold-2"],
    }
    result = run_lexical_rank_audit(
        registration_path=registration,
        output_path=root / "runs" / "audit" / "result.json",
        search=lambda query, limit: rankings[query][:limit],
    )
    assert result.decision == "raw_question_anchor_candidate"
    assert result.raw_question_top5_cases == 1
    assert result.generated_query_top100_cases == 1
    assert result.raw_question_better_cases == 1
    assert result.offline_bm25_queries == 4
    assert result.provider_calls == 0
    assert result.online_search_calls == 0
    assert result.judge_calls == 0
