from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.corpus_answerability import (
    load_corpus_answerability_registration,
    run_corpus_answerability_audit,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_registration(root: Path) -> Path:
    benchmark = root / "benchmarks" / "probe"
    index = root / "runs" / "index"
    benchmark.mkdir(parents=True)
    index.mkdir(parents=True)
    stability = root / "runs" / "stability.json"
    gold = root / "runs" / "gold.json"
    frozen = benchmark / "frozen.txt"
    stability.write_text(
        json.dumps(
            {
                "cases": [
                    {"query_id": "q1", "category": "persistent_retrieval_miss"},
                    {"query_id": "q2", "category": "persistent_retrieval_miss"},
                    {"query_id": "q3", "category": "stable_correct"},
                ]
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
                        "question": "Which code?",
                        "answer": "CODE-123",
                        "gold_docids": ["doc-1"],
                    },
                    {
                        "query_id": "q2",
                        "question": "Which name?",
                        "answer": "Missing Name",
                        "gold_docids": ["doc-2"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    frozen.write_text("frozen", encoding="utf-8")
    registration = benchmark / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": (
                    "persistent-miss-corpus-answerability-registration-v0"
                ),
                "status": "posthoc_registered_failure_cluster",
                "registered_at": "2026-08-16T00:00:00+00:00",
                "purpose": "test",
                "stability_audit": {
                    "path": "runs/stability.json",
                    "sha256": _hash(stability),
                },
                "gold_slice": {"path": "runs/gold.json", "sha256": _hash(gold)},
                "query_ids": ["q1", "q2"],
                "document_index_path": "runs/index",
                "frozen_artifacts": [
                    {"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}
                ],
                "acceptance": {"minimum_answerable_cases": 1},
                "sealed_holdout_access": "forbidden",
                "claim_boundary": "test only",
            }
        ),
        encoding="utf-8",
    )
    return registration


def test_answerability_audit_separates_answer_present_and_missing_document(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    registration = _write_registration(root)
    result = run_corpus_answerability_audit(
        registration_path=registration,
        output_path=root / "runs" / "audit" / "result.json",
        document_loader={"doc-1": "The answer is CODE-123."}.get,
    )
    assert result.decision == "retrieval_layer_confirmed"
    assert result.answerable_cases == 1
    assert result.literal_absent_cases == 1
    assert result.missing_gold_documents == 1
    assert result.provider_calls == 0
    assert result.search_calls == 0
    assert result.judge_calls == 0


def test_registration_requires_the_complete_failure_cluster(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    registration = _write_registration(root)
    payload = json.loads(registration.read_text(encoding="utf-8"))
    payload["query_ids"] = ["q1"]
    registration.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="full persistent-miss cluster"):
        load_corpus_answerability_registration(registration)
