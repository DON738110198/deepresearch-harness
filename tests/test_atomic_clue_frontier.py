from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.atomic_clue_frontier import (
    RetrievedDocument,
    build_atomic_clue_queries,
    load_atomic_clue_registration,
    run_atomic_clue_frontier,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_atomic_queries_keep_declarative_clues_and_drop_answer_request() -> None:
    question = (
        "I want to identify a person. The person was adopted in the early 1940s. "
        "They were dismissed before starting their own practice. "
        "What was the person's name?"
    )
    assert build_atomic_clue_queries(question) == (
        "The person was adopted in the early 1940s",
        "They were dismissed before starting their own practice",
    )


def _write_registration(root: Path, queries: list[str]) -> Path:
    benchmark_dir = root / "benchmarks" / "probe"
    source_dir = root / "runs" / "source"
    benchmark_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    request = source_dir / "request.json"
    gold = source_dir / "gold.json"
    frozen = benchmark_dir / "frozen.txt"
    request.write_text(
        json.dumps(
            {
                "question": (
                    "The person was adopted in the early 1940s. "
                    "They were dismissed before starting their own practice. "
                    "What was the person's name?"
                )
            }
        ),
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps({"rows": [{"query_id": "q1", "gold_docids": ["gold-doc"]}]}),
        encoding="utf-8",
    )
    frozen.write_text("fixed", encoding="utf-8")
    registration = benchmark_dir / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "atomic-clue-frontier-registration-v0",
                "status": "registered_before_retrieval",
                "registered_at": "2026-08-15T00:00:00+00:00",
                "purpose": "test",
                "cases": [
                    {
                        "query_id": "q1",
                        "request": {"path": "runs/source/request.json", "sha256": _hash(request)},
                        "queries": queries,
                    }
                ],
                "gold_slice": {"path": "runs/source/gold.json", "sha256": _hash(gold)},
                "search_url": "http://127.0.0.1:8768/search",
                "retriever_id": "fixed",
                "max_search_results": 20,
                "splitter_id": "declarative_sentence_frontier_v0",
                "frozen_artifacts": [{"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}],
                "acceptance": {"minimum_gold_hit_cases": 1, "search_failures_must_equal": 0},
                "claim_boundary": "test",
            }
        ),
        encoding="utf-8",
    )
    return registration


def test_registration_rejects_manually_changed_query(tmp_path: Path) -> None:
    registration = _write_registration(tmp_path / "repo", ["manually edited query"])
    with pytest.raises(ValueError, match="deterministic splitter"):
        load_atomic_clue_registration(registration)


def test_offline_frontier_persists_snippets_and_scores_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    queries = [
        "The person was adopted in the early 1940s",
        "They were dismissed before starting their own practice",
    ]
    registration = _write_registration(root, queries)

    def fake_search(_url: str, _run_id: str, query: str, _timeout: int):
        docid = "gold-doc" if "dismissed" in query else "noise"
        return ((RetrievedDocument(docid=docid, score=1.0, snippet="saved evidence"),), 2)

    monkeypatch.setattr("deepresearch_harness.atomic_clue_frontier._search", fake_search)
    result = run_atomic_clue_frontier(
        registration_path=registration,
        output_path=root / "runs" / "frontier" / "result.json",
    )
    assert result.decision == "pass"
    assert result.search_calls == 2
    assert result.gold_hit_cases == 1
    assert result.items[0].searches[1].documents[0].snippet == "saved evidence"
