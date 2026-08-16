from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.multi_query_rrf import (
    build_multi_query_rrf_slate,
    fuse_rankings,
    score_multi_query_rrf_slate,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _query_hash(query: str) -> str:
    return sha256(query.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, dict[str, list[str]], dict[Path, bytes]]:
    benchmark = root / "benchmarks" / "probe"
    baseline = root / "runs" / "baseline"
    index = root / "runs" / "index"
    benchmark.mkdir(parents=True)
    index.mkdir(parents=True)
    (index / "segments_1").write_bytes(b"segments")
    (index / "_1.fdt").write_bytes(b"documents")

    source_index = benchmark / "source-index.json"
    _write_json(
        source_index,
        {
            "source_document_index": {
                "path": "runs/index",
                "document_count": 100,
                "files": [
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _hash(path),
                    }
                    for path in sorted(index.iterdir())
                ],
            }
        },
    )

    queries = {
        "q1": ("alpha branch", "beta branch"),
        "q2": ("gamma branch", "delta branch"),
    }
    for query_id, case_queries in queries.items():
        _write_json(
            baseline / query_id / "run.json",
            {
                "query_id": query_id,
                "search_calls": [
                    {"outcome": "ok", "query": query} for query in case_queries
                ],
            },
        )

    gold = root / "runs" / "gold.json"
    _write_json(
        gold,
        {
            "rows": [
                {"query_id": "q1", "gold_docids": ["gold-1"]},
                {"query_id": "q2", "gold_docids": ["gold-2"]},
            ]
        },
    )
    lexical = root / "runs" / "lexical.json"
    _write_json(
        lexical,
        {
            "decision": "index_representation_diagnosis_required",
            "generated_query_top20_cases": 0,
            "generated_query_top100_cases": 1,
            "items": [
                {
                    "query_id": query_id,
                    "generated_queries": [
                        {"query": query, "query_sha256": _query_hash(query)}
                        for query in case_queries
                    ],
                }
                for query_id, case_queries in queries.items()
            ],
        },
    )
    pivot = root / "runs" / "pivot.json"
    _write_json(pivot, {"decision": "freeze_gold_blind_pivot_branch"})

    registration = benchmark / "registration.json"
    _write_json(
        registration,
        {
            "schema_version": "multi-query-rrf-registration-v0",
            "status": "posthoc_registered_failure_cluster",
            "registered_at": "2026-08-16T00:00:00+00:00",
            "purpose": "test",
            "prerequisites": {
                "lexical_rank_result": {
                    "path": "runs/lexical.json",
                    "sha256": _hash(lexical),
                },
                "rejected_pivot_selector": {
                    "path": "runs/pivot.json",
                    "sha256": _hash(pivot),
                },
                "gold_slice": {"path": "runs/gold.json", "sha256": _hash(gold)},
                "source_index_registration": {
                    "path": "benchmarks/probe/source-index.json",
                    "sha256": _hash(source_index),
                },
            },
            "baseline_run_root": "runs/baseline",
            "query_ids": ["q1", "q2"],
            "query_source": "trial1_recorded_successful_search_calls",
            "retrieval": {
                "document_index_path": "runs/index",
                "analyzer_id": "pyserini_default_english_v1.2.0",
                "bm25_k1": 0.9,
                "bm25_b": 0.4,
                "maximum_rank_per_query": 1000,
            },
            "fusion": {
                "method": "reciprocal_rank_fusion_v0",
                "rrf_k": 60,
                "query_weighting": "uniform",
                "duplicate_query_policy": "reject",
                "candidate_score": "sum_1_over_rrf_k_plus_one_based_rank",
                "tie_break": (
                    "best_individual_rank_then_occurrence_count_desc_then_docid_ascending"
                ),
                "evaluation_depths": [20, 100],
            },
            "acceptance": {
                "minimum_fused_gold_doc_recall_at20_cases": 1,
                "minimum_fused_gold_doc_recall_at100_cases_for_typed_reranker": 1,
            },
            "budgets": {
                "expected_frozen_queries": 4,
                "maximum_offline_bm25_queries": 4,
                "maximum_provider_calls": 0,
                "maximum_online_search_calls": 0,
                "maximum_judge_calls": 0,
                "gpu_allowed": False,
            },
            "sealed_holdout_access": "forbidden",
            "claim_boundary": "test only",
        },
    )

    shared_gold_rank = [*[f"a-{index}" for index in range(25)], "gold-1"]
    rankings = {
        "alpha branch": shared_gold_rank,
        "beta branch": [*[f"b-{index}" for index in range(25)], "gold-1"],
        "gamma branch": ["noise-gamma"],
        "delta branch": ["noise-delta"],
    }
    scoring_bytes = {
        gold: gold.read_bytes(),
        lexical: lexical.read_bytes(),
        pivot: pivot.read_bytes(),
    }
    return registration, rankings, scoring_bytes


def test_rrf_rewards_cross_query_agreement_and_has_stable_ties() -> None:
    fused = fuse_rankings(
        (("b", "shared"), ("a", "shared")),
        rrf_k=60,
    )
    assert fused[0].docid == "shared"
    assert fused[0].occurrence_count == 2
    assert [item.docid for item in fused[1:]] == ["a", "b"]


def test_build_is_gold_blind_then_score_applies_registered_gate(tmp_path: Path) -> None:
    registration, rankings, scoring_bytes = _fixture(tmp_path / "repo")
    for path in scoring_bytes:
        path.unlink()

    slate_path = tmp_path / "repo" / "runs" / "rrf" / "slate.json"
    slate = build_multi_query_rrf_slate(
        registration_path=registration,
        output_path=slate_path,
        search=lambda query, limit: rankings[query][:limit],
    )
    assert slate.query_count == 2
    assert slate.offline_bm25_queries == 4
    assert slate.provider_calls == slate.online_search_calls == slate.judge_calls == 0
    assert slate.items[0].fused_top100[0].docid == "gold-1"

    for path, content in scoring_bytes.items():
        path.write_bytes(content)
    result = score_multi_query_rrf_slate(
        registration_path=registration,
        slate_path=slate_path,
        output_path=tmp_path / "repo" / "runs" / "rrf" / "audit.json",
    )
    assert result.decision == "multi_query_rrf_candidate"
    assert result.fused_gold_hit_cases_at20 == 1
    assert result.fused_gold_hit_cases_at100 == 1
    assert result.single_query_gold_hit_cases_at20 == 0
    assert result.single_query_gold_hit_cases_at100 == 1
    assert result.items[0].best_fused_gold_rank == 1
    assert result.items[1].best_fused_gold_rank is None


def test_build_rejects_duplicate_queries_and_does_not_write(tmp_path: Path) -> None:
    registration, rankings, _ = _fixture(tmp_path / "repo")
    run_path = tmp_path / "repo" / "runs" / "baseline" / "q2" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["search_calls"][1]["query"] = "gamma branch"
    _write_json(run_path, run)
    output = tmp_path / "repo" / "runs" / "rrf" / "slate.json"
    with pytest.raises(ValueError, match="duplicate frozen query"):
        build_multi_query_rrf_slate(
            registration_path=registration,
            output_path=output,
            search=lambda query, limit: rankings[query][:limit],
        )
    assert not output.exists()


def test_score_rejects_changed_run_after_slate_persistence(tmp_path: Path) -> None:
    registration, rankings, _ = _fixture(tmp_path / "repo")
    slate_path = tmp_path / "repo" / "runs" / "rrf" / "slate.json"
    build_multi_query_rrf_slate(
        registration_path=registration,
        output_path=slate_path,
        search=lambda query, limit: rankings[query][:limit],
    )
    run_path = tmp_path / "repo" / "runs" / "baseline" / "q1" / "run.json"
    run_path.write_text(run_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "repo" / "runs" / "rrf" / "audit.json"
    with pytest.raises(ValueError, match="baseline run changed"):
        score_multi_query_rrf_slate(
            registration_path=registration,
            slate_path=slate_path,
            output_path=output,
        )
    assert not output.exists()


def test_build_and_score_refuse_overwrite(tmp_path: Path) -> None:
    registration, rankings, _ = _fixture(tmp_path / "repo")
    slate_path = tmp_path / "repo" / "runs" / "rrf" / "slate.json"
    audit_path = tmp_path / "repo" / "runs" / "rrf" / "audit.json"
    build_multi_query_rrf_slate(
        registration_path=registration,
        output_path=slate_path,
        search=lambda query, limit: rankings[query][:limit],
    )
    with pytest.raises(ValueError, match="slate already exists"):
        build_multi_query_rrf_slate(
            registration_path=registration,
            output_path=slate_path,
            search=lambda query, limit: rankings[query][:limit],
        )
    score_multi_query_rrf_slate(
        registration_path=registration,
        slate_path=slate_path,
        output_path=audit_path,
    )
    with pytest.raises(ValueError, match="result already exists"):
        score_multi_query_rrf_slate(
            registration_path=registration,
            slate_path=slate_path,
            output_path=audit_path,
        )
