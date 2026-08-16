from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.dense_rank_audit import (
    choose_dense_rank_decision,
    collect_dense_rank_cases,
    load_dense_rank_registration,
    run_dense_rank_audit,
)
from deepresearch_harness.retrieval_replay import DenseRuntimeSnapshot, RankedHit


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _ranked(query: str, *, gold: str, gold_rank: int) -> list[RankedHit]:
    hits = [
        RankedHit(docid=f"{query}-doc-{index}", score=float(1001 - index))
        for index in range(1, 1001)
    ]
    hits[gold_rank - 1] = RankedHit(docid=gold, score=float(1001 - gold_rank))
    return hits


def test_dense_rank_gate_selects_bounded_reranker_without_external_calls(
    tmp_path: Path,
) -> None:
    registration_path, runtime = _fixture(tmp_path)
    registration = load_dense_rank_registration(registration_path)
    assert registration.depths == (20, 100, 1000)
    frozen = collect_dense_rank_cases(registration_path)
    assert [item.query_id for item in frozen] == ["q1", "q2"]

    output = tmp_path / "runs" / "result" / "audit.json"
    result = run_dense_rank_audit(
        registration_path=registration_path,
        candidate_results={
            "alpha query": _ranked("alpha", gold="gold-1", gold_rank=50),
            "beta query": _ranked("beta", gold="gold-2", gold_rank=75),
        },
        runtime=runtime,
        output_path=output,
    )

    assert result.decision == "dense_pool_reranker_diagnosis"
    assert result.next_action == "preregister_bounded_offline_reranker_gate"
    assert result.dense_gold_hit_cases_at20 == 0
    assert result.dense_gold_hit_cases_at100 == 2
    assert result.dense_gold_hit_cases_at1000 == 2
    assert result.provider_calls == result.online_search_calls == result.judge_calls == 0
    assert result.sealed_holdout_accessed is False
    assert output.is_file()


def test_dense_rank_decision_prefers_top20_then_top100() -> None:
    assert choose_dense_rank_decision(
        dense_top20_cases=4,
        dense_top100_cases=4,
        minimum_top20_cases=4,
        minimum_top100_cases=4,
    )[0] == "dense_top20_candidate"
    assert choose_dense_rank_decision(
        dense_top20_cases=3,
        dense_top100_cases=4,
        minimum_top20_cases=4,
        minimum_top100_cases=4,
    )[0] == "dense_pool_reranker_diagnosis"
    assert choose_dense_rank_decision(
        dense_top20_cases=3,
        dense_top100_cases=3,
        minimum_top20_cases=4,
        minimum_top100_cases=4,
    )[0] == "freeze_dense_channel"


def test_dense_rank_rejects_changed_query_sequence(tmp_path: Path) -> None:
    registration_path, _ = _fixture(tmp_path)
    run_path = tmp_path / "runs" / "baseline" / "q1" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["search_calls"][0]["query"] = "changed query"
    _write_json(run_path, run)

    with pytest.raises(ValueError, match="recorded query sequence changed"):
        collect_dense_rank_cases(registration_path)


def test_dense_rank_rejects_short_candidate_ranking(tmp_path: Path) -> None:
    registration_path, runtime = _fixture(tmp_path)
    with pytest.raises(ValueError, match="returned 999 hits instead of 1000"):
        run_dense_rank_audit(
            registration_path=registration_path,
            candidate_results={
                "alpha query": _ranked("alpha", gold="gold-1", gold_rank=50)[:-1],
                "beta query": _ranked("beta", gold="gold-2", gold_rank=75),
            },
            runtime=runtime,
            output_path=tmp_path / "runs" / "result" / "audit.json",
        )


def _fixture(tmp_path: Path) -> tuple[Path, DenseRuntimeSnapshot]:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    benchmark = tmp_path / "benchmarks" / "browsecomp_plus_v0"
    runs = tmp_path / "runs"
    runs.mkdir()

    lexical_path = runs / "prior" / "lexical.json"
    _write_json(
        lexical_path,
        {
            "decision": "index_representation_diagnosis_required",
            "generated_query_top20_cases": 0,
            "items": [{"query_id": "q1"}, {"query_id": "q2"}],
        },
    )
    passage_path = runs / "prior" / "passage.json"
    _write_json(
        passage_path,
        {
            "decision": "freeze_passage_index_branch",
            "full_document_gold_hit_cases_at20": 0,
            "items": [
                {
                    "query_id": "q1",
                    "gold_docids": ["gold-1"],
                    "queries": [
                        {
                            "query": "alpha query",
                            "query_sha256": _text_hash("alpha query"),
                        }
                    ],
                    "passage_gold_hit": True,
                },
                {
                    "query_id": "q2",
                    "gold_docids": ["gold-2"],
                    "queries": [
                        {
                            "query": "beta query",
                            "query_sha256": _text_hash("beta query"),
                        }
                    ],
                    "passage_gold_hit": False,
                },
            ],
        },
    )
    gold_path = runs / "prior" / "gold.json"
    _write_json(
        gold_path,
        {
            "rows": [
                {"query_id": "q1", "gold_docids": ["gold-1"]},
                {"query_id": "q2", "gold_docids": ["gold-2"]},
            ]
        },
    )
    baseline = runs / "baseline"
    _write_json(
        baseline / "q1" / "run.json",
        {
            "query_id": "q1",
            "search_calls": [{"outcome": "ok", "query": "alpha query"}],
        },
    )
    _write_json(
        baseline / "q2" / "run.json",
        {
            "query_id": "q2",
            "search_calls": [{"outcome": "ok", "query": "beta query"}],
        },
    )

    model_dir = runs / "external" / "model"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.safetensors"
    model_file.write_bytes(b"fixture-model")
    index_root = runs / "external" / "indexes"
    index_dir = index_root / "dense"
    index_dir.mkdir(parents=True)
    shard = index_dir / "corpus.pkl"
    shard.write_bytes(b"fixture-index")

    manifest_path = benchmark / "retriever.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "browsecomp-plus-retriever-candidates-v0",
            "target_manifest_sha256": "0" * 64,
            "candidates": [
                {
                    "candidate_id": "fixture-dense",
                    "kind": "dense_faiss",
                    "model": {
                        "name": "fixture/model",
                        "revision": "1" * 40,
                        "model_file": model_file.name,
                        "model_file_sha256": _hash(model_file),
                    },
                    "index": {
                        "dataset": "fixture/index",
                        "revision": "2" * 40,
                        "subdirectory": "dense",
                        "shards": [{"filename": shard.name, "sha256": _hash(shard)}],
                    },
                    "query_prefix": "Query:",
                    "pooling": "eos",
                    "normalize": True,
                    "max_length": 512,
                    "top_k": 5,
                }
            ],
            "replay": {
                "source_queries": "frozen_agent_search_calls",
                "baseline": "stored_bm25_top5",
                "candidate_depth": 5,
                "fusion": "reciprocal_rank_fusion",
                "fusion_k": 60,
                "fused_depth": 5,
                "official_metric_status": "diagnostic_not_official",
            },
        },
    )
    registration_path = benchmark / "registration.json"
    _write_json(
        registration_path,
        {
            "schema_version": "persistent-miss-dense-rank-registration-v0",
            "status": "posthoc_registered_failure_cluster",
            "registered_at": "2026-08-16T00:00:00+00:00",
            "purpose": "fixture dense-rank audit",
            "lexical_rank_result": {
                "path": lexical_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(lexical_path),
            },
            "passage_index_result": {
                "path": passage_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(passage_path),
            },
            "gold_slice": {
                "path": gold_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(gold_path),
            },
            "baseline_run_root": baseline.relative_to(tmp_path).as_posix(),
            "query_ids": ["q1", "q2"],
            "retriever_manifest": {
                "path": manifest_path.relative_to(tmp_path).as_posix(),
                "normalized_sha256": normalized_text_file_sha256(manifest_path),
            },
            "candidate_id": "fixture-dense",
            "model_directory": model_dir.relative_to(tmp_path).as_posix(),
            "index_root": index_root.relative_to(tmp_path).as_posix(),
            "query_source": "trial1_recorded_successful_search_calls",
            "depths": [20, 100, 1000],
            "batch_size": 8,
            "acceptance": {
                "minimum_dense_top20_cases": 2,
                "minimum_dense_top100_cases_for_pool_diagnosis": 2,
            },
            "budgets": {
                "maximum_provider_calls": 0,
                "maximum_online_search_calls": 0,
                "maximum_judge_calls": 0,
            },
            "sealed_holdout_access": "forbidden",
            "claim_boundary": "fixture diagnostic only",
        },
    )
    runtime = DenseRuntimeSnapshot(
        device="cpu",
        torch_version="fixture",
        transformers_version="fixture",
        faiss_version="fixture",
        tevatron_version="fixture",
        model_file_sha256=_hash(model_file),
        local_model_files_sha256={},
        index_shards_sha256={shard.name: _hash(shard)},
        index_documents=2000,
        embedding_dimensions=8,
        load_latency_ms=1,
        search_latency_ms=1,
    )
    return registration_path, runtime
