from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.raw_question_dense_rank import (
    build_raw_question_dense_slate,
    choose_raw_question_dense_decision,
    compare_ranks,
    score_raw_question_dense_slate,
)
from deepresearch_harness.retrieval_replay import DenseRuntimeSnapshot, RankedHit


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _ranking(prefix: str, *, gold: str | None = None, rank: int = 1) -> list[RankedHit]:
    hits = [
        RankedHit(docid=f"{prefix}-doc-{index}", score=float(1001 - index))
        for index in range(1, 1001)
    ]
    if gold is not None:
        hits[rank - 1] = RankedHit(docid=gold, score=float(1001 - rank))
    return hits


def test_two_phase_raw_question_dense_audit_preserves_gold_blind_builder(
    tmp_path: Path,
) -> None:
    registration, runtime, paths = _fixture(tmp_path)
    hidden_bytes = {path: path.read_bytes() for path in paths}
    for path in paths:
        path.unlink()

    slate_path = tmp_path / "runs" / "audit" / "slate.json"
    slate = build_raw_question_dense_slate(
        registration_path=registration,
        output_path=slate_path,
        candidate_results={
            "raw question one": _ranking("one", gold="gold-1", rank=10),
            "raw question two": _ranking("two"),
        },
        runtime=runtime,
    )
    assert slate.gold_inputs_opened is False
    assert slate.generated_rank_result_opened is False
    assert slate.visibility_result_opened is False
    assert slate.dense_query_encodes == 2

    for path, content in hidden_bytes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    result = score_raw_question_dense_slate(
        registration_path=registration,
        slate_path=slate_path,
        output_path=tmp_path / "runs" / "audit" / "result.json",
    )
    assert result.decision == "raw_question_top20_candidate"
    assert result.raw_question_gold_hit_cases_at20 == 1
    assert result.generated_query_gold_hit_cases_at20 == 0
    assert result.raw_rank_wins == 1
    assert result.raw_rank_losses == 1
    assert result.provider_calls == result.online_search_calls == result.judge_calls == 0
    assert result.gpu_used is False


def test_decision_prioritizes_top20_then_top100_then_rank_signal() -> None:
    assert choose_raw_question_dense_decision(
        raw_top20_cases=4,
        raw_top100_cases=4,
        raw_rank_wins=4,
        minimum_top20_cases=4,
        minimum_top100_cases=4,
        minimum_rank_wins=4,
    )[0] == "raw_question_top20_candidate"
    assert choose_raw_question_dense_decision(
        raw_top20_cases=3,
        raw_top100_cases=4,
        raw_rank_wins=4,
        minimum_top20_cases=4,
        minimum_top100_cases=4,
        minimum_rank_wins=4,
    )[0] == "raw_question_pool_candidate"
    assert choose_raw_question_dense_decision(
        raw_top20_cases=3,
        raw_top100_cases=3,
        raw_rank_wins=4,
        minimum_top20_cases=4,
        minimum_top100_cases=4,
        minimum_rank_wins=4,
    )[0] == "raw_question_alignment_signal_only"
    assert choose_raw_question_dense_decision(
        raw_top20_cases=3,
        raw_top100_cases=3,
        raw_rank_wins=3,
        minimum_top20_cases=4,
        minimum_top100_cases=4,
        minimum_rank_wins=4,
    )[0] == "freeze_raw_question_dense"


def test_rank_comparison_treats_missing_as_below_depth() -> None:
    assert compare_ranks(10, 20) == "win"
    assert compare_ranks(20, 10) == "loss"
    assert compare_ranks(10, 10) == "tie"
    assert compare_ranks(10, None) == "win"
    assert compare_ranks(None, 10) == "loss"
    assert compare_ranks(None, None) == "tie"


def test_builder_rejects_short_dense_ranking(tmp_path: Path) -> None:
    registration, runtime, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="999 hits instead of 1000"):
        build_raw_question_dense_slate(
            registration_path=registration,
            output_path=tmp_path / "runs" / "audit" / "slate.json",
            candidate_results={
                "raw question one": _ranking("one")[:-1],
                "raw question two": _ranking("two"),
            },
            runtime=runtime,
        )


def _fixture(
    tmp_path: Path,
) -> tuple[Path, DenseRuntimeSnapshot, tuple[Path, Path, Path]]:
    benchmark = tmp_path / "benchmarks" / "browsecomp_plus_v0"
    runs = tmp_path / "runs"
    benchmark.mkdir(parents=True)
    runs.mkdir()

    questions = runs / "questions.json"
    _write_json(
        questions,
        {
            "schema_version": "browsecomp-plus-development-queries-v0",
            "target_manifest_sha256": "0" * 64,
            "query_partitions_sha256": "1" * 64,
            "partition": "development",
            "query_count": 2,
            "queries_sha256": "2" * 64,
            "queries": [
                {"query_id": "q1", "question": "raw question one"},
                {"query_id": "q2", "question": "raw question two"},
            ],
        },
    )
    visibility = runs / "visibility.json"
    _write_json(
        visibility,
        {
            "decision": "reject_head_truncation_hypothesis",
            "items": [{"query_id": "q1"}, {"query_id": "q2"}],
        },
    )
    generated = runs / "generated.json"
    _write_json(
        generated,
        {
            "decision": "freeze_dense_channel",
            "dense_gold_hit_cases_at20": 0,
            "dense_gold_hit_cases_at100": 1,
            "dense_gold_hit_cases_at1000": 2,
            "items": [
                {
                    "query_id": "q1",
                    "gold_docids": ["gold-1"],
                    "best_gold_rank": 100,
                },
                {
                    "query_id": "q2",
                    "gold_docids": ["gold-2"],
                    "best_gold_rank": 50,
                },
            ],
        },
    )
    gold = runs / "gold.json"
    _write_json(
        gold,
        {
            "rows": [
                {"query_id": "q1", "gold_docids": ["gold-1"]},
                {"query_id": "q2", "gold_docids": ["gold-2"]},
            ]
        },
    )

    model_dir = runs / "model"
    model_dir.mkdir()
    model_file = model_dir / "model.safetensors"
    model_file.write_bytes(b"fixture-model")
    index_root = runs / "indexes"
    index_dir = index_root / "dense"
    index_dir.mkdir(parents=True)
    shard = index_dir / "corpus.pkl"
    shard.write_bytes(b"fixture-index")
    retriever_manifest = benchmark / "retriever.json"
    _write_json(
        retriever_manifest,
        {
            "schema_version": "browsecomp-plus-retriever-candidates-v0",
            "target_manifest_sha256": "0" * 64,
            "candidates": [
                {
                    "candidate_id": "fixture-dense",
                    "kind": "dense_faiss",
                    "model": {
                        "name": "fixture/model",
                        "revision": "3" * 40,
                        "model_file": model_file.name,
                        "model_file_sha256": _hash(model_file),
                    },
                    "index": {
                        "dataset": "fixture/index",
                        "revision": "4" * 40,
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

    registration = benchmark / "registration.json"
    _write_json(
        registration,
        {
            "schema_version": "raw-question-dense-rank-registration-v0",
            "status": "posthoc_registered_failure_cluster",
            "registered_at": "2026-08-16T00:00:00+00:00",
            "purpose": "fixture raw-question dense audit",
            "prerequisites": {
                "dense_document_visibility_result": {
                    "path": visibility.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(visibility),
                },
                "generated_query_dense_rank_result": {
                    "path": generated.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(generated),
                },
                "gold_slice": {
                    "path": gold.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(gold),
                },
            },
            "question_artifact": {
                "path": questions.relative_to(tmp_path).as_posix(),
                "sha256": _hash(questions),
            },
            "query_ids": ["q1", "q2"],
            "retriever_manifest": {
                "path": retriever_manifest.relative_to(tmp_path).as_posix(),
                "normalized_sha256": normalized_text_file_sha256(retriever_manifest),
            },
            "candidate_id": "fixture-dense",
            "model_directory": model_dir.relative_to(tmp_path).as_posix(),
            "index_root": index_root.relative_to(tmp_path).as_posix(),
            "query_source": "frozen_pre_generation_question_artifact",
            "depths": [20, 100, 1000],
            "batch_size": 2,
            "comparison": {
                "baseline": "fixture baseline",
                "rank_win_policy": "fixture win policy",
                "tie_policy": "equal ranks are ties",
            },
            "acceptance": {
                "minimum_raw_question_top20_cases": 1,
                "minimum_raw_question_top100_cases_for_pool_diagnosis": 1,
                "minimum_raw_question_rank_wins_for_alignment_signal": 1,
            },
            "budgets": {
                "expected_dense_query_encodes": 2,
                "maximum_provider_calls": 0,
                "maximum_online_search_calls": 0,
                "maximum_judge_calls": 0,
                "gpu_allowed": False,
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
    return registration, runtime, (visibility, generated, gold)
