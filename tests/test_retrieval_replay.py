import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.browsecomp_evaluation import (
    DevelopmentGoldRow,
    DevelopmentGoldSlice,
)
from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.pi_browsecomp import PiSmokeItem, PiSmokeSummary
from deepresearch_harness.retrieval_replay import (
    DenseRuntimeSnapshot,
    RankedHit,
    collect_frozen_search_queries,
    load_retriever_candidates,
    reciprocal_rank_fusion,
    score_retrieval_replay,
    select_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "benchmarks" / "browsecomp_plus_v0" / "target_manifest.json"
RETRIEVERS = ROOT / "benchmarks" / "browsecomp_plus_v0" / "retriever_candidates.json"


def test_retriever_candidate_is_pinned_and_rrf_is_deterministic() -> None:
    manifest = load_retriever_candidates(RETRIEVERS, target_manifest_path=TARGET)
    candidate = select_candidate(manifest, "qwen3-embedding-0.6b")

    assert candidate.model.revision == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    assert candidate.index.revision == "b3f37f70c33829eb09d04784a54277a31871fd63"
    assert len(candidate.index.shards) == 4
    assert reciprocal_rank_fusion(
        ["doc-a", "doc-c"], ["doc-b", "doc-c"], k=60, depth=3
    ) == ["doc-c", "doc-a", "doc-b"]


def test_counterfactual_replay_preserves_queries_and_attributes_recall(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    source_dir = tmp_path / "runs" / "source"
    run_dir = source_dir / "q-1"
    run_dir.mkdir(parents=True)
    answer = "Explanation: fixture.\nExact Answer: fixture\nConfidence: 50%"
    run_payload = {
        "schema_version": "pi-browsecomp-run-v0",
        "adapter_version": "pi-browsecomp-v0",
        "pi_version": "0.84.1",
        "run_id": "run-1",
        "query_id": "q-1",
        "model": "deepseek-v4-flash",
        "thinking_level": "high",
        "system_prompt": "",
        "prompt_sha256": "a" * 64,
        "started_at": "2026-08-13T00:00:00Z",
        "latency_ms": 10,
        "status": "succeeded",
        "stop_reason": "completed",
        "answer_text": answer,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 10,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 20,
            "cost_usd": 0,
        },
        "answer_schema_complete": True,
        "model_requests": 1,
        "search_calls": [
            {
                "query": "frozen clue query",
                "outcome": "ok",
                "latency_ms": 1,
                "results": [
                    {"docid": "doc-a", "score": 2.0, "snippet": "baseline"},
                    {"docid": "doc-x", "score": 1.0, "snippet": "distractor"},
                ],
            }
        ],
        "messages": [],
    }
    run_path = run_dir / "run.json"
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    item = PiSmokeItem(
        query_id="q-1",
        status="succeeded",
        answer_schema_complete=True,
        run_path=str(run_path),
        run_sha256=sha256(run_path.read_bytes()).hexdigest(),
        prediction_sha256=sha256(answer.encode()).hexdigest(),
        search_calls=1,
        output_tokens=10,
        output_budget_overshoot_tokens=0,
        total_tokens=20,
        cost_usd=0,
        latency_ms=10,
    )
    summary = PiSmokeSummary(
        created_at="2026-08-13T00:00:00Z",
        target_manifest_sha256=normalized_text_file_sha256(TARGET),
        development_queries_sha256="c" * 64,
        model="deepseek-v4-flash",
        query_count=1,
        succeeded=1,
        budget_exhausted=0,
        failed=0,
        schema_complete=1,
        total_search_calls=1,
        total_output_tokens=10,
        total_output_budget_overshoot_tokens=0,
        total_tokens=20,
        total_cost_usd=0,
        total_latency_ms=10,
        items=[item],
    )
    summary_path = source_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="retriever manifest hash"):
        PiSmokeSummary.model_validate(
            {**summary.model_dump(), "retriever_id": "dense-fixture"}
        )
    gold = DevelopmentGoldSlice(
        created_at="2026-08-13T00:01:00Z",
        target_manifest_sha256=normalized_text_file_sha256(TARGET),
        source_summary_sha256=sha256(summary_path.read_bytes()).hexdigest(),
        prediction_set_sha256="d" * 64,
        accessed_fields=(
            "query_id",
            "query",
            "answer",
            "gold_docs.docid",
            "evidence_docs.docid",
        ),
        excluded_fields=(
            "gold_docs.text",
            "gold_docs.url",
            "evidence_docs.text",
            "evidence_docs.url",
            "negative_docs",
        ),
        query_count=1,
        rows=[
            DevelopmentGoldRow(
                query_id="q-1",
                question="fixture?",
                answer="fixture",
                gold_docids=["doc-a", "doc-b"],
                evidence_docids=["doc-a", "doc-b"],
            )
        ],
    )
    gold_path = tmp_path / "runs" / "gold.json"
    gold_path.write_text(gold.model_dump_json(indent=2), encoding="utf-8")
    candidate = select_candidate(
        load_retriever_candidates(RETRIEVERS, target_manifest_path=TARGET),
        "qwen3-embedding-0.6b",
    )
    runtime = DenseRuntimeSnapshot(
        device="cpu",
        torch_version="fixture",
        transformers_version="fixture",
        faiss_version="fixture",
        tevatron_version="fixture",
        model_file_sha256=candidate.model.model_file_sha256,
        local_model_files_sha256={"config.json": "e" * 64},
        index_shards_sha256={
            shard.filename: shard.sha256 for shard in candidate.index.shards
        },
        index_documents=10,
        embedding_dimensions=4,
        load_latency_ms=1,
        search_latency_ms=1,
    )

    assert collect_frozen_search_queries(source_dir) == ["frozen clue query"]
    replay = score_retrieval_replay(
        source_dir=source_dir,
        gold_slice_path=gold_path,
        retriever_manifest_path=RETRIEVERS,
        target_manifest_path=TARGET,
        candidate_id=candidate.candidate_id,
        candidate_results={
            "frozen clue query": [
                RankedHit(docid="doc-b", score=2.0),
                RankedHit(docid="doc-y", score=1.0),
            ]
        },
        runtime=runtime,
        output_path=tmp_path / "runs" / "replay.json",
    )

    assert replay.search_calls == 1
    assert replay.unique_search_queries == 1
    assert replay.baseline_evidence_recall_percent == 50.0
    assert replay.candidate_evidence_recall_percent == 50.0
    assert replay.fused_evidence_recall_percent == 100.0
    assert replay.evidence_recall_delta_candidate_pp == 0.0
    assert replay.evidence_recall_delta_fused_pp == 50.0
    assert replay.retriever_manifest_sha256 == normalized_text_file_sha256(RETRIEVERS)
    assert replay.official_accuracy_status == "planned_not_run"
