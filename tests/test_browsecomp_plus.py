import base64
import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepresearch_harness.browsecomp_plus import (
    BrowseCompPlusTargetManifest,
    BROWSECOMP_PLUS_CANARY,
    PiBrowseCompRun,
    _decrypt_query,
    freeze_query_partitions,
    load_browsecomp_plus_target,
    load_deepseek_provider_snapshot,
    load_development_queries,
    load_official_evaluator_manifest,
    load_query_partitions,
    partition_query_id,
)
from deepresearch_harness.bm25_server import SearchRequest, truncate_with_tokenizer
from deepresearch_harness.pi_browsecomp import (
    export_pi_runs_for_official_evaluator,
    PiSmokeItem,
    PiSmokeSummary,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "browsecomp_plus_v0" / "target_manifest.json"
EVALUATOR_MANIFEST = ROOT / "benchmarks" / "browsecomp_plus_v0" / "official_evaluator.json"
PROVIDER_SNAPSHOT = (
    ROOT / "benchmarks" / "browsecomp_plus_v0" / "deepseek_provider_snapshot.json"
)


def test_target_manifest_is_strict_and_pinned() -> None:
    manifest = load_browsecomp_plus_target(MANIFEST)

    assert manifest.benchmark.repository_commit == "046949032b0328319cc9a02663a759ec601d9402"
    assert manifest.split.frozen_before_gold_access is True
    assert manifest.benchmark.standard_search.top_k == 5
    assert manifest.benchmark.standard_search.snippet_max_tokens == 512
    assert manifest.benchmark.standard_search.system_prompt_policy == "empty"
    assert {track.model for track in manifest.model_tracks} == {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    }


def test_target_manifest_rejects_unknown_fields() -> None:
    payload = load_browsecomp_plus_target(MANIFEST).model_dump(mode="json")
    payload["unregistered_change"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BrowseCompPlusTargetManifest.model_validate(payload)


def test_official_evaluator_is_revision_pinned_and_target_bound(tmp_path: Path) -> None:
    evaluator = load_official_evaluator_manifest(
        EVALUATOR_MANIFEST, target_manifest_path=MANIFEST
    )

    assert evaluator.judge.revision == "9216db5781bf21249d130ec9da846c4624c16137"
    assert evaluator.inference.enable_thinking is False

    crlf_manifest = tmp_path / "target-crlf.json"
    normalized = MANIFEST.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf_manifest.write_bytes(normalized.replace(b"\n", b"\r\n"))
    load_official_evaluator_manifest(
        EVALUATOR_MANIFEST, target_manifest_path=crlf_manifest
    )


def test_deepseek_provider_versions_and_prices_are_snapshot_bound() -> None:
    snapshot = load_deepseek_provider_snapshot(
        PROVIDER_SNAPSHOT, target_manifest_path=MANIFEST
    )
    by_model = {model.api_model: model for model in snapshot.models}

    assert by_model["deepseek-v4-flash"].documented_model_version == (
        "DeepSeek-V4-Flash-0731"
    )
    assert by_model["deepseek-v4-pro"].pricing.output_usd_per_million == 0.87


def test_query_partition_is_deterministic() -> None:
    policy = load_browsecomp_plus_target(MANIFEST).split

    assert partition_query_id("example-17", policy) == partition_query_id(" example-17 ", policy)
    assert partition_query_id("example-17", policy) in {"development", "sealed_holdout"}


def test_freeze_query_partitions_writes_ids_only(tmp_path: Path) -> None:
    input_path = tmp_path / "queries.tsv"
    input_path.write_text("q-2\tsecret question two\nq-1\tsecret question one\n", encoding="utf-8")
    output_path = tmp_path / "partitions.json"

    artifact = freeze_query_partitions(
        manifest_path=MANIFEST,
        query_ids_tsv=input_path,
        output_path=output_path,
    )
    persisted = output_path.read_text(encoding="utf-8")

    assert artifact.query_count == 2
    assert [row.query_id for row in artifact.rows] == ["q-1", "q-2"]
    assert "secret question" not in persisted


def test_freeze_query_partitions_rejects_duplicate_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "queries.tsv"
    input_path.write_text("q-1\tone\nq-1\ttwo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate query IDs"):
        freeze_query_partitions(
            manifest_path=MANIFEST,
            query_ids_tsv=input_path,
            output_path=tmp_path / "partitions.json",
        )


def test_query_partitions_are_bound_to_target_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "queries.tsv"
    input_path.write_text("q-1\nq-2\n", encoding="utf-8")
    output_path = tmp_path / "partitions.json"
    freeze_query_partitions(
        manifest_path=MANIFEST,
        query_ids_tsv=input_path,
        output_path=output_path,
    )

    artifact = load_query_partitions(output_path, manifest_path=MANIFEST)
    assert artifact.query_count == 2

    changed_manifest = tmp_path / "target.json"
    changed_manifest.write_bytes(MANIFEST.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="different target manifest"):
        load_query_partitions(output_path, manifest_path=changed_manifest)


def test_query_decryption_is_limited_to_the_requested_string() -> None:
    plaintext = "development question only"
    digest = sha256(BROWSECOMP_PLUS_CANARY.encode("utf-8")).digest()
    source = plaintext.encode("utf-8")
    key = (digest * (len(source) // len(digest) + 1))[: len(source)]
    encrypted = base64.b64encode(
        bytes(left ^ right for left, right in zip(source, key, strict=True))
    ).decode("ascii")

    assert _decrypt_query(encrypted) == plaintext


def test_pi_run_contract_requires_empty_system_prompt() -> None:
    payload = {
        "schema_version": "pi-browsecomp-run-v0",
        "adapter_version": "pi-browsecomp-v0",
        "pi_version": "0.84.1",
        "run_id": "run-1",
        "query_id": "query-1",
        "model": "deepseek-v4-flash",
        "thinking_level": "high",
        "system_prompt": "",
        "prompt_sha256": "a" * 64,
        "started_at": "2026-08-13T00:00:00Z",
        "latency_ms": 10,
        "status": "succeeded",
        "stop_reason": "completed",
        "answer_text": "Exact Answer: fixture",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 2,
            "cost_usd": 0.001,
        },
        "model_requests": 1,
        "search_calls": [],
        "messages": [],
    }
    assert PiBrowseCompRun.model_validate(payload).system_prompt == ""
    payload["system_prompt"] = "coding agent prompt"
    with pytest.raises(ValidationError):
        PiBrowseCompRun.model_validate(payload)


def test_bm25_request_is_strict_and_snippet_truncation_is_token_bounded() -> None:
    class Tokenizer:
        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[str]:
            assert add_special_tokens is False
            return text.split()

        @staticmethod
        def decode(tokens: list[str], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return " ".join(tokens)

    assert SearchRequest.model_validate({"query": "bounded search"}).query == "bounded search"
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "x", "k": 10})
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "   "})
    assert truncate_with_tokenizer("one two three", Tokenizer(), 2) == "one two"


def test_pi_smoke_summary_rejects_inconsistent_budget_counts() -> None:
    item = PiSmokeItem(
        query_id="q-1",
        status="budget_exhausted",
        search_calls=2,
        output_tokens=10_001,
        output_budget_overshoot_tokens=1,
        total_tokens=20_000,
        cost_usd=0.01,
        latency_ms=100,
        error="global output budget reached",
    )
    payload = {
        "created_at": "2026-08-13T00:00:00Z",
        "target_manifest_sha256": "a" * 64,
        "development_queries_sha256": "b" * 64,
        "model": "deepseek-v4-flash",
        "query_count": 1,
        "succeeded": 0,
        "budget_exhausted": 0,
        "failed": 1,
        "total_search_calls": 2,
        "total_output_tokens": 10_001,
        "total_output_budget_overshoot_tokens": 1,
        "total_tokens": 20_000,
        "total_cost_usd": 0.01,
        "total_latency_ms": 100,
        "items": [item.model_dump()],
    }

    valid = payload | {"budget_exhausted": 1, "failed": 0}
    assert PiSmokeSummary.model_validate(valid).total_output_tokens == 10_001

    with pytest.raises(ValidationError, match="budget_exhausted does not match items"):
        PiSmokeSummary.model_validate(payload)


def test_official_export_is_hash_bound_and_keeps_terminal_answer(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    source_dir = tmp_path / "runs" / "source"
    run_dir = source_dir / "q-1"
    run_dir.mkdir(parents=True)
    answer = "Exact Answer: Q7-BENCH\nConfidence: 90%"
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
        "stop_reason": "completed_with_output_token_overshoot:1",
        "answer_text": answer,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10_001,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 9_000,
            "total_tokens": 10_101,
            "cost_usd": 0.01,
        },
        "output_budget_overshoot_tokens": 1,
        "model_requests": 1,
        "provider_request_limits": [],
        "search_calls": [
            {
                "query": "fixture",
                "outcome": "ok",
                "latency_ms": 1,
                "results": [{"docid": "doc-1", "score": 1.0, "snippet": "fixture"}],
            }
        ],
        "messages": [],
    }
    run_path = run_dir / "run.json"
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    run_hash = sha256(run_path.read_bytes()).hexdigest()
    prediction_hash = sha256(answer.encode()).hexdigest()
    item = PiSmokeItem(
        query_id="q-1",
        status="succeeded",
        run_path=str(run_path),
        run_sha256=run_hash,
        prediction_sha256=prediction_hash,
        search_calls=1,
        output_tokens=10_001,
        output_budget_overshoot_tokens=1,
        total_tokens=10_101,
        cost_usd=0.01,
        latency_ms=10,
    )
    failed_item = PiSmokeItem(
        query_id="q-2",
        status="failed",
        search_calls=0,
        output_tokens=0,
        output_budget_overshoot_tokens=0,
        total_tokens=0,
        cost_usd=0,
        latency_ms=0,
        error="provider unavailable",
    )
    summary = PiSmokeSummary(
        created_at="2026-08-13T00:00:00Z",
        target_manifest_sha256="b" * 64,
        development_queries_sha256="c" * 64,
        model="deepseek-v4-flash",
        query_count=2,
        succeeded=1,
        budget_exhausted=0,
        failed=1,
        total_search_calls=1,
        total_output_tokens=10_001,
        total_output_budget_overshoot_tokens=1,
        total_tokens=10_101,
        total_cost_usd=0.01,
        total_latency_ms=10,
        items=[item, failed_item],
    )
    (source_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )

    exported = export_pi_runs_for_official_evaluator(
        source_dir=source_dir,
        output_dir=tmp_path / "runs" / "official",
    )
    official_run = json.loads(
        (tmp_path / "runs" / "official" / "inputs" / "q-1.json").read_text(
            encoding="utf-8"
        )
    )
    failed_run = json.loads(
        (tmp_path / "runs" / "official" / "inputs" / "q-2.json").read_text(
            encoding="utf-8"
        )
    )

    assert exported.completed == 1
    assert exported.incomplete == 1
    assert official_run["status"] == "completed"
    assert official_run["result"][-1]["output"] == answer
    assert official_run["retrieved_docids"] == ["doc-1"]
    assert failed_run["status"] == "incomplete"
    assert failed_run["result"] == []


def test_development_query_loader_rejects_holdout_ids(tmp_path: Path) -> None:
    policy = load_browsecomp_plus_target(MANIFEST).split
    holdout_id = next(
        f"candidate-{index}"
        for index in range(1000)
        if partition_query_id(f"candidate-{index}", policy) == "sealed_holdout"
    )
    input_path = tmp_path / "ids.tsv"
    input_path.write_text(f"{holdout_id}\n", encoding="utf-8")
    partitions_path = tmp_path / "partitions.json"
    freeze_query_partitions(
        manifest_path=MANIFEST,
        query_ids_tsv=input_path,
        output_path=partitions_path,
    )
    payload = {
        "schema_version": "browsecomp-plus-development-queries-v0",
        "target_manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "query_partitions_sha256": sha256(partitions_path.read_bytes()).hexdigest(),
        "partition": "development",
        "query_count": 1,
        "queries_sha256": sha256(f"{holdout_id}\tsecret".encode()).hexdigest(),
        "queries": [{"query_id": holdout_id, "question": "secret"}],
    }
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed-holdout"):
        load_development_queries(
            queries_path,
            manifest_path=MANIFEST,
            partitions_path=partitions_path,
        )
