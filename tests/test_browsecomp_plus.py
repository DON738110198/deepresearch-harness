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
    has_required_answer_schema,
    load_browsecomp_plus_target,
    load_deepseek_provider_snapshot,
    load_development_queries,
    load_official_evaluator_manifest,
    load_query_partitions,
    partition_query_id,
)
from deepresearch_harness.bm25_server import SearchRequest, truncate_with_tokenizer
from deepresearch_harness.browsecomp_evaluation import (
    DevelopmentGoldRow,
    DevelopmentGoldSlice,
    _prediction_set_sha256,
    extract_exact_answer,
    normalize_exact_answer,
    score_gold_diagnostic,
)
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


def test_answer_reserve_run_contract_tracks_phase_budget_and_schema() -> None:
    answer = (
        "Explanation: The fixture supports this answer [doc-1].\n"
        "Exact Answer: fixture\nConfidence: 80%"
    )
    payload = {
        "schema_version": "pi-browsecomp-run-v0",
        "adapter_version": "pi-browsecomp-v2",
        "pi_version": "0.84.1",
        "run_id": "run-reserve",
        "query_id": "query-reserve",
        "model": "deepseek-v4-flash",
        "thinking_level": "high",
        "control_policy": "answer_reserve_v0",
        "compilation_thinking_level": "high",
        "system_prompt": "",
        "prompt_sha256": "a" * 64,
        "started_at": "2026-08-13T00:00:00Z",
        "latency_ms": 10,
        "status": "succeeded",
        "stop_reason": "completed",
        "answer_text": answer,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 9_500,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 8_500,
            "total_tokens": 9_600,
            "cost_usd": 0.01,
        },
        "exploration_stop_reason": "exploration_output_token_limit_reached:8000",
        "exploration_output_tokens": 8_100,
        "compilation_output_tokens": 1_400,
        "answer_compiler_invoked": True,
        "answer_compiler_prompt_sha256": (
            "3e86775e1455eadc50433cf24b93f17585a79c22c0b548cdcd08dc8de5547861"
        ),
        "answer_schema_complete": True,
        "model_requests": 3,
        "provider_request_limits": [
            {
                "request_index": 1,
                "phase": "exploration",
                "output_limit_field": "max_tokens",
                "thinking_type": "enabled",
                "remaining_output_tokens": 8_000,
                "global_remaining_output_tokens": 10_000,
                "phase_remaining_output_tokens": 8_000,
                "requested_output_tokens": 10_000,
                "applied_output_tokens": 8_000,
            },
            {
                "request_index": 2,
                "phase": "compilation",
                "output_limit_field": "max_tokens",
                "thinking_type": "enabled",
                "remaining_output_tokens": 1_900,
                "global_remaining_output_tokens": 1_900,
                "phase_remaining_output_tokens": 2_000,
                "requested_output_tokens": 10_000,
                "applied_output_tokens": 1_900,
            },
            {
                "request_index": 3,
                "phase": "compilation",
                "output_limit_field": "max_tokens",
                "thinking_type": "enabled",
                "remaining_output_tokens": 500,
                "global_remaining_output_tokens": 500,
                "phase_remaining_output_tokens": 600,
                "requested_output_tokens": 10_000,
                "applied_output_tokens": 500,
            },
        ],
        "search_calls": [],
        "messages": [],
    }

    assert has_required_answer_schema(answer) is True
    assert PiBrowseCompRun.model_validate(payload).answer_compiler_invoked is True
    nonthinking_payload = json.loads(json.dumps(payload))
    nonthinking_payload["control_policy"] = "answer_reserve_nonthinking_v0"
    nonthinking_payload["compilation_thinking_level"] = "off"
    for limit in nonthinking_payload["provider_request_limits"]:
        if limit["phase"] == "compilation":
            limit["thinking_type"] = "disabled"
    assert (
        PiBrowseCompRun.model_validate(nonthinking_payload).compilation_thinking_level
        == "off"
    )
    payload["answer_compiler_prompt_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="prompt hash"):
        PiBrowseCompRun.model_validate(payload)


def test_rare_anchor_bootstrap_trace_binds_phases_and_tool_calls() -> None:
    answer = (
        "Explanation: The evidence supports the fixture [doc-1].\n"
        "Exact Answer: fixture\nConfidence: 80%"
    )
    search_calls = [
        {
            "query": f"rare anchor {index}",
            "outcome": "ok",
            "latency_ms": 1,
            "results": [
                {"docid": f"doc-{index}", "score": 1.0, "snippet": "fixture"}
            ],
        }
        for index in range(1, 4)
    ]
    payload = {
        "schema_version": "pi-browsecomp-run-v0",
        "adapter_version": "pi-browsecomp-v5",
        "pi_version": "0.84.1",
        "run_id": "rare-run",
        "query_id": "rare-query",
        "model": "deepseek-v4-flash",
        "thinking_level": "high",
        "control_policy": "rare_anchor_portfolio_v0",
        "compilation_thinking_level": "off",
        "system_prompt": "",
        "prompt_sha256": "a" * 64,
        "started_at": "2026-08-13T00:00:00Z",
        "latency_ms": 10,
        "status": "succeeded",
        "stop_reason": "completed",
        "answer_text": answer,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 7_000,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 6_000,
            "total_tokens": 7_100,
            "cost_usd": 0.01,
        },
        "bootstrap_stop_reason": "bootstrap_search_completed",
        "bootstrap_output_tokens": 100,
        "bootstrap_prompt_sha256": "b" * 64,
        "exploration_output_tokens": 6_500,
        "compilation_output_tokens": 400,
        "answer_compiler_invoked": True,
        "answer_compiler_prompt_sha256": (
            "3e86775e1455eadc50433cf24b93f17585a79c22c0b548cdcd08dc8de5547861"
        ),
        "answer_schema_complete": True,
        "model_requests": 3,
        "provider_request_limits": [
            {
                "request_index": 1,
                "phase": "bootstrap",
                "output_limit_field": "max_tokens",
                "thinking_type": "disabled",
                "remaining_output_tokens": 1_024,
                "global_remaining_output_tokens": 10_000,
                "phase_remaining_output_tokens": 1_024,
                "applied_output_tokens": 1_024,
            },
            {
                "request_index": 2,
                "phase": "exploration",
                "output_limit_field": "max_tokens",
                "thinking_type": "enabled",
                "remaining_output_tokens": 6_976,
                "global_remaining_output_tokens": 9_900,
                "phase_remaining_output_tokens": 6_976,
                "applied_output_tokens": 6_976,
            },
            {
                "request_index": 3,
                "phase": "compilation",
                "output_limit_field": "max_tokens",
                "thinking_type": "disabled",
                "remaining_output_tokens": 2_000,
                "global_remaining_output_tokens": 3_400,
                "phase_remaining_output_tokens": 2_000,
                "applied_output_tokens": 2_000,
            },
        ],
        "search_calls": search_calls,
        "messages": [],
    }

    run = PiBrowseCompRun.model_validate(payload)
    assert run.bootstrap_output_tokens == 100
    assert len(run.search_calls) == 3
    payload["search_calls"] = []
    with pytest.raises(ValidationError, match="must record a search call"):
        PiBrowseCompRun.model_validate(payload)


def test_bm25_request_is_strict_and_snippet_truncation_is_token_bounded() -> None:
    class Tokenizer:
        @staticmethod
        def encode(
            text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
            max_length: int,
        ) -> list[str]:
            assert add_special_tokens is False
            assert truncation is True
            return text.split()[:max_length]

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
    assert truncate_with_tokenizer("one  two", Tokenizer(), 3) == "one  two"


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


def test_exact_answer_diagnostic_is_explicitly_strict() -> None:
    response = "Explanation: Evidence [doc-1].\nExact Answer:  Ada Lovelace  \nConfidence: 90%"

    assert extract_exact_answer(response) == "Ada Lovelace"
    assert normalize_exact_answer("  ADA   Lovelace ") == "ada lovelace"
    assert normalize_exact_answer("Ada Lovelace.") != normalize_exact_answer(
        "Ada Lovelace"
    )


def test_gold_diagnostic_is_prediction_hash_bound(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    source_dir = tmp_path / "runs" / "source"
    run_dir = source_dir / "q-1"
    run_dir.mkdir(parents=True)
    answer = "Explanation: Supported [doc-1].\nExact Answer: Q7-BENCH\nConfidence: 90%"
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
            "input_tokens": 100,
            "output_tokens": 100,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 50,
            "total_tokens": 200,
            "cost_usd": 0.01,
        },
        "answer_schema_complete": True,
        "model_requests": 1,
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
    item = PiSmokeItem(
        query_id="q-1",
        status="succeeded",
        answer_schema_complete=True,
        run_path=str(run_path),
        run_sha256=sha256(run_path.read_bytes()).hexdigest(),
        prediction_sha256=sha256(answer.encode()).hexdigest(),
        search_calls=1,
        output_tokens=100,
        output_budget_overshoot_tokens=0,
        total_tokens=200,
        cost_usd=0.01,
        latency_ms=10,
    )
    summary = PiSmokeSummary(
        created_at="2026-08-13T00:00:00Z",
        target_manifest_sha256="b" * 64,
        development_queries_sha256="c" * 64,
        model="deepseek-v4-flash",
        query_count=1,
        succeeded=1,
        budget_exhausted=0,
        failed=0,
        schema_complete=1,
        total_search_calls=1,
        total_output_tokens=100,
        total_output_budget_overshoot_tokens=0,
        total_tokens=200,
        total_cost_usd=0.01,
        total_latency_ms=10,
        items=[item],
    )
    summary_path = source_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    gold = DevelopmentGoldSlice(
        created_at="2026-08-13T00:01:00Z",
        target_manifest_sha256="b" * 64,
        source_summary_sha256=sha256(summary_path.read_bytes()).hexdigest(),
        prediction_set_sha256=_prediction_set_sha256(summary),
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
                answer="Q7-BENCH",
                gold_docids=["doc-1"],
                evidence_docids=["doc-1", "doc-2"],
            )
        ],
    )
    gold_path = tmp_path / "runs" / "gold.json"
    gold_path.write_text(gold.model_dump_json(indent=2), encoding="utf-8")

    diagnostic = score_gold_diagnostic(
        source_dir=source_dir,
        gold_slice_path=gold_path,
        output_path=tmp_path / "runs" / "diagnostic.json",
    )

    assert diagnostic.normalized_exact_match_percent == 100.0
    assert diagnostic.evidence_recall_percent == 50.0
    assert diagnostic.gold_recall_percent == 100.0
    assert diagnostic.official_accuracy_status == "planned_not_run"
