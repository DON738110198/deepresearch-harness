from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import (
    PiBrowseCompRun,
    has_required_answer_schema,
    load_browsecomp_plus_target,
    load_development_queries,
    load_pi_browsecomp_run,
    normalized_text_file_sha256,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PiPriorAttempt(StrictContract):
    attempt_number: int = Field(ge=1)
    status: Literal["failed"]
    archive_path: str = Field(min_length=1)
    source_summary_path: str = Field(min_length=1)
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    search_calls: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    output_budget_overshoot_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    answer_compiler_invoked: bool = False
    error: str | None = None

    @model_validator(mode="after")
    def has_attempt_result(self) -> "PiPriorAttempt":
        if self.run_sha256 is None and self.error_sha256 is None:
            raise ValueError("archived attempt requires a run or error artifact")
        return self


class PiSmokeItem(StrictContract):
    query_id: str = Field(min_length=1)
    status: Literal["succeeded", "failed", "budget_exhausted"]
    control_policy: Literal[
        "standard",
        "answer_reserve_v0",
        "answer_reserve_v1",
        "answer_reserve_nonthinking_v0",
        "first_tool_deadline_v0",
        "tool_bootstrap_v0",
        "rare_anchor_portfolio_v0",
    ] = "standard"
    answer_schema_complete: bool | None = None
    answer_compiler_invoked: bool = False
    latest_attempt_answer_compiler_invoked: bool | None = None
    run_path: str | None = None
    run_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    search_calls: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    output_budget_overshoot_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    error: str | None = None
    attempt_count: int = Field(default=1, ge=1)
    prior_attempts: list[PiPriorAttempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def attempts_and_cumulative_usage_match(self) -> "PiSmokeItem":
        if self.attempt_count != len(self.prior_attempts) + 1:
            raise ValueError("attempt_count does not match prior_attempts")
        expected_numbers = list(range(1, self.attempt_count))
        if [attempt.attempt_number for attempt in self.prior_attempts] != expected_numbers:
            raise ValueError("prior attempts must be ordered and contiguous")
        cumulative_fields = (
            "search_calls",
            "output_tokens",
            "output_budget_overshoot_tokens",
            "total_tokens",
            "cost_usd",
            "latency_ms",
        )
        for field_name in cumulative_fields:
            prior_total = sum(
                getattr(attempt, field_name) for attempt in self.prior_attempts
            )
            if getattr(self, field_name) + 1e-12 < prior_total:
                raise ValueError(f"{field_name} is smaller than archived attempts")
        if any(
            attempt.answer_compiler_invoked for attempt in self.prior_attempts
        ) and not self.answer_compiler_invoked:
            raise ValueError("compiler invocation must include archived attempts")
        if self.latest_attempt_answer_compiler_invoked is not None:
            expected_compiler_invoked = (
                self.latest_attempt_answer_compiler_invoked
                or any(
                    attempt.answer_compiler_invoked
                    for attempt in self.prior_attempts
                )
            )
            if self.answer_compiler_invoked != expected_compiler_invoked:
                raise ValueError("compiler invocation does not match attempts")
        elif self.prior_attempts:
            raise ValueError("resumed item must identify the latest compiler attempt")
        return self


class PiSmokeSummary(StrictContract):
    schema_version: Literal[
        "pi-browsecomp-unscored-smoke-v0",
        "pi-browsecomp-unscored-smoke-v1",
    ] = "pi-browsecomp-unscored-smoke-v1"
    status: Literal["unscored_smoke"] = "unscored_smoke"
    gold_accessed: Literal[False] = False
    created_at: str
    updated_at: str | None = None
    resume_count: int = Field(default=0, ge=0)
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    thinking_level: Literal["high"] = "high"
    control_policy: Literal[
        "standard",
        "answer_reserve_v0",
        "answer_reserve_v1",
        "answer_reserve_nonthinking_v0",
        "first_tool_deadline_v0",
        "tool_bootstrap_v0",
        "rare_anchor_portfolio_v0",
    ] = "standard"
    system_prompt_policy: Literal["empty"] = "empty"
    retriever_id: str = Field(default="bm25", min_length=1)
    retriever_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    query_count: int = Field(gt=0)
    succeeded: int = Field(ge=0)
    budget_exhausted: int = Field(ge=0)
    failed: int = Field(ge=0)
    schema_complete: int | None = Field(default=None, ge=0)
    answer_compiler_invocations: int = Field(default=0, ge=0)
    total_search_calls: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_output_budget_overshoot_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    total_latency_ms: int = Field(ge=0)
    items: list[PiSmokeItem] = Field(min_length=1)

    @model_validator(mode="after")
    def counts_and_totals_match_items(self) -> "PiSmokeSummary":
        query_ids = [item.query_id for item in self.items]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("summary query IDs must be unique")
        expected = {
            "query_count": len(self.items),
            "succeeded": sum(item.status == "succeeded" for item in self.items),
            "budget_exhausted": sum(
                item.status == "budget_exhausted" for item in self.items
            ),
            "failed": sum(item.status == "failed" for item in self.items),
            "answer_compiler_invocations": sum(
                _answer_compiler_attempt_count(item) for item in self.items
            ),
            "total_search_calls": sum(item.search_calls for item in self.items),
            "total_output_tokens": sum(item.output_tokens for item in self.items),
            "total_output_budget_overshoot_tokens": sum(
                item.output_budget_overshoot_tokens for item in self.items
            ),
            "total_tokens": sum(item.total_tokens for item in self.items),
            "total_cost_usd": sum(item.cost_usd for item in self.items),
            "total_latency_ms": sum(item.latency_ms for item in self.items),
        }
        if self.schema_complete is not None:
            if any(item.answer_schema_complete is None for item in self.items):
                raise ValueError("schema_complete requires per-item schema flags")
            expected["schema_complete"] = sum(
                bool(item.answer_schema_complete) for item in self.items
            )
        if any(item.control_policy != self.control_policy for item in self.items):
            raise ValueError("item control policy does not match summary")
        if self.resume_count == 0 and self.updated_at is not None:
            raise ValueError("an unresumed summary cannot have updated_at")
        if self.resume_count > 0 and self.updated_at is None:
            raise ValueError("a resumed summary requires updated_at")
        if any(item.attempt_count > self.resume_count + 1 for item in self.items):
            raise ValueError("item attempts exceed the recorded resume count")
        if self.schema_version == "pi-browsecomp-unscored-smoke-v0" and (
            self.resume_count or any(item.prior_attempts for item in self.items)
        ):
            raise ValueError("v0 summary cannot contain resume history")
        if self.retriever_id == "bm25" and self.retriever_manifest_sha256 is not None:
            raise ValueError("BM25 summary cannot record a dense retriever manifest")
        if self.retriever_id != "bm25" and self.retriever_manifest_sha256 is None:
            raise ValueError("non-BM25 summary requires a retriever manifest hash")
        for field_name, expected_value in expected.items():
            actual_value = getattr(self, field_name)
            if isinstance(expected_value, float):
                matches = abs(actual_value - expected_value) <= 1e-12
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(f"{field_name} does not match items")
        return self


class OfficialRunExportItem(StrictContract):
    query_id: str = Field(min_length=1)
    source_run_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluator_status: Literal["completed", "incomplete"]
    exported_path: str = Field(min_length=1)
    exported_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OfficialRunExportManifest(StrictContract):
    schema_version: Literal["browsecomp-plus-official-run-export-v0"] = (
        "browsecomp-plus-official-run-export-v0"
    )
    created_at: str
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(gt=0)
    completed: int = Field(ge=0)
    incomplete: int = Field(ge=0)
    items: list[OfficialRunExportItem] = Field(min_length=1)

    @model_validator(mode="after")
    def counts_match_items(self) -> "OfficialRunExportManifest":
        query_ids = [item.query_id for item in self.items]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("export query IDs must be unique")
        if self.query_count != len(self.items):
            raise ValueError("query_count does not match items")
        if self.completed != sum(
            item.evaluator_status == "completed" for item in self.items
        ):
            raise ValueError("completed does not match items")
        if self.incomplete != sum(
            item.evaluator_status == "incomplete" for item in self.items
        ):
            raise ValueError("incomplete does not match items")
        return self


def run_pi_unscored_smoke(
    *,
    manifest_path: Path,
    partitions_path: Path,
    queries_path: Path,
    output_dir: Path,
    node_executable: Path,
    adapter_dir: Path,
    search_url: str,
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"],
    control_policy: Literal[
        "standard",
        "answer_reserve_v0",
        "answer_reserve_v1",
        "answer_reserve_nonthinking_v0",
        "first_tool_deadline_v0",
        "tool_bootstrap_v0",
        "rare_anchor_portfolio_v0",
    ] = "standard",
    timeout_seconds: int,
    retriever_id: str = "bm25",
    retriever_manifest_path: Path | None = None,
    resume_failed: bool = False,
) -> PiSmokeSummary:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    if not node_executable.is_file():
        raise ValueError(f"Node executable does not exist: {node_executable}")
    runner_path = (adapter_dir / "src" / "runner.mjs").resolve()
    if not runner_path.is_file():
        raise ValueError(f"Pi adapter runner does not exist: {runner_path}")
    repository_root = manifest_path.resolve().parents[2]
    if not output_dir.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("benchmark traces must be written under ignored runs/")
    output_exists = output_dir.exists() and any(output_dir.iterdir())
    summary_path = output_dir / "summary.json"
    if resume_failed and not summary_path.is_file():
        raise ValueError("failed-query resume requires an existing summary.json")
    if output_exists and not resume_failed:
        raise ValueError("benchmark output directory must be empty")
    if not retriever_id.strip():
        raise ValueError("retriever_id must not be blank")
    if retriever_id == "bm25" and retriever_manifest_path is not None:
        raise ValueError("BM25 uses the target manifest, not a retriever manifest")
    if retriever_id != "bm25" and retriever_manifest_path is None:
        raise ValueError("non-BM25 runs require a pinned retriever manifest")
    retriever_manifest_sha256 = (
        normalized_text_file_sha256(retriever_manifest_path)
        if retriever_manifest_path is not None
        else None
    )
    target_manifest_sha256 = normalized_text_file_sha256(manifest_path)
    development_queries_sha256 = normalized_text_file_sha256(queries_path)

    manifest = load_browsecomp_plus_target(manifest_path)
    queries = load_development_queries(
        queries_path,
        manifest_path=manifest_path,
        partitions_path=partitions_path,
    )
    track_by_model = {track.model: track for track in manifest.model_tracks}
    if model not in track_by_model:
        raise ValueError(f"model is absent from target manifest: {model}")
    safe_query_ids = [_safe_id(query.query_id) for query in queries.queries]
    if len(safe_query_ids) != len(set(safe_query_ids)):
        raise ValueError("query IDs collide after filesystem-safe normalization")
    contract = manifest.benchmark.standard_search
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_summary: PiSmokeSummary | None = None
    existing_summary_sha256: str | None = None
    source_summary_archive_path: str | None = None
    existing_by_query_id: dict[str, PiSmokeItem] = {}
    if resume_failed:
        summary_bytes = summary_path.read_bytes()
        existing_summary_sha256 = sha256(summary_bytes).hexdigest()
        existing_summary = PiSmokeSummary.model_validate_json(summary_bytes)
        _validate_resume_summary(
            summary=existing_summary,
            target_manifest_sha256=target_manifest_sha256,
            development_queries_sha256=development_queries_sha256,
            model=model,
            control_policy=control_policy,
            retriever_id=retriever_id,
            retriever_manifest_sha256=retriever_manifest_sha256,
            query_ids=[query.query_id for query in queries.queries],
        )
        existing_by_query_id = {
            item.query_id: item for item in existing_summary.items
        }
        source_summary_archive_path = _archive_source_summary(
            output_dir=output_dir,
            summary_bytes=summary_bytes,
            resume_number=existing_summary.resume_count + 1,
        )

    items: list[PiSmokeItem] = []

    for query in queries.queries:
        query_root = output_dir / _safe_id(query.query_id)
        existing_item = existing_by_query_id.get(query.query_id)
        if existing_item is not None:
            existing_request = _build_request(
                output_name=output_dir.name,
                query_id=query.query_id,
                question=query.question,
                model=model,
                thinking_level=track_by_model[model].thinking_level,
                control_policy=control_policy,
                max_output_tokens=contract.max_output_tokens,
                max_iterations=contract.max_iterations,
                search_url=search_url,
                attempt_number=existing_item.attempt_count,
            )
            _validate_existing_attempt(
                query_root=query_root,
                item=existing_item,
                request=existing_request,
                model=model,
                control_policy=control_policy,
            )
            if existing_item.status != "failed":
                items.append(existing_item)
                continue
            if existing_summary_sha256 is None or source_summary_archive_path is None:
                raise AssertionError("resume summary hash was not initialized")
            prior_attempt = _record_prior_attempt(
                query_root=query_root,
                item=existing_item,
                source_summary_path=source_summary_archive_path,
                source_summary_sha256=existing_summary_sha256,
            )
            attempt_number = existing_item.attempt_count + 1
        else:
            prior_attempt = None
            attempt_number = 1

        query_dir = _attempt_directory(query_root, attempt_number)
        query_dir.mkdir(parents=True, exist_ok=True)
        request_path = query_dir / "request.json"
        run_path = query_dir / "run.json"
        request = _build_request(
            output_name=output_dir.name,
            query_id=query.query_id,
            question=query.question,
            model=model,
            thinking_level=track_by_model[model].thinking_level,
            control_policy=control_policy,
            max_output_tokens=contract.max_output_tokens,
            max_iterations=contract.max_iterations,
            search_url=search_url,
            attempt_number=attempt_number,
        )
        _atomic_write(request_path, json.dumps(request, indent=2, ensure_ascii=False))
        try:
            completed = subprocess.run(
                [str(node_executable.resolve()), str(runner_path), str(request_path.resolve())],
                cwd=adapter_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(_redact(completed.stderr.strip()))
            run = PiBrowseCompRun.model_validate_json(completed.stdout)
            if run.query_id != query.query_id or run.model != model:
                raise ValueError("Pi run identity does not match its request")
            if run.run_id != request["run_id"] or run.control_policy != control_policy:
                raise ValueError("Pi run controls do not match its request")
            _atomic_write(run_path, completed.stdout)
            run_bytes = run_path.read_bytes()
            current_item = _summarize_run(run, run_path, run_bytes)
        except Exception as error:
            detail = _redact(str(error))[:4000]
            error_path = query_dir / "error.json"
            _atomic_write(
                error_path,
                json.dumps({"query_id": query.query_id, "error": detail}, indent=2),
            )
            current_item = PiSmokeItem(
                query_id=query.query_id,
                status="failed",
                control_policy=control_policy,
                answer_schema_complete=False,
                answer_compiler_invoked=False,
                latest_attempt_answer_compiler_invoked=False,
                search_calls=0,
                output_tokens=0,
                output_budget_overshoot_tokens=0,
                total_tokens=0,
                cost_usd=0,
                latency_ms=0,
                error=detail,
            )
        if existing_item is not None and prior_attempt is not None:
            current_item = _merge_retry_item(
                previous=existing_item,
                current=current_item,
                archived_previous=prior_attempt,
            )
            existing_by_query_id[query.query_id] = current_item
            interim_items = [
                existing_by_query_id[item.query_id] for item in queries.queries
            ]
            interim_summary = _build_smoke_summary(
                created_at=existing_summary.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
                resume_count=existing_summary.resume_count + 1,
                target_manifest_sha256=target_manifest_sha256,
                development_queries_sha256=development_queries_sha256,
                model=model,
                control_policy=control_policy,
                retriever_id=retriever_id,
                retriever_manifest_sha256=retriever_manifest_sha256,
                items=interim_items,
            )
            _atomic_write(summary_path, interim_summary.model_dump_json(indent=2))
        items.append(current_item)

    now = datetime.now(timezone.utc).isoformat()
    summary = _build_smoke_summary(
        created_at=existing_summary.created_at if existing_summary else now,
        updated_at=now if existing_summary else None,
        resume_count=(existing_summary.resume_count + 1) if existing_summary else 0,
        target_manifest_sha256=target_manifest_sha256,
        development_queries_sha256=development_queries_sha256,
        model=model,
        control_policy=control_policy,
        retriever_id=retriever_id,
        retriever_manifest_sha256=retriever_manifest_sha256,
        items=items,
    )
    _atomic_write(summary_path, summary.model_dump_json(indent=2))
    return summary


def export_pi_runs_for_official_evaluator(
    *, source_dir: Path, output_dir: Path
) -> OfficialRunExportManifest:
    summary_path = source_dir / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    repository_root = _find_repository_root(source_dir.resolve())
    if not source_dir.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("source benchmark traces must be under ignored runs/")
    if not output_dir.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("official evaluator exports must be written under ignored runs/")
    inputs_dir = output_dir / "inputs"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("official evaluator export directory must be empty")
    inputs_dir.mkdir(parents=True, exist_ok=True)

    exported_items: list[OfficialRunExportItem] = []
    for item in summary.items:
        query_root = source_dir / _safe_id(item.query_id)
        validate_pi_attempt_archives(query_root=query_root, item=item)
        if item.run_sha256 is None:
            if item.status != "failed" or item.prediction_sha256 is not None:
                raise ValueError(f"query {item.query_id} has an inconsistent missing run")
            run_hash = None
            prediction_hash = None
            evaluator_status: Literal["completed", "incomplete"] = "incomplete"
            model = summary.model
            thinking_level = summary.thinking_level
            compilation_thinking_level = (
                "off"
                if summary.control_policy
                in {
                    "answer_reserve_nonthinking_v0",
                    "first_tool_deadline_v0",
                    "tool_bootstrap_v0",
                    "rare_anchor_portfolio_v0",
                }
                else summary.thinking_level
            )
            query_id = item.query_id
            search_calls = item.search_calls
            input_tokens = max(item.total_tokens - item.output_tokens, 0)
            cache_read_tokens = 0
            output_tokens = item.output_tokens
            reasoning_tokens = 0
            total_tokens = item.total_tokens
            retrieved_docids: list[str] = []
            answer_text = ""
        else:
            run_path = query_root / "run.json"
            run_bytes = run_path.read_bytes()
            run_hash = sha256(run_bytes).hexdigest()
            if item.run_sha256 != run_hash:
                raise ValueError(f"source run hash mismatch for query {item.query_id}")
            run = load_pi_browsecomp_run(run_path)
            if run.query_id != item.query_id or run.model != summary.model:
                raise ValueError(f"source run identity mismatch for query {item.query_id}")
            prediction_hash = sha256(run.answer_text.encode("utf-8")).hexdigest()
            if item.prediction_sha256 != prediction_hash:
                raise ValueError(f"prediction hash mismatch for query {item.query_id}")
            evaluator_status = "completed" if run.status == "succeeded" else "incomplete"
            model = run.model
            thinking_level = run.thinking_level
            compilation_thinking_level = run.compilation_thinking_level
            query_id = run.query_id
            search_calls = len(run.search_calls)
            input_tokens = run.usage.input_tokens
            cache_read_tokens = run.usage.cache_read_tokens
            output_tokens = run.usage.output_tokens
            reasoning_tokens = run.usage.reasoning_tokens
            total_tokens = run.usage.total_tokens
            retrieved_docids = list(
                dict.fromkeys(
                    result.docid
                    for call in run.search_calls
                    for result in call.results
                )
            )
            answer_text = run.answer_text

        exported_payload = {
            "metadata": {
                "model": model,
                "reasoning": {"effort": thinking_level},
                "compilation_reasoning": {"effort": compilation_thinking_level},
                "max_tokens": 10_000,
                "control_policy": summary.control_policy,
                "retriever_id": summary.retriever_id,
                "retriever_manifest_sha256": summary.retriever_manifest_sha256,
                "source_run_sha256": run_hash,
                "target_manifest_sha256": summary.target_manifest_sha256,
            },
            "query_id": query_id,
            "tool_call_counts": {"search": search_calls},
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_cached": cache_read_tokens,
                "output_tokens": output_tokens,
                "included_reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
            },
            "status": evaluator_status,
            "retrieved_docids": retrieved_docids,
            "result": (
                [
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": answer_text,
                    }
                ]
                if evaluator_status == "completed"
                else []
            ),
        }
        export_path = inputs_dir / f"{_safe_id(query_id)}.json"
        _atomic_write(export_path, json.dumps(exported_payload, indent=2, ensure_ascii=False))
        export_bytes = export_path.read_bytes()
        exported_items.append(
            OfficialRunExportItem(
                query_id=query_id,
                source_run_sha256=run_hash,
                prediction_sha256=prediction_hash,
                evaluator_status=evaluator_status,
                exported_path=str(export_path),
                exported_sha256=sha256(export_bytes).hexdigest(),
            )
        )

    export_manifest = OfficialRunExportManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        target_manifest_sha256=summary.target_manifest_sha256,
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        query_count=len(exported_items),
        completed=sum(item.evaluator_status == "completed" for item in exported_items),
        incomplete=sum(item.evaluator_status == "incomplete" for item in exported_items),
        items=exported_items,
    )
    _atomic_write(
        output_dir / "export_manifest.json",
        export_manifest.model_dump_json(indent=2),
    )
    return export_manifest


def validate_pi_attempt_archives(*, query_root: Path, item: PiSmokeItem) -> None:
    """Verify every prior-attempt artifact and its source-summary binding."""
    _validate_prior_attempt_artifacts(query_root=query_root, item=item)


def _build_smoke_summary(
    *,
    created_at: str,
    updated_at: str | None,
    resume_count: int,
    target_manifest_sha256: str,
    development_queries_sha256: str,
    model: str,
    control_policy: str,
    retriever_id: str,
    retriever_manifest_sha256: str | None,
    items: list[PiSmokeItem],
) -> PiSmokeSummary:
    return PiSmokeSummary(
        created_at=created_at,
        updated_at=updated_at,
        resume_count=resume_count,
        target_manifest_sha256=target_manifest_sha256,
        development_queries_sha256=development_queries_sha256,
        model=model,
        control_policy=control_policy,
        retriever_id=retriever_id,
        retriever_manifest_sha256=retriever_manifest_sha256,
        query_count=len(items),
        succeeded=sum(item.status == "succeeded" for item in items),
        budget_exhausted=sum(item.status == "budget_exhausted" for item in items),
        failed=sum(item.status == "failed" for item in items),
        schema_complete=sum(bool(item.answer_schema_complete) for item in items),
        answer_compiler_invocations=sum(
            _answer_compiler_attempt_count(item) for item in items
        ),
        total_search_calls=sum(item.search_calls for item in items),
        total_output_tokens=sum(item.output_tokens for item in items),
        total_output_budget_overshoot_tokens=sum(
            item.output_budget_overshoot_tokens for item in items
        ),
        total_tokens=sum(item.total_tokens for item in items),
        total_cost_usd=sum(item.cost_usd for item in items),
        total_latency_ms=sum(item.latency_ms for item in items),
        items=items,
    )


def _build_request(
    *,
    output_name: str,
    query_id: str,
    question: str,
    model: str,
    thinking_level: str,
    control_policy: str,
    max_output_tokens: int,
    max_iterations: int,
    search_url: str,
    attempt_number: int,
) -> dict[str, object]:
    if attempt_number < 1:
        raise ValueError("attempt_number must be positive")
    base_run_id = f"{output_name}-{query_id}"
    run_id = (
        base_run_id
        if attempt_number == 1
        else f"{base_run_id}-attempt-{attempt_number:02d}"
    )
    return {
        "schema_version": "pi-browsecomp-request-v0",
        "run_id": run_id,
        "query_id": query_id,
        "question": question,
        "model": model,
        "thinking_level": thinking_level,
        "control_policy": control_policy,
        "max_output_tokens": max_output_tokens,
        "max_iterations": max_iterations,
        "search": {"kind": "http", "url": search_url, "timeout_ms": 120_000},
    }


def _validate_resume_summary(
    *,
    summary: PiSmokeSummary,
    target_manifest_sha256: str,
    development_queries_sha256: str,
    model: str,
    control_policy: str,
    retriever_id: str,
    retriever_manifest_sha256: str | None,
    query_ids: list[str],
) -> None:
    expected = {
        "target_manifest_sha256": target_manifest_sha256,
        "development_queries_sha256": development_queries_sha256,
        "model": model,
        "control_policy": control_policy,
        "retriever_id": retriever_id,
        "retriever_manifest_sha256": retriever_manifest_sha256,
    }
    for field_name, expected_value in expected.items():
        if getattr(summary, field_name) != expected_value:
            raise ValueError(f"resume {field_name} does not match frozen summary")
    if [item.query_id for item in summary.items] != query_ids:
        raise ValueError("resume query IDs or ordering do not match frozen summary")
    if summary.failed == 0:
        raise ValueError("frozen summary has no failed queries to resume")


def _archive_source_summary(
    *, output_dir: Path, summary_bytes: bytes, resume_number: int
) -> str:
    relative_path = Path("resume_history") / f"resume-{resume_number:02d}" / "summary.json"
    archive_path = output_dir / relative_path
    if archive_path.exists():
        if archive_path.read_bytes() != summary_bytes:
            raise ValueError(f"resume history already contains different bytes: {archive_path}")
    else:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(archive_path, summary_bytes)
    return relative_path.as_posix()


def _attempt_directory(query_root: Path, attempt_number: int) -> Path:
    allowed_entries = {"attempts"} if attempt_number > 1 else set()
    if query_root.exists():
        unexpected = {
            path.name for path in query_root.iterdir() if path.name not in allowed_entries
        }
        if unexpected:
            raise ValueError(
                f"attempt {attempt_number} has unexpected live artifacts: "
                f"{sorted(unexpected)}"
            )
    for name in ("request.json", "run.json", "error.json"):
        if (query_root / name).exists():
            raise ValueError(
                f"attempt {attempt_number} live artifact already exists: {query_root / name}"
            )
    return query_root


def _validate_existing_attempt(
    *,
    query_root: Path,
    item: PiSmokeItem,
    request: dict[str, object],
    model: str,
    control_policy: str,
) -> None:
    _validate_prior_attempt_artifacts(query_root=query_root, item=item)
    request_path = query_root / "request.json"
    if not request_path.is_file():
        raise ValueError(f"missing live request for query {item.query_id}")
    try:
        actual_request = json.loads(request_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid live request for query {item.query_id}") from error
    if actual_request != request:
        raise ValueError(f"live request does not match frozen controls for {item.query_id}")

    run_path = query_root / "run.json"
    error_path = query_root / "error.json"
    if item.run_sha256 is None:
        if item.status != "failed" or item.prediction_sha256 is not None:
            raise ValueError(f"missing run is inconsistent for query {item.query_id}")
        if run_path.exists() or not error_path.is_file():
            raise ValueError(f"failed query artifacts are inconsistent for {item.query_id}")
        error_payload = json.loads(error_path.read_text(encoding="utf-8"))
        if (
            error_payload.get("query_id") != item.query_id
            or error_payload.get("error") != item.error
        ):
            raise ValueError(f"failed query error does not match summary for {item.query_id}")
        return

    if not run_path.is_file() or error_path.exists():
        raise ValueError(f"live run artifacts are inconsistent for {item.query_id}")
    run_bytes = run_path.read_bytes()
    if sha256(run_bytes).hexdigest() != item.run_sha256:
        raise ValueError(f"live run hash mismatch for query {item.query_id}")
    run = PiBrowseCompRun.model_validate_json(run_bytes)
    if (
        run.query_id != item.query_id
        or run.model != model
        or run.control_policy != control_policy
        or run.run_id != request["run_id"]
        or run.status != item.status
    ):
        raise ValueError(f"live run identity mismatch for query {item.query_id}")
    prediction_sha256 = sha256(run.answer_text.encode("utf-8")).hexdigest()
    if prediction_sha256 != item.prediction_sha256:
        raise ValueError(f"live prediction hash mismatch for query {item.query_id}")
    expected_latest = {
        "search_calls": len(run.search_calls),
        "output_tokens": run.usage.output_tokens,
        "output_budget_overshoot_tokens": run.output_budget_overshoot_tokens,
        "total_tokens": run.usage.total_tokens,
        "cost_usd": run.usage.cost_usd,
        "latency_ms": run.latency_ms,
    }
    for field_name, expected_value in expected_latest.items():
        actual_value = _latest_attempt_value(item, field_name)
        if isinstance(expected_value, float):
            matches = abs(actual_value - expected_value) <= 1e-12
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(f"live {field_name} mismatch for query {item.query_id}")
    if _latest_attempt_compiler_invoked(item) != run.answer_compiler_invoked:
        raise ValueError(f"live compiler trace mismatch for query {item.query_id}")
    if item.answer_schema_complete != has_required_answer_schema(run.answer_text):
        raise ValueError(f"live answer schema mismatch for query {item.query_id}")


def _validate_prior_attempt_artifacts(*, query_root: Path, item: PiSmokeItem) -> None:
    output_dir = query_root.parent
    for attempt in item.prior_attempts:
        expected_relative = Path("attempts") / f"attempt-{attempt.attempt_number:02d}"
        if Path(attempt.archive_path) != expected_relative:
            raise ValueError(f"unexpected attempt archive path for query {item.query_id}")
        archive_path = (query_root / expected_relative).resolve()
        if not archive_path.is_relative_to(query_root.resolve()) or not archive_path.is_dir():
            raise ValueError(f"invalid attempt archive for query {item.query_id}")
        artifact_hashes = {
            "request.json": attempt.request_sha256,
            "run.json": attempt.run_sha256,
            "error.json": attempt.error_sha256,
        }
        for name, expected_hash in artifact_hashes.items():
            artifact_path = archive_path / name
            if expected_hash is None:
                if artifact_path.exists():
                    raise ValueError(f"unexpected archived {name} for query {item.query_id}")
            elif not artifact_path.is_file() or sha256(artifact_path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"archived {name} hash mismatch for query {item.query_id}")
        summary_path = (output_dir / attempt.source_summary_path).resolve()
        if (
            not summary_path.is_relative_to(output_dir.resolve())
            or not summary_path.is_file()
            or sha256(summary_path.read_bytes()).hexdigest()
            != attempt.source_summary_sha256
        ):
            raise ValueError(f"archived summary hash mismatch for query {item.query_id}")


def _record_prior_attempt(
    *,
    query_root: Path,
    item: PiSmokeItem,
    source_summary_path: str,
    source_summary_sha256: str,
) -> PiPriorAttempt:
    archive_relative = Path("attempts") / f"attempt-{item.attempt_count:02d}"
    archive_path = query_root / archive_relative
    if archive_path.exists():
        raise ValueError(f"attempt archive already exists: {archive_path}")

    hashes: dict[str, str | None] = {}
    for name in ("request.json", "run.json", "error.json"):
        live_path = query_root / name
        if live_path.is_file():
            hashes[name] = sha256(live_path.read_bytes()).hexdigest()
        else:
            hashes[name] = None
    if hashes["request.json"] is None:
        raise ValueError(f"cannot archive missing request for query {item.query_id}")
    if hashes["run.json"] is None and hashes["error.json"] is None:
        raise ValueError(f"cannot archive missing result for query {item.query_id}")
    if hashes["run.json"] != item.run_sha256:
        raise ValueError(f"archived run hash does not match query {item.query_id}")
    prior_attempt = PiPriorAttempt(
        attempt_number=item.attempt_count,
        status=item.status,
        archive_path=archive_relative.as_posix(),
        source_summary_path=source_summary_path,
        source_summary_sha256=source_summary_sha256,
        request_sha256=hashes["request.json"],
        run_sha256=hashes["run.json"],
        error_sha256=hashes["error.json"],
        prediction_sha256=item.prediction_sha256,
        search_calls=int(_latest_attempt_value(item, "search_calls")),
        output_tokens=int(_latest_attempt_value(item, "output_tokens")),
        output_budget_overshoot_tokens=int(
            _latest_attempt_value(item, "output_budget_overshoot_tokens")
        ),
        total_tokens=int(_latest_attempt_value(item, "total_tokens")),
        cost_usd=float(_latest_attempt_value(item, "cost_usd")),
        latency_ms=int(_latest_attempt_value(item, "latency_ms")),
        answer_compiler_invoked=_latest_attempt_compiler_invoked(item),
        error=item.error,
    )
    archive_path.mkdir(parents=True)
    for name, artifact_hash in hashes.items():
        if artifact_hash is not None:
            (query_root / name).replace(archive_path / name)
    return prior_attempt


def _merge_retry_item(
    *, previous: PiSmokeItem, current: PiSmokeItem, archived_previous: PiPriorAttempt
) -> PiSmokeItem:
    if previous.query_id != current.query_id:
        raise ValueError("cannot merge retry attempts from different queries")
    latest_compiler = _latest_attempt_compiler_invoked(current)
    return PiSmokeItem(
        query_id=current.query_id,
        status=current.status,
        control_policy=current.control_policy,
        answer_schema_complete=current.answer_schema_complete,
        answer_compiler_invoked=previous.answer_compiler_invoked or latest_compiler,
        latest_attempt_answer_compiler_invoked=latest_compiler,
        run_path=current.run_path,
        run_sha256=current.run_sha256,
        prediction_sha256=current.prediction_sha256,
        search_calls=previous.search_calls + current.search_calls,
        output_tokens=previous.output_tokens + current.output_tokens,
        output_budget_overshoot_tokens=(
            previous.output_budget_overshoot_tokens
            + current.output_budget_overshoot_tokens
        ),
        total_tokens=previous.total_tokens + current.total_tokens,
        cost_usd=previous.cost_usd + current.cost_usd,
        latency_ms=previous.latency_ms + current.latency_ms,
        error=current.error,
        attempt_count=previous.attempt_count + 1,
        prior_attempts=[*previous.prior_attempts, archived_previous],
    )


def _latest_attempt_value(item: PiSmokeItem, field_name: str) -> int | float:
    value = getattr(item, field_name) - sum(
        getattr(attempt, field_name) for attempt in item.prior_attempts
    )
    if isinstance(value, float) and abs(value) <= 1e-12:
        return 0.0
    if value < 0:
        raise ValueError(f"negative latest attempt {field_name} for query {item.query_id}")
    return value


def _latest_attempt_compiler_invoked(item: PiSmokeItem) -> bool:
    if item.latest_attempt_answer_compiler_invoked is not None:
        return item.latest_attempt_answer_compiler_invoked
    if item.prior_attempts:
        raise ValueError("resumed item lacks latest compiler trace")
    return item.answer_compiler_invoked


def _answer_compiler_attempt_count(item: PiSmokeItem) -> int:
    return sum(
        attempt.answer_compiler_invoked for attempt in item.prior_attempts
    ) + int(_latest_attempt_compiler_invoked(item))


def _summarize_run(run: PiBrowseCompRun, run_path: Path, run_bytes: bytes) -> PiSmokeItem:
    return PiSmokeItem(
        query_id=run.query_id,
        status=run.status,
        control_policy=run.control_policy,
        answer_schema_complete=has_required_answer_schema(run.answer_text),
        answer_compiler_invoked=run.answer_compiler_invoked,
        latest_attempt_answer_compiler_invoked=run.answer_compiler_invoked,
        run_path=str(run_path),
        run_sha256=sha256(run_bytes).hexdigest(),
        prediction_sha256=sha256(run.answer_text.encode("utf-8")).hexdigest(),
        search_calls=len(run.search_calls),
        output_tokens=run.usage.output_tokens,
        output_budget_overshoot_tokens=run.output_budget_overshoot_tokens,
        total_tokens=run.usage.total_tokens,
        cost_usd=run.usage.cost_usd,
        latency_ms=run.latency_ms,
        error=None if run.status == "succeeded" else run.stop_reason,
    )


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not safe or safe in {".", ".."}:
        raise ValueError("query ID cannot be represented as a safe directory name")
    return safe


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate repository root for benchmark export")


def _atomic_write(path: Path, text: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(content)
    temp_path.replace(path)


def _redact(text: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return text.replace(api_key, "[REDACTED]") if api_key else text
