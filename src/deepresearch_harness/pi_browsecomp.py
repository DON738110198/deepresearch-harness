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
    load_browsecomp_plus_target,
    load_development_queries,
    load_pi_browsecomp_run,
    normalized_text_file_sha256,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PiSmokeItem(StrictContract):
    query_id: str = Field(min_length=1)
    status: Literal["succeeded", "failed", "budget_exhausted"]
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


class PiSmokeSummary(StrictContract):
    schema_version: Literal["pi-browsecomp-unscored-smoke-v0"] = (
        "pi-browsecomp-unscored-smoke-v0"
    )
    status: Literal["unscored_smoke"] = "unscored_smoke"
    gold_accessed: Literal[False] = False
    created_at: str
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    thinking_level: Literal["high"] = "high"
    system_prompt_policy: Literal["empty"] = "empty"
    query_count: int = Field(gt=0)
    succeeded: int = Field(ge=0)
    budget_exhausted: int = Field(ge=0)
    failed: int = Field(ge=0)
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
            "total_search_calls": sum(item.search_calls for item in self.items),
            "total_output_tokens": sum(item.output_tokens for item in self.items),
            "total_output_budget_overshoot_tokens": sum(
                item.output_budget_overshoot_tokens for item in self.items
            ),
            "total_tokens": sum(item.total_tokens for item in self.items),
            "total_cost_usd": sum(item.cost_usd for item in self.items),
            "total_latency_ms": sum(item.latency_ms for item in self.items),
        }
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
    timeout_seconds: int,
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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("benchmark output directory must be empty")

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
    items: list[PiSmokeItem] = []

    for query in queries.queries:
        query_dir = output_dir / _safe_id(query.query_id)
        query_dir.mkdir(parents=True, exist_ok=True)
        request_path = query_dir / "request.json"
        run_path = query_dir / "run.json"
        request = {
            "schema_version": "pi-browsecomp-request-v0",
            "run_id": f"{output_dir.name}-{query.query_id}",
            "query_id": query.query_id,
            "question": query.question,
            "model": model,
            "thinking_level": track_by_model[model].thinking_level,
            "max_output_tokens": contract.max_output_tokens,
            "max_iterations": contract.max_iterations,
            "search": {"kind": "http", "url": search_url, "timeout_ms": 120_000},
        }
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
            _atomic_write(run_path, completed.stdout)
            run_bytes = run_path.read_bytes()
            items.append(_summarize_run(run, run_path, run_bytes))
        except Exception as error:
            detail = _redact(str(error))[:4000]
            error_path = query_dir / "error.json"
            _atomic_write(
                error_path,
                json.dumps({"query_id": query.query_id, "error": detail}, indent=2),
            )
            items.append(
                PiSmokeItem(
                    query_id=query.query_id,
                    status="failed",
                    search_calls=0,
                    output_tokens=0,
                    output_budget_overshoot_tokens=0,
                    total_tokens=0,
                    cost_usd=0,
                    latency_ms=0,
                    error=detail,
                )
            )

    summary = PiSmokeSummary(
        created_at=datetime.now(timezone.utc).isoformat(),
        target_manifest_sha256=normalized_text_file_sha256(manifest_path),
        development_queries_sha256=normalized_text_file_sha256(queries_path),
        model=model,
        query_count=len(items),
        succeeded=sum(item.status == "succeeded" for item in items),
        budget_exhausted=sum(item.status == "budget_exhausted" for item in items),
        failed=sum(item.status == "failed" for item in items),
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
    _atomic_write(output_dir / "summary.json", summary.model_dump_json(indent=2))
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
        if item.run_sha256 is None:
            if item.status != "failed" or item.prediction_sha256 is not None:
                raise ValueError(f"query {item.query_id} has an inconsistent missing run")
            run_hash = None
            prediction_hash = None
            evaluator_status: Literal["completed", "incomplete"] = "incomplete"
            model = summary.model
            thinking_level = summary.thinking_level
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
            run_path = source_dir / _safe_id(item.query_id) / "run.json"
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
                "max_tokens": 10_000,
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


def _summarize_run(run: PiBrowseCompRun, run_path: Path, run_bytes: bytes) -> PiSmokeItem:
    return PiSmokeItem(
        query_id=run.query_id,
        status=run.status,
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


def _redact(text: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return text.replace(api_key, "[REDACTED]") if api_key else text
