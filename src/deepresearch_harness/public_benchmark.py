from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, HttpUrl, JsonValue, field_validator

from .contracts import HarnessConfig, RunState
from .pipeline import BenchmarkResearchPipeline
from .providers import provider_from_config
from .web_research import live_collector_from_config


class LiveDRBenchManifest(BaseModel):
    benchmark_id: str = Field(min_length=1)
    status: Literal["frozen"]
    dataset_id: Literal["microsoft/LiveDRBench"]
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_config: Literal["preview"] = "preview"
    dataset_split: Literal["test"] = "test"
    task_keys: list[int] = Field(min_length=1, max_length=10)
    source_repository: HttpUrl
    source_repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    selection_rationale: str = Field(min_length=1)
    evaluator: Literal["compatibility_exact_main_claim_v1"]
    official_evaluator_status: Literal["planned_not_run"]

    @field_validator("task_keys")
    @classmethod
    def task_keys_are_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("task_keys must be unique")
        return value


class LiveDRBenchTask(BaseModel):
    key: int
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truths: JsonValue
    eval_info: dict[str, Any] = Field(default_factory=dict)


class ExactClaimMetrics(BaseModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    matches: int = Field(ge=0)
    predicted_claims: int = Field(ge=0)
    reference_claims: int = Field(ge=0)
    compared_fields: list[str]


class BenchmarkTaskResult(BaseModel):
    key: int
    category: str
    status: Literal["succeeded", "failed"]
    run_id: str | None = None
    state_path: str | None = None
    report_path: str | None = None
    error: str | None = None
    structured_output_present: bool = False
    official_shape_compatible: bool = False
    total_tokens: int = Field(ge=0, default=0)
    estimated_cost_usd: float = Field(ge=0, default=0)
    traced_latency_ms: int = Field(ge=0, default=0)
    search_calls: int = Field(ge=0, default=0)
    fetched_sources: int = Field(ge=0, default=0)
    fetch_errors: int = Field(ge=0, default=0)
    exact_main_claim: ExactClaimMetrics | None = None


class BenchmarkPilotSummary(BaseModel):
    benchmark_id: str
    status: Literal["succeeded", "completed_with_failures"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    output_dir: str
    dataset_id: str
    dataset_revision: str
    dataset_response_sha256: str
    selected_questions_sha256: str
    source_repository_commit: str
    variant: Literal["b1_benchmark_structured"] = "b1_benchmark_structured"
    provider_kind: str
    model: str
    thinking_mode: str | None
    evaluator: Literal["compatibility_exact_main_claim_v1"]
    official_evaluator_status: Literal["planned_not_run"]
    task_count: int = Field(ge=1)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    structured_output_rate: float = Field(ge=0, le=1)
    official_shape_compatible_rate: float = Field(ge=0, le=1)
    macro_exact_precision: float = Field(ge=0, le=1)
    macro_exact_recall: float = Field(ge=0, le=1)
    macro_exact_f1: float = Field(ge=0, le=1)
    total_tokens: int = Field(ge=0)
    total_estimated_cost_usd: float = Field(ge=0)
    wall_time_seconds: float = Field(ge=0)
    results: list[BenchmarkTaskResult]


class CompatibilityScoreRow(BaseModel):
    key: int
    category: str
    official_shape_compatible: bool
    exact_main_claim: ExactClaimMetrics


class CompatibilityScoreArtifact(BaseModel):
    benchmark_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_revision: str
    dataset_response_sha256: str
    predictions_sha256: str
    task_count: int = Field(ge=1)
    prediction_coverage_rate: float = Field(ge=0, le=1)
    official_shape_compatible_rate: float = Field(ge=0, le=1)
    macro_exact_precision: float = Field(ge=0, le=1)
    macro_exact_recall: float = Field(ge=0, le=1)
    macro_exact_f1: float = Field(ge=0, le=1)
    evaluator: Literal["compatibility_exact_main_claim_v1"]
    official_evaluator_status: Literal["planned_not_run"]
    results: list[CompatibilityScoreRow]


def load_livedrbench_manifest(path: Path) -> LiveDRBenchManifest:
    return LiveDRBenchManifest.model_validate_json(path.read_text(encoding="utf-8"))


def fetch_livedrbench_tasks(
    manifest: LiveDRBenchManifest,
    *,
    timeout_seconds: int = 30,
) -> tuple[list[LiveDRBenchTask], str]:
    query = urlencode(
        {
            "dataset": manifest.dataset_id,
            "config": manifest.dataset_config,
            "split": manifest.dataset_split,
            "offset": 0,
            "length": 100,
            "revision": manifest.dataset_revision,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "deepresearch-harness/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read(2_000_001)
    if len(body) > 2_000_000:
        raise ValueError("LiveDRBench dataset response exceeded 2 MB")
    payload = json.loads(body)
    rows_by_key: dict[int, dict[str, Any]] = {}
    for wrapped in payload.get("rows", []):
        row = wrapped.get("row", {})
        if isinstance(row.get("key"), int):
            rows_by_key[row["key"]] = row
    missing = [key for key in manifest.task_keys if key not in rows_by_key]
    if missing:
        raise ValueError(f"LiveDRBench keys not found at pinned revision: {missing}")

    tasks: list[LiveDRBenchTask] = []
    for key in manifest.task_keys:
        row = rows_by_key[key]
        misc = _json_object(row.get("misc", "{}"), field="misc", key=key)
        tasks.append(
            LiveDRBenchTask(
                key=key,
                category=row["category"],
                question=row["question"],
                ground_truths=json.loads(row["ground_truths"]),
                eval_info=misc.get("eval_info", {}),
            )
        )
    return tasks, sha256(body).hexdigest()


def run_livedrbench_pilot(
    *,
    manifest_path: Path,
    config: HarnessConfig,
    output_root: Path,
) -> BenchmarkPilotSummary:
    manifest = load_livedrbench_manifest(manifest_path)
    tasks, dataset_response_hash = fetch_livedrbench_tasks(manifest)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / manifest.benchmark_id / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    questions_hash = sha256(
        json.dumps(
            [{"key": task.key, "category": task.category, "question": task.question} for task in tasks],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(output_dir / "manifest.snapshot.json", manifest.model_dump(mode="json"))

    provider = provider_from_config(config)
    results: list[BenchmarkTaskResult] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task in tasks:
        task_output = output_dir / "runs" / f"key-{task.key}"
        pipeline = BenchmarkResearchPipeline(
            provider=provider,
            collector=live_collector_from_config(config.search),
            output_dir=task_output,
            max_evidence=config.run.max_evidence,
            budget_limits=config.run.budget,
            report_language="auto",
        )
        try:
            state = pipeline.run(task.question, decision_context=_benchmark_context(manifest))
            prediction = _coerce_prediction_shape(state.structured_answer, task.ground_truths)
            metrics = exact_main_claim_metrics(task, prediction)
            predictions.append({"key": task.key, "preds": prediction})
            results.append(
                BenchmarkTaskResult(
                    key=task.key,
                    category=task.category,
                    status="succeeded",
                    run_id=state.run_id,
                    state_path=str(task_output / state.run_id / "state.json"),
                    report_path=state.report_path,
                    structured_output_present=state.structured_answer is not None,
                    official_shape_compatible=official_shape_compatible(task, prediction),
                    total_tokens=state.total_usage.input_tokens + state.total_usage.output_tokens,
                    estimated_cost_usd=state.total_usage.estimated_cost_usd,
                    traced_latency_ms=sum(event.latency_ms for event in state.trace),
                    search_calls=sum(event.stage == "search" for event in state.trace),
                    fetched_sources=sum(event.stage == "fetch" and event.outcome == "ok" for event in state.trace),
                    fetch_errors=sum(event.stage == "fetch" and event.outcome == "error" for event in state.trace),
                    exact_main_claim=metrics,
                )
            )
        except Exception as error:
            state_path = _latest_state_path(task_output)
            failed_state = _load_failed_state(state_path)
            results.append(
                BenchmarkTaskResult(
                    key=task.key,
                    category=task.category,
                    status="failed",
                    run_id=failed_state.run_id if failed_state else None,
                    state_path=str(state_path) if state_path else None,
                    report_path=failed_state.report_path if failed_state else None,
                    error=str(error),
                    structured_output_present=(
                        failed_state.structured_answer is not None if failed_state else False
                    ),
                    total_tokens=(
                        failed_state.total_usage.input_tokens + failed_state.total_usage.output_tokens
                        if failed_state
                        else 0
                    ),
                    estimated_cost_usd=(failed_state.total_usage.estimated_cost_usd if failed_state else 0),
                    traced_latency_ms=(sum(event.latency_ms for event in failed_state.trace) if failed_state else 0),
                    search_calls=(sum(event.stage == "search" for event in failed_state.trace) if failed_state else 0),
                    fetched_sources=(
                        sum(event.stage == "fetch" and event.outcome == "ok" for event in failed_state.trace)
                        if failed_state
                        else 0
                    ),
                    fetch_errors=(
                        sum(event.stage == "fetch" and event.outcome == "error" for event in failed_state.trace)
                        if failed_state
                        else 0
                    ),
                    exact_main_claim=exact_main_claim_metrics(task, []),
                )
            )
        _write_json(output_dir / "predictions.partial.json", predictions)
        _write_json(output_dir / "task_results.partial.json", [item.model_dump(mode="json") for item in results])

    completed = sum(item.status == "succeeded" for item in results)
    failed = len(results) - completed
    scored = [item.exact_main_claim for item in results if item.exact_main_claim is not None]
    summary = BenchmarkPilotSummary(
        benchmark_id=manifest.benchmark_id,
        status="completed_with_failures" if failed else "succeeded",
        output_dir=str(output_dir),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_response_sha256=dataset_response_hash,
        selected_questions_sha256=questions_hash,
        source_repository_commit=manifest.source_repository_commit,
        provider_kind=config.provider.kind,
        model=provider.model,
        thinking_mode=config.provider.thinking_mode,
        evaluator=manifest.evaluator,
        official_evaluator_status=manifest.official_evaluator_status,
        task_count=len(tasks),
        completed=completed,
        failed=failed,
        structured_output_rate=sum(item.structured_output_present for item in results) / len(results),
        official_shape_compatible_rate=sum(item.official_shape_compatible for item in results) / len(results),
        macro_exact_precision=_mean([item.precision for item in scored]),
        macro_exact_recall=_mean([item.recall for item in scored]),
        macro_exact_f1=_mean([item.f1 for item in scored]),
        total_tokens=sum(item.total_tokens for item in results),
        total_estimated_cost_usd=sum(item.estimated_cost_usd for item in results),
        wall_time_seconds=time.perf_counter() - started,
        results=results,
    )
    _write_json(output_dir / "predictions.json", predictions)
    _write_json(output_dir / "summary.json", summary.model_dump(mode="json"))
    return summary


def score_livedrbench_predictions(
    *,
    manifest_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> CompatibilityScoreArtifact:
    manifest = load_livedrbench_manifest(manifest_path)
    tasks, dataset_response_hash = fetch_livedrbench_tasks(manifest)
    raw_predictions = predictions_path.read_bytes()
    payload = json.loads(raw_predictions)
    predictions = {
        item["key"]: item.get("preds")
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("key"), int)
    }
    rows: list[CompatibilityScoreRow] = []
    for task in tasks:
        prediction = predictions.get(task.key, [])
        rows.append(
            CompatibilityScoreRow(
                key=task.key,
                category=task.category,
                official_shape_compatible=official_shape_compatible(task, prediction),
                exact_main_claim=exact_main_claim_metrics(task, prediction),
            )
        )
    artifact = CompatibilityScoreArtifact(
        benchmark_id=manifest.benchmark_id,
        dataset_revision=manifest.dataset_revision,
        dataset_response_sha256=dataset_response_hash,
        predictions_sha256=sha256(raw_predictions).hexdigest(),
        task_count=len(tasks),
        prediction_coverage_rate=sum(task.key in predictions for task in tasks) / len(tasks),
        official_shape_compatible_rate=sum(row.official_shape_compatible for row in rows) / len(rows),
        macro_exact_precision=_mean([row.exact_main_claim.precision for row in rows]),
        macro_exact_recall=_mean([row.exact_main_claim.recall for row in rows]),
        macro_exact_f1=_mean([row.exact_main_claim.f1 for row in rows]),
        evaluator=manifest.evaluator,
        official_evaluator_status=manifest.official_evaluator_status,
        results=rows,
    )
    _write_json(output_path, artifact.model_dump(mode="json"))
    return artifact


def exact_main_claim_metrics(task: LiveDRBenchTask, prediction: JsonValue) -> ExactClaimMetrics:
    fields = _comparison_fields(task)
    reference = _extract_claims(task.ground_truths, fields)
    predicted = _extract_claims(prediction, fields)
    matches = len(reference & predicted)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ExactClaimMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        matches=matches,
        predicted_claims=len(predicted),
        reference_claims=len(reference),
        compared_fields=sorted(fields),
    )


def official_shape_compatible(task: LiveDRBenchTask, prediction: JsonValue) -> bool:
    references = task.ground_truths
    if not isinstance(references, list) or not isinstance(prediction, list):
        return False
    if len(references) != len(prediction):
        return False
    for reference, predicted in zip(references, prediction):
        if type(reference) is not type(predicted):
            return False
        if task.category in {"entities", "prior-art"}:
            if not isinstance(predicted, list):
                return False
            expected_type = str if task.category == "entities" else dict
            if any(not isinstance(item, expected_type) for item in predicted):
                return False
        elif task.category.startswith("scifacts-"):
            if not isinstance(predicted, list) or any(not isinstance(item, dict) for item in predicted):
                return False
    return True


def _comparison_fields(task: LiveDRBenchTask) -> set[str]:
    requested = task.eval_info.get("main_claims") or task.eval_info.get("primary_keys")
    if requested:
        return {_canonical_field(str(item)) for item in requested}
    defaults = {
        "entities": set(),
        "scifacts-geo": {"papertitle"},
        "scifacts-materials": {"papertitle", "material"},
        "novel-datasets-identification": {"title"},
        "novel-datasets-identi-extraction": set(),
        "novel-datasets-peer": {"name"},
        "prior-art": {"title"},
        "flights": {"timeutc", "attemptnumber"},
    }
    return defaults.get(task.category, set())


def _extract_claims(value: JsonValue, fields: set[str]) -> set[str]:
    claims: set[str] = set()

    def visit(item: Any, active_field: str | None = None) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                field = _canonical_field(str(key))
                if not fields or field in fields:
                    visit(child, field if fields else None)
                else:
                    visit(child, active_field)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, active_field)
            return
        if item is None or (fields and active_field is None):
            return
        normalized = _normalize_value(item)
        if normalized:
            claims.add(f"{active_field}:{normalized}" if active_field else normalized)

    visit(value)
    return claims


def _coerce_prediction_shape(answer: JsonValue | None, ground_truths: JsonValue) -> JsonValue:
    if answer is None:
        return []
    if not isinstance(ground_truths, list) or not ground_truths:
        return answer
    expected = ground_truths[0]
    if len(ground_truths) == 1:
        if isinstance(expected, list):
            if isinstance(answer, list) and (not answer or not isinstance(answer[0], list)):
                return [answer]
        elif isinstance(expected, dict) and isinstance(answer, dict):
            return [answer]
    return answer


def _canonical_field(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", value.casefold())
    aliases = {
        "attempt": "attemptnumber",
        "attemptno": "attemptnumber",
        "attemptnumber": "attemptnumber",
        "timeutc": "timeutc",
    }
    return aliases.get(compact, compact)


def _normalize_value(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _json_object(value: str, *, field: str, key: int) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"LiveDRBench {field} must be an object for key {key}")
    return parsed


def _benchmark_context(manifest: LiveDRBenchManifest) -> str:
    return (
        f"Compatibility pilot for {manifest.dataset_id} at revision {manifest.dataset_revision}. "
        "The model receives only the public question and fetched evidence, never benchmark ground truth. "
        "Use one-pass B1 and follow the question's requested JSON output exactly."
    )


def _latest_state_path(task_output: Path) -> Path | None:
    candidates = sorted(task_output.glob("*/state.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_failed_state(path: Path | None) -> RunState | None:
    if path is None:
        return None
    try:
        return RunState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
