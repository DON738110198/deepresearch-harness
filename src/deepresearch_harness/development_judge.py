from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DevelopmentGoldSlice
from .browsecomp_plus import load_pi_browsecomp_run, normalized_text_file_sha256
from .pi_browsecomp import PiSmokeSummary, validate_pi_attempt_archives
from .screening_judge import (
    ScreeningInference,
    VllmChatClient,
    calibrate_screening_judge,
    create_judge_prompt,
    load_screening_manifest,
    parse_judge_response,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DevelopmentJudgeObservation(StrictContract):
    query_id: str = Field(min_length=1)
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correct: bool | None
    confidence: float | None = Field(default=None, ge=0, le=100)
    parse_error: bool
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    result_path: str = Field(min_length=1)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DevelopmentJudgeResult(StrictContract):
    schema_version: Literal["browsecomp-plus-development-service-judge-v0"] = (
        "browsecomp-plus-development-service-judge-v0"
    )
    created_at: str
    status: Literal["succeeded", "failed"]
    metric_status: Literal["calibrated_development_diagnostic_not_official"] = (
        "calibrated_development_diagnostic_not_official"
    )
    judge_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_slice_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_model: Literal["Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-32B"]
    judge_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    served_model_name: str = Field(min_length=1)
    inference: ScreeningInference
    query_count: int = Field(gt=0)
    evaluations: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy_percent: float | None = Field(default=None, ge=0, le=100)
    parse_failures: int = Field(ge=0)
    request_failures: int = Field(ge=0)
    request_errors: list[str]
    elapsed_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    observations: list[DevelopmentJudgeObservation]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def counts_match_observations(self) -> "DevelopmentJudgeResult":
        if self.evaluations != len(self.observations):
            raise ValueError("development judge evaluation count differs")
        if self.correct != sum(row.correct is True for row in self.observations):
            raise ValueError("development judge correct count differs")
        if self.parse_failures != sum(row.parse_error for row in self.observations):
            raise ValueError("development judge parse-failure count differs")
        if self.request_failures != len(self.request_errors):
            raise ValueError("development judge request-failure count differs")
        if self.query_count != self.evaluations + self.request_failures:
            raise ValueError("development judge query accounting differs")
        valid_labels = sum(row.correct is not None for row in self.observations)
        expected_accuracy = (
            round(self.correct / valid_labels * 100, 6) if valid_labels else None
        )
        if self.accuracy_percent != expected_accuracy:
            raise ValueError("development judge accuracy differs from labels")
        if self.prompt_tokens != sum(row.prompt_tokens for row in self.observations):
            raise ValueError("development judge prompt-token count differs")
        if self.completion_tokens != sum(
            row.completion_tokens for row in self.observations
        ):
            raise ValueError("development judge completion-token count differs")
        if self.total_tokens != sum(row.total_tokens for row in self.observations):
            raise ValueError("development judge total-token count differs")
        return self


def run_development_service_judge(
    *,
    judge_manifest_path: Path,
    calibration_result_path: Path,
    reference_screening_result_path: Path,
    official_comparison_path: Path,
    source_dir: Path,
    gold_slice_path: Path,
    output_dir: Path,
    base_url: str,
    concurrency: int = 16,
    timeout_seconds: float = 600,
    retries: int = 2,
) -> DevelopmentJudgeResult:
    if concurrency < 1:
        raise ValueError("development judge concurrency must be positive")
    repository_root = _find_repository_root(source_dir.resolve())
    if not output_dir.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("development judge output must stay under ignored runs/")
    if output_dir.exists():
        raise ValueError("development judge output directory must not exist")

    manifest = load_screening_manifest(judge_manifest_path)
    calibration = calibrate_screening_judge(
        screening_manifest_path=judge_manifest_path,
        screening_result_path=reference_screening_result_path,
        official_comparison_path=official_comparison_path,
        output_path=calibration_result_path,
        validate_existing=True,
    )
    if calibration.get("status") != "accepted_for_development_screening":
        raise ValueError("persistent judge calibration was not accepted")
    if calibration.get("screening_manifest_sha256") != normalized_text_file_sha256(
        judge_manifest_path
    ):
        raise ValueError("persistent judge calibration targets another manifest")

    summary_path = source_dir / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    gold_bytes = gold_slice_path.read_bytes()
    gold = DevelopmentGoldSlice.model_validate_json(gold_bytes)
    if gold.source_summary_sha256 != sha256(summary_bytes).hexdigest():
        raise ValueError("development gold was not opened for this prediction summary")
    if gold.prediction_set_sha256 != _prediction_set_sha256(summary):
        raise ValueError("development gold prediction set hash does not match")
    if summary.target_manifest_sha256 != manifest.target_manifest_sha256:
        raise ValueError("development predictions target another benchmark")
    if gold.target_manifest_sha256 != manifest.target_manifest_sha256:
        raise ValueError("development gold targets another benchmark")

    gold_by_id = {row.query_id: row for row in gold.rows}
    inputs: list[dict[str, str]] = []
    for item in summary.items:
        reference = gold_by_id.get(item.query_id)
        if (
            reference is None
            or item.status != "succeeded"
            or item.run_sha256 is None
            or item.prediction_sha256 is None
        ):
            raise ValueError(f"query is not frozen and scoreable: {item.query_id}")
        query_root = source_dir / _safe_id(item.query_id)
        validate_pi_attempt_archives(query_root=query_root, item=item)
        run_path = query_root / "run.json"
        run_bytes = run_path.read_bytes()
        if sha256(run_bytes).hexdigest() != item.run_sha256:
            raise ValueError(f"source run hash mismatch for query {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        if sha256(run.answer_text.encode("utf-8")).hexdigest() != (
            item.prediction_sha256
        ):
            raise ValueError(f"prediction hash mismatch for query {item.query_id}")
        inputs.append(
            {
                "query_id": item.query_id,
                "question": reference.question,
                "answer": reference.answer,
                "response": run.answer_text,
                "prediction_sha256": item.prediction_sha256,
            }
        )
    if len(inputs) != summary.query_count or set(gold_by_id) != {
        row["query_id"] for row in inputs
    }:
        raise ValueError("development judge input grid differs from frozen gold")

    client = VllmChatClient(
        base_url=base_url,
        served_model_name=manifest.engine.served_model_name,
        inference=manifest.inference,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    if manifest.engine.served_model_name not in client.model_ids():
        raise ValueError("vLLM server does not expose the registered model name")

    output_dir.mkdir(parents=True)
    results_dir = output_dir / "results"
    results_dir.mkdir()
    started = time.perf_counter()
    observations: list[DevelopmentJudgeObservation] = []
    request_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _evaluate_input,
                row=row,
                results_dir=results_dir,
                client=client,
            ): row
            for row in inputs
        }
        for future in as_completed(futures):
            try:
                observations.append(future.result())
            except Exception as error:
                row = futures[future]
                request_errors.append(
                    f"{row['query_id']}:{type(error).__name__}:{error}"
                )
    observations.sort(key=lambda row: row.query_id)
    request_errors.sort()
    parse_failures = sum(row.parse_error for row in observations)
    valid_labels = sum(row.correct is not None for row in observations)
    correct = sum(row.correct is True for row in observations)
    result = DevelopmentJudgeResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        status=(
            "succeeded"
            if not request_errors and parse_failures == 0
            else "failed"
        ),
        judge_manifest_sha256=normalized_text_file_sha256(judge_manifest_path),
        calibration_sha256=sha256(calibration_result_path.read_bytes()).hexdigest(),
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        gold_slice_sha256=sha256(gold_bytes).hexdigest(),
        judge_model=manifest.judge.model,
        judge_revision=manifest.judge.revision,
        served_model_name=manifest.engine.served_model_name,
        inference=manifest.inference,
        query_count=len(inputs),
        evaluations=len(observations),
        correct=correct,
        accuracy_percent=(
            round(correct / valid_labels * 100, 6) if valid_labels else None
        ),
        parse_failures=parse_failures,
        request_failures=len(request_errors),
        request_errors=request_errors,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        prompt_tokens=sum(row.prompt_tokens for row in observations),
        completion_tokens=sum(row.completion_tokens for row in observations),
        total_tokens=sum(row.total_tokens for row in observations),
        observations=observations,
        claim_boundary=manifest.claim_boundary,
    )
    _atomic_json(output_dir / "development_judge_result.json", result.model_dump(mode="json"))
    return result


def _evaluate_input(*, row, results_dir, client) -> DevelopmentJudgeObservation:
    prompt = create_judge_prompt(row["question"], row["response"], row["answer"])
    started = time.perf_counter()
    judge_text, usage = client.judge(prompt)
    latency_ms = round((time.perf_counter() - started) * 1000)
    parsed = parse_judge_response(judge_text)
    prompt_tokens = _usage_int(usage, "prompt_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    payload = {
        "schema_version": "browsecomp-plus-development-service-judge-item-v0",
        "query_id": row["query_id"],
        "response": row["response"],
        "correct_answer": row["answer"],
        "judge_prompt": prompt,
        "judge_response": judge_text,
        "judge_result": parsed,
        "usage": usage,
        "latency_ms": latency_ms,
    }
    result_path = results_dir / f"{_safe_id(row['query_id'])}_eval.json"
    _atomic_json(result_path, payload)
    return DevelopmentJudgeObservation(
        query_id=row["query_id"],
        prediction_sha256=row["prediction_sha256"],
        correct=parsed["correct"],
        confidence=parsed["confidence"],
        parse_error=bool(parsed["parse_error"]),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        result_path=result_path.relative_to(results_dir.parent).as_posix(),
        result_sha256=sha256(result_path.read_bytes()).hexdigest(),
    )


def _usage_int(usage: dict[str, object], field_name: str) -> int:
    value = usage.get(field_name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return 0
    return int(value)


def _prediction_set_sha256(summary: PiSmokeSummary) -> str:
    canonical = "\n".join(
        f"{item.query_id}\t{item.prediction_sha256}\t{item.run_sha256}"
        for item in sorted(summary.items, key=lambda row: row.query_id)
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not safe or safe in {".", ".."}:
        raise ValueError("query ID cannot be represented safely")
    return safe


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "runs"
        ).is_dir():
            return candidate
    raise ValueError("could not locate repository root")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
