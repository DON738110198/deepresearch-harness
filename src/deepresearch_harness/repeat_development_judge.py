from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_judge import (
    OfficialJudgeObservation,
    OfficialJudgePairedSummary,
    OfficialJudgeTrialScore,
    OfficialJudgeVariantAggregate,
)
from .browsecomp_plus import normalized_text_file_sha256
from .browsecomp_repeats import (
    MetricDistribution,
    RepeatComparisonSummary,
    RepeatExperimentManifest,
    aggregate_repeat_experiment,
)
from .development_judge import (
    DevelopmentJudgeResult,
    run_development_service_judge,
)
from .screening_judge import ScreeningInference, load_screening_manifest


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepeatJudgeResultRef(StrictContract):
    trial_id: str = Field(min_length=1)
    execution_order: Literal["baseline_first", "candidate_first"]
    variant: Literal["baseline", "candidate"]
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RepeatDevelopmentJudgeComparison(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-repeat-development-service-judge-v0"
    ] = "browsecomp-plus-repeat-development-service-judge-v0"
    created_at: str
    status: Literal["calibrated_development_diagnostic_not_official"] = (
        "calibrated_development_diagnostic_not_official"
    )
    leaderboard_status: Literal["not_submitted"] = "not_submitted"
    repeat_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeat_comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_model: Literal["Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-32B"]
    judge_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    served_model_name: str = Field(min_length=1)
    inference: ScreeningInference
    trial_count: int = Field(ge=3)
    queries_per_trial: int = Field(gt=0)
    evaluations: int = Field(gt=0)
    parse_failures: Literal[0] = 0
    request_failures: Literal[0] = 0
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    baseline: OfficialJudgeVariantAggregate
    candidate: OfficialJudgeVariantAggregate
    paired: OfficialJudgePairedSummary
    trials: list[OfficialJudgeTrialScore] = Field(min_length=3)
    observations: list[OfficialJudgeObservation] = Field(min_length=1)
    result_refs: list[RepeatJudgeResultRef] = Field(min_length=6)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def grid_is_complete(self) -> "RepeatDevelopmentJudgeComparison":
        expected = self.trial_count * self.queries_per_trial * 2
        if self.evaluations != expected or len(self.observations) != expected:
            raise ValueError("repeat development judge evaluation grid is incomplete")
        if len(self.trials) != self.trial_count:
            raise ValueError("repeat development judge trial count differs")
        if len(self.result_refs) != self.trial_count * 2:
            raise ValueError("repeat development judge result references differ")
        if self.baseline.evaluations != expected // 2:
            raise ValueError("repeat development judge baseline count differs")
        if self.candidate.evaluations != expected // 2:
            raise ValueError("repeat development judge candidate count differs")
        if self.paired.comparisons != expected // 2:
            raise ValueError("repeat development judge paired count differs")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("repeat development judge token totals differ")
        keys = {
            (row.trial_id, row.variant, row.query_id)
            for row in self.observations
        }
        if len(keys) != expected:
            raise ValueError("repeat development judge observations are duplicated")
        return self


def run_repeat_development_service_judge(
    *,
    repeat_experiment_path: Path,
    repeat_comparison_path: Path,
    target_manifest_path: Path,
    judge_manifest_path: Path,
    calibration_result_path: Path,
    reference_screening_result_path: Path,
    official_comparison_path: Path,
    output_dir: Path,
    base_url: str,
    concurrency: int = 16,
    timeout_seconds: float = 600,
    retries: int = 2,
) -> RepeatDevelopmentJudgeComparison:
    repository_root = _find_repository_root(repeat_experiment_path.resolve())
    _require_under_runs(output_dir, repository_root)
    if output_dir.exists():
        raise ValueError("repeat development judge output directory must not exist")
    output_dir.mkdir(parents=True)

    experiment = RepeatExperimentManifest.model_validate_json(
        repeat_experiment_path.read_text(encoding="utf-8")
    )
    for pair in experiment.pairs:
        for variant in ("baseline", "candidate"):
            inputs = getattr(pair, variant)
            summary_path = _resolve_repository_file(
                inputs.summary_path, repository_root, require_runs=True
            )
            gold_path = _resolve_repository_file(
                inputs.gold_slice_path, repository_root, require_runs=True
            )
            run_development_service_judge(
                judge_manifest_path=judge_manifest_path,
                calibration_result_path=calibration_result_path,
                reference_screening_result_path=reference_screening_result_path,
                official_comparison_path=official_comparison_path,
                source_dir=summary_path.parent,
                gold_slice_path=gold_path,
                output_dir=output_dir / "trials" / f"{pair.trial_id}-{variant}",
                base_url=base_url,
                concurrency=concurrency,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )

    return aggregate_repeat_development_judge(
        repeat_experiment_path=repeat_experiment_path,
        repeat_comparison_path=repeat_comparison_path,
        target_manifest_path=target_manifest_path,
        judge_manifest_path=judge_manifest_path,
        calibration_result_path=calibration_result_path,
        results_root=output_dir / "trials",
        output_path=output_dir / "repeat_development_judge_comparison.json",
    )


def aggregate_repeat_development_judge(
    *,
    repeat_experiment_path: Path,
    repeat_comparison_path: Path,
    target_manifest_path: Path,
    judge_manifest_path: Path,
    calibration_result_path: Path,
    results_root: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> RepeatDevelopmentJudgeComparison:
    repository_root = _find_repository_root(repeat_experiment_path.resolve())
    for path in (
        repeat_experiment_path,
        repeat_comparison_path,
        calibration_result_path,
        results_root,
        output_path,
    ):
        _require_under_runs(path, repository_root)
    existing = None
    if output_path.exists() and not validate_existing:
        raise ValueError("repeat development judge comparison already exists")
    if output_path.exists():
        existing = RepeatDevelopmentJudgeComparison.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

    repeat = aggregate_repeat_experiment(
        manifest_path=repeat_experiment_path,
        target_manifest_path=target_manifest_path,
        output_path=repeat_comparison_path,
        validate_existing=True,
    )
    experiment_bytes = repeat_experiment_path.read_bytes()
    experiment = RepeatExperimentManifest.model_validate_json(experiment_bytes)
    manifest = load_screening_manifest(judge_manifest_path)
    calibration_bytes = calibration_result_path.read_bytes()
    calibration = _load_calibration(calibration_bytes)
    manifest_hash = normalized_text_file_sha256(judge_manifest_path)
    if calibration.get("status") != "accepted_for_development_screening":
        raise ValueError("persistent judge calibration is not accepted")
    if calibration.get("screening_manifest_sha256") != manifest_hash:
        raise ValueError("persistent judge calibration targets another manifest")
    if repeat.experiment_manifest_sha256 != normalized_text_file_sha256(
        repeat_experiment_path
    ):
        raise ValueError("repeat comparison targets another experiment")

    result_refs: list[RepeatJudgeResultRef] = []
    results: dict[tuple[str, str], DevelopmentJudgeResult] = {}
    observations: list[OfficialJudgeObservation] = []
    prompt_tokens = 0
    completion_tokens = 0
    for pair in experiment.pairs:
        for variant in ("baseline", "candidate"):
            inputs = getattr(pair, variant)
            summary_path = _resolve_repository_file(
                inputs.summary_path, repository_root, require_runs=True
            )
            gold_path = _resolve_repository_file(
                inputs.gold_slice_path, repository_root, require_runs=True
            )
            result_path = (
                results_root
                / f"{pair.trial_id}-{variant}"
                / "development_judge_result.json"
            )
            result_bytes = result_path.read_bytes()
            result = DevelopmentJudgeResult.model_validate_json(result_bytes)
            if result.status != "succeeded":
                raise ValueError("repeat development judge contains a failed variant")
            if result.parse_failures or result.request_failures:
                raise ValueError("repeat development judge contains judge failures")
            if result.judge_manifest_sha256 != manifest_hash:
                raise ValueError("repeat development judge changes judge manifest")
            if result.calibration_sha256 != sha256(calibration_bytes).hexdigest():
                raise ValueError("repeat development judge changes calibration")
            if result.source_summary_sha256 != sha256(
                summary_path.read_bytes()
            ).hexdigest():
                raise ValueError("repeat development judge changes source summary")
            if result.gold_slice_sha256 != sha256(gold_path.read_bytes()).hexdigest():
                raise ValueError("repeat development judge changes gold slice")
            if (
                result.judge_model != manifest.judge.model
                or result.judge_revision != manifest.judge.revision
                or result.inference != manifest.inference
            ):
                raise ValueError("repeat development judge changes model contract")
            if result.evaluations != repeat.queries_per_trial:
                raise ValueError("repeat development judge query count differs")
            result_refs.append(
                RepeatJudgeResultRef(
                    trial_id=pair.trial_id,
                    execution_order=pair.execution_order,
                    variant=variant,
                    path=result_path.relative_to(repository_root).as_posix(),
                    sha256=sha256(result_bytes).hexdigest(),
                )
            )
            results[(pair.trial_id, variant)] = result
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            observations.extend(
                OfficialJudgeObservation(
                    trial_id=pair.trial_id,
                    execution_order=pair.execution_order,
                    variant=variant,
                    query_id=row.query_id,
                    correct=bool(row.correct),
                    confidence=row.confidence,
                    prediction_sha256=row.prediction_sha256,
                    result_path=(
                        result_path.parent / row.result_path
                    ).relative_to(repository_root).as_posix(),
                    result_sha256=row.result_sha256,
                )
                for row in _validated_observations(result, result_path.parent)
            )

    trials: list[OfficialJudgeTrialScore] = []
    correctness = {
        (row.trial_id, row.query_id, row.variant): row.correct
        for row in observations
    }
    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    for pair in experiment.pairs:
        baseline_result = results[(pair.trial_id, "baseline")]
        candidate_result = results[(pair.trial_id, "candidate")]
        baseline_ids = {row.query_id for row in baseline_result.observations}
        candidate_ids = {row.query_id for row in candidate_result.observations}
        if baseline_ids != candidate_ids:
            raise ValueError("repeat development judge paired query IDs differ")
        baseline_correct = baseline_result.correct
        candidate_correct = candidate_result.correct
        trials.append(
            OfficialJudgeTrialScore(
                trial_id=pair.trial_id,
                execution_order=pair.execution_order,
                baseline_correct=baseline_correct,
                candidate_correct=candidate_correct,
                query_count=len(baseline_ids),
                baseline_accuracy_percent=round(
                    baseline_correct / len(baseline_ids) * 100, 6
                ),
                candidate_accuracy_percent=round(
                    candidate_correct / len(candidate_ids) * 100, 6
                ),
            )
        )
        for query_id in baseline_ids:
            baseline_value = correctness[(pair.trial_id, query_id, "baseline")]
            candidate_value = correctness[(pair.trial_id, query_id, "candidate")]
            if candidate_value and not baseline_value:
                candidate_wins += 1
            elif baseline_value and not candidate_value:
                baseline_wins += 1
            else:
                ties += 1

    artifact = RepeatDevelopmentJudgeComparison(
        created_at=(
            existing.created_at
            if existing is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        repeat_experiment_sha256=sha256(experiment_bytes).hexdigest(),
        repeat_comparison_sha256=sha256(
            repeat_comparison_path.read_bytes()
        ).hexdigest(),
        judge_manifest_sha256=manifest_hash,
        calibration_sha256=sha256(calibration_bytes).hexdigest(),
        judge_model=manifest.judge.model,
        judge_revision=manifest.judge.revision,
        served_model_name=manifest.engine.served_model_name,
        inference=manifest.inference,
        trial_count=repeat.trial_count,
        queries_per_trial=repeat.queries_per_trial,
        evaluations=len(observations),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        baseline=_variant_aggregate("baseline", observations, trials),
        candidate=_variant_aggregate("candidate", observations, trials),
        paired=OfficialJudgePairedSummary(
            comparisons=repeat.trial_count * repeat.queries_per_trial,
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            ties=ties,
        ),
        trials=trials,
        observations=sorted(
            observations, key=lambda row: (row.trial_id, row.variant, row.query_id)
        ),
        result_refs=sorted(
            result_refs, key=lambda row: (row.trial_id, row.variant)
        ),
        claim_boundary=(
            "Calibrated development Judge diagnostic using the frozen official "
            "grader prompt and model contract. It is not an official evaluator "
            "execution, leaderboard score, or model-capability claim."
        ),
    )
    if existing is not None:
        if artifact != existing:
            raise ValueError(
                "existing repeat development judge comparison changed with sources"
            )
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return artifact


def _variant_aggregate(
    variant: Literal["baseline", "candidate"],
    observations: list[OfficialJudgeObservation],
    trials: list[OfficialJudgeTrialScore],
) -> OfficialJudgeVariantAggregate:
    selected = [row for row in observations if row.variant == variant]
    correct = sum(row.correct for row in selected)
    values = [
        (
            row.baseline_accuracy_percent
            if variant == "baseline"
            else row.candidate_accuracy_percent
        )
        for row in trials
    ]
    return OfficialJudgeVariantAggregate(
        variant=variant,
        correct=correct,
        evaluations=len(selected),
        pooled_accuracy_percent=round(correct / len(selected) * 100, 6),
        trial_accuracy_percent=_distribution(values),
    )


def _distribution(values: list[float]) -> MetricDistribution:
    return MetricDistribution(
        trials=len(values),
        mean=round(mean(values), 6),
        sample_stddev=round(stdev(values), 6),
        minimum=round(min(values), 6),
        maximum=round(max(values), 6),
    )


def _load_calibration(raw: bytes) -> dict[str, object]:
    import json

    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("persistent judge calibration must be an object")
    return value


def _validated_observations(
    result: DevelopmentJudgeResult, result_root: Path
):
    for row in result.observations:
        item_path = (result_root / row.result_path).resolve()
        if not item_path.is_relative_to(result_root.resolve()) or not item_path.is_file():
            raise ValueError("repeat development judge item result is missing")
        if sha256(item_path.read_bytes()).hexdigest() != row.result_sha256:
            raise ValueError("repeat development judge item result hash changed")
        yield row


def _resolve_repository_file(
    value: str, repository_root: Path, *, require_runs: bool
) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("repeat judge source paths must be repository-relative")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise ValueError("repeat judge source path escapes the repository")
    if require_runs and not resolved.is_relative_to(
        (repository_root / "runs").resolve()
    ):
        raise ValueError("repeat judge source must stay under ignored runs/")
    if not resolved.is_file():
        raise ValueError(f"repeat judge source is missing: {value}")
    return resolved


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "runs"
        ).is_dir():
            return candidate
    raise ValueError("could not locate repository root for repeat development judge")


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("repeat development judge artifacts must stay under runs/")
