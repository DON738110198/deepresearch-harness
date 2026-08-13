from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import (
    OfficialGroundTruthExport,
    export_official_development_ground_truth,
)
from .browsecomp_plus import (
    OfficialJudgeInference,
    load_official_evaluator_manifest,
    normalized_text_file_sha256,
)
from .browsecomp_repeats import (
    MetricDistribution,
    RepeatExperimentManifest,
    aggregate_repeat_experiment,
)
from .pi_browsecomp import OfficialRunExportManifest


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeBatchArtifact(StrictContract):
    role: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def path_is_relative(self) -> "JudgeBatchArtifact":
        _validate_relative_path(self.path)
        return self


class OfficialJudgeBatchItem(StrictContract):
    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    trial_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    execution_order: Literal["baseline_first", "candidate_first"]
    variant: Literal["baseline", "candidate"]
    query_id: str = Field(min_length=1)
    source_export_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    staged_input_path: str = Field(min_length=1)
    staged_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def staged_input_is_an_exact_copy(self) -> "OfficialJudgeBatchItem":
        _validate_relative_path(self.staged_input_path)
        if self.source_input_sha256 != self.staged_input_sha256:
            raise ValueError("official judge staged input must be an exact source copy")
        return self


class OfficialJudgeBatchManifest(StrictContract):
    schema_version: Literal["browsecomp-plus-official-judge-batch-v0"] = (
        "browsecomp-plus-official-judge-batch-v0"
    )
    created_at: str
    status: Literal["prepared_not_run"] = "prepared_not_run"
    generator_model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    repeat_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeat_comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    official_evaluator_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_assets_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    evaluator_script: Literal["scripts_evaluation/evaluate_run.py"]
    evaluator_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_model: Literal["Qwen/Qwen3-32B"]
    judge_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    inference: OfficialJudgeInference
    registration_status: Literal[
        "pre_generation", "reconstructed_after_interruption"
    ]
    trial_count: int = Field(ge=3)
    queries_per_trial: int = Field(gt=0)
    input_count: int = Field(gt=0)
    input_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_export_manifest_path: str = Field(min_length=1)
    ground_truth_export_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_path: str = Field(min_length=1)
    ground_truth_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_artifacts: list[JudgeBatchArtifact] = Field(min_length=4)
    items: list[OfficialJudgeBatchItem] = Field(min_length=1)

    @model_validator(mode="after")
    def batch_shape_is_complete(self) -> "OfficialJudgeBatchManifest":
        _validate_relative_path(self.ground_truth_export_manifest_path)
        _validate_relative_path(self.ground_truth_path)
        if self.input_count != len(self.items):
            raise ValueError("official judge input_count does not match items")
        expected_count = self.trial_count * self.queries_per_trial * 2
        if self.input_count != expected_count:
            raise ValueError("official judge batch is not a complete paired grid")
        batch_ids = [item.batch_id for item in self.items]
        staged_paths = [item.staged_input_path for item in self.items]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("official judge batch IDs must be unique")
        if len(staged_paths) != len(set(staged_paths)):
            raise ValueError("official judge staged input paths must be unique")
        roles = [artifact.role for artifact in self.audit_artifacts]
        paths = [artifact.path for artifact in self.audit_artifacts]
        if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
            raise ValueError("official judge audit artifacts must be unique")

        by_trial: dict[str, list[OfficialJudgeBatchItem]] = {}
        for item in self.items:
            by_trial.setdefault(item.trial_id, []).append(item)
        if len(by_trial) != self.trial_count:
            raise ValueError("official judge trial_count does not match items")
        expected_queries: set[str] | None = None
        for trial_items in by_trial.values():
            orders = {item.execution_order for item in trial_items}
            if len(orders) != 1:
                raise ValueError("official judge trial changes execution order")
            queries = {item.query_id for item in trial_items}
            if len(queries) != self.queries_per_trial:
                raise ValueError("official judge trial query count differs")
            if expected_queries is None:
                expected_queries = queries
            elif queries != expected_queries:
                raise ValueError("official judge trials use different query sets")
            cells = {(item.query_id, item.variant) for item in trial_items}
            expected_cells = {
                (query_id, variant)
                for query_id in queries
                for variant in ("baseline", "candidate")
            }
            if cells != expected_cells:
                raise ValueError("official judge trial is missing a paired variant")
        if self.input_set_sha256 != _input_set_sha256(self.items):
            raise ValueError("official judge input-set hash does not match items")
        return self


class JudgeRuntimeVersions(StrictContract):
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    transformers: str = Field(min_length=1)
    vllm: str = Field(min_length=1)
    cuda_available: Literal[True]
    cuda_device_count: int = Field(ge=2)


class JudgeGpuDevice(StrictContract):
    index: int = Field(ge=0)
    uuid: str = Field(min_length=1)
    memory_used_mib: int = Field(ge=0)
    memory_total_mib: int = Field(gt=0)
    utilization_percent: int = Field(ge=0, le=100)
    compute_pids: list[int]


class JudgeGpuCheck(StrictContract):
    captured_at: str
    devices: list[JudgeGpuDevice] = Field(min_length=2)


class OfficialJudgeExecutionRegistration(StrictContract):
    schema_version: Literal["browsecomp-plus-official-judge-execution-v0"] = (
        "browsecomp-plus-official-judge-execution-v0"
    )
    registered_at: str
    status: Literal["registered_pre_inference"] = "registered_pre_inference"
    batch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    official_evaluator_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    launcher_path: str = Field(min_length=1)
    launcher_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_verification_path: str = Field(min_length=1)
    asset_verification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_assets_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_clean: Literal[True]
    evaluator_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_model: Literal["Qwen/Qwen3-32B"]
    judge_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_dir: str = Field(min_length=1)
    runtime: JudgeRuntimeVersions
    gpu_ids: tuple[int, int]
    gpu_checks: list[JudgeGpuCheck] = Field(min_length=3)
    inference: OfficialJudgeInference
    batch_size: int = Field(gt=0)
    tensor_parallel_size: Literal[2]
    environment_overrides: dict[str, str] = Field(default_factory=dict)
    command: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def selected_gpus_were_idle(self) -> "OfficialJudgeExecutionRegistration":
        _validate_relative_path(self.launcher_path)
        _validate_relative_path(self.asset_verification_path)
        if len(set(self.gpu_ids)) != 2:
            raise ValueError("official judge requires two distinct GPU IDs")
        for check in self.gpu_checks:
            by_index = {device.index: device for device in check.devices}
            if set(by_index) != set(self.gpu_ids):
                raise ValueError("official judge GPU check does not match selected IDs")
            for device in by_index.values():
                if (
                    device.memory_used_mib > 512
                    or device.utilization_percent > 5
                    or device.compute_pids
                ):
                    raise ValueError("official judge registered an occupied GPU")
        if set(self.environment_overrides) - {"NCCL_P2P_DISABLE"}:
            raise ValueError("official judge registered an unsupported environment override")
        if self.environment_overrides.get("NCCL_P2P_DISABLE") not in {None, "1"}:
            raise ValueError("NCCL_P2P_DISABLE must be exactly '1' when present")
        return self


class JudgeExecutionFile(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def path_is_relative(self) -> "JudgeExecutionFile":
        _validate_relative_path(self.path)
        return self


class OfficialJudgeExecutionResult(StrictContract):
    schema_version: Literal["browsecomp-plus-official-judge-execution-result-v0"] = (
        "browsecomp-plus-official-judge-execution-result-v0"
    )
    started_at: str
    completed_at: str
    status: Literal["succeeded", "failed"]
    exit_code: int
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stdout: JudgeExecutionFile
    stderr: JudgeExecutionFile
    output_files: list[JudgeExecutionFile]

    @model_validator(mode="after")
    def status_matches_exit_code(self) -> "OfficialJudgeExecutionResult":
        if (self.exit_code == 0) != (self.status == "succeeded"):
            raise ValueError("official judge execution status disagrees with exit code")
        paths = [item.path for item in self.output_files]
        if len(paths) != len(set(paths)):
            raise ValueError("official judge execution output paths must be unique")
        return self


class OfficialJudgeObservation(StrictContract):
    trial_id: str
    execution_order: Literal["baseline_first", "candidate_first"]
    variant: Literal["baseline", "candidate"]
    query_id: str
    correct: bool
    confidence: float | None = Field(default=None, ge=0, le=100)
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_path: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OfficialJudgeTrialScore(StrictContract):
    trial_id: str
    execution_order: Literal["baseline_first", "candidate_first"]
    baseline_correct: int = Field(ge=0)
    candidate_correct: int = Field(ge=0)
    query_count: int = Field(gt=0)
    baseline_accuracy_percent: float = Field(ge=0, le=100)
    candidate_accuracy_percent: float = Field(ge=0, le=100)


class OfficialJudgeVariantAggregate(StrictContract):
    variant: Literal["baseline", "candidate"]
    correct: int = Field(ge=0)
    evaluations: int = Field(gt=0)
    pooled_accuracy_percent: float = Field(ge=0, le=100)
    trial_accuracy_percent: MetricDistribution


class OfficialJudgePairedSummary(StrictContract):
    comparisons: int = Field(gt=0)
    candidate_wins: int = Field(ge=0)
    baseline_wins: int = Field(ge=0)
    ties: int = Field(ge=0)

    @model_validator(mode="after")
    def outcomes_match_count(self) -> "OfficialJudgePairedSummary":
        if self.candidate_wins + self.baseline_wins + self.ties != self.comparisons:
            raise ValueError("official judge paired outcomes do not match count")
        return self


class OfficialJudgeComparison(StrictContract):
    schema_version: Literal["browsecomp-plus-official-judge-comparison-v0"] = (
        "browsecomp-plus-official-judge-comparison-v0"
    )
    created_at: str
    status: Literal["official_evaluator_development_slice"] = (
        "official_evaluator_development_slice"
    )
    leaderboard_status: Literal["not_submitted"] = "not_submitted"
    batch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_model: Literal["Qwen/Qwen3-32B"]
    judge_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    inference: OfficialJudgeInference
    trial_count: int = Field(ge=3)
    queries_per_trial: int = Field(gt=0)
    evaluations: int = Field(gt=0)
    parse_failures: Literal[0] = 0
    baseline: OfficialJudgeVariantAggregate
    candidate: OfficialJudgeVariantAggregate
    paired: OfficialJudgePairedSummary
    trials: list[OfficialJudgeTrialScore] = Field(min_length=3)
    observations: list[OfficialJudgeObservation] = Field(min_length=1)


def prepare_official_judge_batch(
    *,
    repeat_experiment_path: Path,
    repeat_comparison_path: Path,
    target_manifest_path: Path,
    official_evaluator_path: Path,
    output_dir: Path,
) -> OfficialJudgeBatchManifest:
    repository_root = _find_repository_root(repeat_experiment_path.resolve())
    _require_under_runs(repeat_experiment_path, repository_root)
    _require_under_runs(repeat_comparison_path, repository_root)
    _require_under_runs(output_dir, repository_root)
    if not repeat_comparison_path.is_file():
        raise ValueError("official judge batch requires a frozen repeat comparison")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("official judge batch output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = aggregate_repeat_experiment(
        manifest_path=repeat_experiment_path,
        target_manifest_path=target_manifest_path,
        output_path=repeat_comparison_path,
        validate_existing=True,
    )
    experiment = RepeatExperimentManifest.model_validate_json(
        repeat_experiment_path.read_text(encoding="utf-8")
    )
    evaluator = load_official_evaluator_manifest(
        official_evaluator_path, target_manifest_path=target_manifest_path
    )
    experiment_hash = normalized_text_file_sha256(repeat_experiment_path)
    comparison_hash = normalized_text_file_sha256(repeat_comparison_path)
    evaluator_hash = normalized_text_file_sha256(official_evaluator_path)
    if comparison.experiment_manifest_sha256 != experiment_hash:
        raise ValueError("repeat comparison is not bound to the experiment manifest")
    if comparison.target_manifest_sha256 != evaluator.target_manifest_sha256:
        raise ValueError("repeat comparison and official evaluator target different benchmarks")

    audit_dir = output_dir / "audit"
    inputs_dir = output_dir / "inputs"
    audit_dir.mkdir()
    inputs_dir.mkdir()
    audit_artifacts = [
        _copy_audit(
            "repeat-experiment",
            repeat_experiment_path,
            audit_dir / "repeat_experiment.json",
            output_dir,
        ),
        _copy_audit(
            "repeat-comparison",
            repeat_comparison_path,
            audit_dir / "repeat_comparison.json",
            output_dir,
        ),
        _copy_audit(
            "official-evaluator",
            official_evaluator_path,
            audit_dir / "official_evaluator.json",
            output_dir,
        ),
        _copy_audit(
            "judge-assets",
            official_evaluator_path.parent / evaluator.judge_assets_manifest,
            audit_dir / "official_judge_assets.json",
            output_dir,
        ),
    ]

    items: list[OfficialJudgeBatchItem] = []
    first_gold_path: Path | None = None
    for pair in experiment.pairs:
        for variant in ("baseline", "candidate"):
            source = getattr(pair, variant)
            if first_gold_path is None:
                first_gold_path = _resolve_repository_path(
                    source.gold_slice_path, repository_root
                )
            export_manifest_path = _resolve_repository_path(
                source.official_export_manifest_path, repository_root
            )
            export_bytes = export_manifest_path.read_bytes()
            export = OfficialRunExportManifest.model_validate_json(export_bytes)
            if export.incomplete or export.completed != comparison.queries_per_trial:
                raise ValueError("official judge source export is incomplete")
            export_role = f"{_safe_id(pair.trial_id)}-{variant}-export"
            audit_artifacts.append(
                _copy_audit(
                    export_role,
                    export_manifest_path,
                    audit_dir / f"{export_role}.json",
                    output_dir,
                )
            )
            for export_item in sorted(export.items, key=lambda row: row.query_id):
                if (
                    export_item.evaluator_status != "completed"
                    or export_item.prediction_sha256 is None
                ):
                    raise ValueError("official judge batch requires completed predictions")
                source_input = (
                    export_manifest_path.parent
                    / "inputs"
                    / f"{_safe_id(export_item.query_id)}.json"
                )
                source_bytes = source_input.read_bytes()
                source_hash = sha256(source_bytes).hexdigest()
                if source_hash != export_item.exported_sha256:
                    raise ValueError("official judge source input hash mismatch")
                _validate_official_input(
                    source_bytes,
                    query_id=export_item.query_id,
                    prediction_sha256=export_item.prediction_sha256,
                )
                batch_id = (
                    f"{_safe_id(pair.trial_id)}-{variant}__"
                    f"{_safe_id(export_item.query_id)}"
                )
                staged_path = inputs_dir / f"{batch_id}.json"
                if staged_path.exists():
                    raise ValueError("official judge batch filename collision")
                _atomic_write_bytes(staged_path, source_bytes)
                items.append(
                    OfficialJudgeBatchItem(
                        batch_id=batch_id,
                        trial_id=pair.trial_id,
                        execution_order=pair.execution_order,
                        variant=variant,
                        query_id=export_item.query_id,
                        source_export_manifest_sha256=sha256(export_bytes).hexdigest(),
                        source_input_sha256=source_hash,
                        staged_input_path=staged_path.relative_to(output_dir).as_posix(),
                        staged_input_sha256=sha256(staged_path.read_bytes()).hexdigest(),
                        prediction_sha256=export_item.prediction_sha256,
                    )
                )

    assert first_gold_path is not None
    ground_truth_dir = output_dir / "ground_truth"
    ground_truth_export = export_official_development_ground_truth(
        gold_slice_path=first_gold_path,
        output_dir=ground_truth_dir,
    )
    expected_query_ids = sorted({item.query_id for item in items})
    if ground_truth_export.query_ids != expected_query_ids:
        raise ValueError("official judge ground truth and batch query IDs differ")
    ground_truth_path = ground_truth_dir / "development_ground_truth.jsonl"
    ground_truth_export_path = ground_truth_dir / "export_manifest.json"

    manifest = OfficialJudgeBatchManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        generator_model=experiment.model,
        repeat_experiment_sha256=experiment_hash,
        repeat_comparison_sha256=comparison_hash,
        official_evaluator_manifest_sha256=evaluator_hash,
        judge_assets_manifest_sha256=evaluator.judge_assets_sha256,
        target_manifest_sha256=evaluator.target_manifest_sha256,
        repository_commit=evaluator.repository_commit,
        evaluator_script=evaluator.evaluator_script,
        evaluator_script_sha256=evaluator.evaluator_script_sha256,
        uv_lock_sha256=evaluator.uv_lock_sha256,
        judge_model=evaluator.judge.name,
        judge_revision=evaluator.judge.revision,
        inference=evaluator.inference,
        registration_status=experiment.registration_status,
        trial_count=comparison.trial_count,
        queries_per_trial=comparison.queries_per_trial,
        input_count=len(items),
        input_set_sha256=_input_set_sha256(items),
        ground_truth_export_manifest_path=ground_truth_export_path.relative_to(
            output_dir
        ).as_posix(),
        ground_truth_export_manifest_sha256=sha256(
            ground_truth_export_path.read_bytes()
        ).hexdigest(),
        ground_truth_path=ground_truth_path.relative_to(output_dir).as_posix(),
        ground_truth_sha256=ground_truth_export.ground_truth_sha256,
        audit_artifacts=audit_artifacts,
        items=sorted(items, key=lambda row: row.batch_id),
    )
    _atomic_write_text(
        output_dir / "batch_manifest.json", manifest.model_dump_json(indent=2)
    )
    validate_official_judge_batch(output_dir / "batch_manifest.json")
    return manifest


def validate_official_judge_batch(path: Path) -> OfficialJudgeBatchManifest:
    batch_root = path.resolve().parent
    manifest = OfficialJudgeBatchManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    for artifact in manifest.audit_artifacts:
        artifact_path = _resolve_batch_path(artifact.path, batch_root)
        if sha256(artifact_path.read_bytes()).hexdigest() != artifact.sha256:
            raise ValueError(f"official judge audit artifact hash mismatch: {artifact.role}")
    for item in manifest.items:
        staged_path = _resolve_batch_path(item.staged_input_path, batch_root)
        staged_bytes = staged_path.read_bytes()
        if sha256(staged_bytes).hexdigest() != item.staged_input_sha256:
            raise ValueError("official judge staged input hash mismatch")
        _validate_official_input(
            staged_bytes,
            query_id=item.query_id,
            prediction_sha256=item.prediction_sha256,
        )
    ground_truth_path = _resolve_batch_path(manifest.ground_truth_path, batch_root)
    if sha256(ground_truth_path.read_bytes()).hexdigest() != manifest.ground_truth_sha256:
        raise ValueError("official judge ground-truth hash mismatch")
    export_path = _resolve_batch_path(
        manifest.ground_truth_export_manifest_path, batch_root
    )
    export_bytes = export_path.read_bytes()
    if sha256(export_bytes).hexdigest() != (
        manifest.ground_truth_export_manifest_sha256
    ):
        raise ValueError("official judge ground-truth export manifest hash mismatch")
    export = OfficialGroundTruthExport.model_validate_json(export_bytes)
    if export.ground_truth_sha256 != manifest.ground_truth_sha256:
        raise ValueError("official judge ground-truth export disagrees with batch")
    if export.query_ids != sorted({item.query_id for item in manifest.items}):
        raise ValueError("official judge ground-truth IDs disagree with batch")
    return manifest


def aggregate_official_judge_results(
    *,
    batch_manifest_path: Path,
    execution_registration_path: Path,
    execution_result_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> OfficialJudgeComparison:
    repository_root = _find_repository_root(batch_manifest_path.resolve())
    _require_under_runs(batch_manifest_path, repository_root)
    _require_under_runs(output_path, repository_root)
    existing: OfficialJudgeComparison | None = None
    if output_path.exists() and not validate_existing:
        raise ValueError("official judge comparison output already exists")
    if output_path.exists():
        existing = OfficialJudgeComparison.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

    batch_bytes = batch_manifest_path.read_bytes()
    batch = validate_official_judge_batch(batch_manifest_path)
    registration_bytes = execution_registration_path.read_bytes()
    registration = OfficialJudgeExecutionRegistration.model_validate_json(
        registration_bytes
    )
    result_bytes = execution_result_path.read_bytes()
    execution = OfficialJudgeExecutionResult.model_validate_json(result_bytes)
    batch_hash = sha256(batch_bytes).hexdigest()
    registration_hash = sha256(registration_bytes).hexdigest()
    if registration.batch_manifest_sha256 != batch_hash:
        raise ValueError("official judge registration targets a different batch")
    if execution.batch_manifest_sha256 != batch_hash:
        raise ValueError("official judge execution targets a different batch")
    if execution.registration_sha256 != registration_hash:
        raise ValueError("official judge execution is not bound to its registration")
    if execution.status != "succeeded" or execution.exit_code != 0:
        raise ValueError("official judge execution did not succeed")
    _validate_registration_against_batch(
        registration, batch, execution_registration_path.resolve().parent
    )

    execution_root = execution_result_path.resolve().parent
    output_files: dict[str, JudgeExecutionFile] = {}
    for file_ref in execution.output_files:
        file_path = _resolve_batch_path(file_ref.path, execution_root)
        if sha256(file_path.read_bytes()).hexdigest() != file_ref.sha256:
            raise ValueError("official judge execution output hash mismatch")
        basename = file_path.name
        if basename in output_files:
            raise ValueError("official judge execution output basenames collide")
        output_files[basename] = file_ref
    _validate_execution_log(execution.stdout, execution_root)
    _validate_execution_log(execution.stderr, execution_root)

    ground_truth = _load_ground_truth(
        _resolve_batch_path(batch.ground_truth_path, batch_manifest_path.parent)
    )
    expected_result_names = {f"{item.batch_id}_eval.json" for item in batch.items}
    actual_result_names = {
        name for name in output_files if name.endswith("_eval.json")
    }
    if actual_result_names != expected_result_names:
        raise ValueError("official judge result file set differs from the frozen batch")
    observations: list[OfficialJudgeObservation] = []
    correctness: dict[tuple[str, str, str], bool] = {}
    for item in batch.items:
        expected_name = f"{item.batch_id}_eval.json"
        file_ref = output_files.get(expected_name)
        if file_ref is None:
            raise ValueError(f"official judge result is missing: {expected_name}")
        result_path = _resolve_batch_path(file_ref.path, execution_root)
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        observation = _parse_official_result(
            result_payload,
            item=item,
            ground_truth=ground_truth,
            result_path=file_ref.path,
            result_sha256=file_ref.sha256,
            max_output_tokens=batch.inference.max_output_tokens,
            judge_model_path=registration.model_dir,
        )
        observations.append(observation)
        correctness[(item.trial_id, item.query_id, item.variant)] = observation.correct

    trials: list[OfficialJudgeTrialScore] = []
    for trial_id in sorted({item.trial_id for item in batch.items}):
        trial_items = [item for item in batch.items if item.trial_id == trial_id]
        query_ids = sorted({item.query_id for item in trial_items})
        order = trial_items[0].execution_order
        baseline_correct = sum(
            correctness[(trial_id, query_id, "baseline")] for query_id in query_ids
        )
        candidate_correct = sum(
            correctness[(trial_id, query_id, "candidate")] for query_id in query_ids
        )
        trials.append(
            OfficialJudgeTrialScore(
                trial_id=trial_id,
                execution_order=order,
                baseline_correct=baseline_correct,
                candidate_correct=candidate_correct,
                query_count=len(query_ids),
                baseline_accuracy_percent=round(
                    baseline_correct / len(query_ids) * 100, 6
                ),
                candidate_accuracy_percent=round(
                    candidate_correct / len(query_ids) * 100, 6
                ),
            )
        )

    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    for trial in trials:
        query_ids = sorted(
            {
                item.query_id
                for item in batch.items
                if item.trial_id == trial.trial_id
            }
        )
        for query_id in query_ids:
            baseline_value = correctness[(trial.trial_id, query_id, "baseline")]
            candidate_value = correctness[(trial.trial_id, query_id, "candidate")]
            if candidate_value and not baseline_value:
                candidate_wins += 1
            elif baseline_value and not candidate_value:
                baseline_wins += 1
            else:
                ties += 1

    artifact = OfficialJudgeComparison(
        created_at=(
            existing.created_at
            if existing is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        batch_manifest_sha256=batch_hash,
        execution_registration_sha256=registration_hash,
        execution_result_sha256=sha256(result_bytes).hexdigest(),
        judge_model=batch.judge_model,
        judge_revision=batch.judge_revision,
        inference=batch.inference,
        trial_count=batch.trial_count,
        queries_per_trial=batch.queries_per_trial,
        evaluations=len(observations),
        baseline=_variant_aggregate("baseline", observations, trials),
        candidate=_variant_aggregate("candidate", observations, trials),
        paired=OfficialJudgePairedSummary(
            comparisons=batch.trial_count * batch.queries_per_trial,
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            ties=ties,
        ),
        trials=trials,
        observations=sorted(
            observations, key=lambda row: (row.trial_id, row.variant, row.query_id)
        ),
    )
    if existing is not None:
        if artifact != existing:
            raise ValueError("existing official judge comparison no longer matches sources")
        return existing
    _atomic_write_text(output_path, artifact.model_dump_json(indent=2))
    return artifact


def _validate_registration_against_batch(
    registration: OfficialJudgeExecutionRegistration,
    batch: OfficialJudgeBatchManifest,
    registration_root: Path,
) -> None:
    expected = {
        "official_evaluator_manifest_sha256": batch.official_evaluator_manifest_sha256,
        "judge_assets_manifest_sha256": batch.judge_assets_manifest_sha256,
        "repository_commit": batch.repository_commit,
        "evaluator_script_sha256": batch.evaluator_script_sha256,
        "uv_lock_sha256": batch.uv_lock_sha256,
        "judge_model": batch.judge_model,
        "judge_revision": batch.judge_revision,
        "inference": batch.inference,
        "batch_size": batch.input_count,
    }
    for field_name, expected_value in expected.items():
        if getattr(registration, field_name) != expected_value:
            raise ValueError(f"official judge registration changes {field_name}")
    _validate_registered_command(registration, batch)
    verification_path = _resolve_batch_path(
        registration.asset_verification_path, registration_root
    )
    verification_bytes = verification_path.read_bytes()
    if sha256(verification_bytes).hexdigest() != registration.asset_verification_sha256:
        raise ValueError("official judge asset verification hash mismatch")
    verification = json.loads(verification_bytes)
    if (
        verification.get("passed") is not True
        or verification.get("matched") != verification.get("file_count")
        or verification.get("model") != batch.judge_model
        or verification.get("revision") != batch.judge_revision
        or verification.get("manifest_sha256") != batch.judge_assets_manifest_sha256
    ):
        raise ValueError("official judge asset verification does not match the batch")
    launcher_path = _resolve_batch_path(registration.launcher_path, registration_root)
    if sha256(launcher_path.read_bytes()).hexdigest() != registration.launcher_sha256:
        raise ValueError("official judge launcher hash mismatch")


def _validate_registered_command(
    registration: OfficialJudgeExecutionRegistration,
    batch: OfficialJudgeBatchManifest,
) -> None:
    command = registration.command
    if len(command) != 24 or not command[1].replace("\\", "/").endswith(
        f"/{batch.evaluator_script}"
    ):
        raise ValueError("official judge command does not invoke the pinned evaluator")
    flags = command[2::2]
    values = command[3::2]
    if len(flags) != len(set(flags)):
        raise ValueError("official judge command repeats an argument")
    arguments = dict(zip(flags, values, strict=True))
    expected = {
        "--temperature": str(batch.inference.temperature),
        "--top_p": str(batch.inference.top_p),
        "--top_k": str(batch.inference.top_k),
        "--max_output_tokens": str(batch.inference.max_output_tokens),
        "--batch_size": str(batch.input_count),
        "--tensor_parallel_size": "2",
        "--model": registration.model_dir,
    }
    expected_flags = {
        "--input_dir",
        "--ground_truth",
        "--eval_dir",
        "--model",
        "--temperature",
        "--top_p",
        "--top_k",
        "--max_output_tokens",
        "--batch_size",
        "--tensor_parallel_size",
        "--qrel_evidence",
    }
    if set(arguments) != expected_flags or any(
        arguments.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("official judge command changes the inference contract")
    normalized = {name: value.replace("\\", "/") for name, value in arguments.items()}
    if (
        not normalized["--input_dir"].endswith("/inputs")
        or not normalized["--ground_truth"].endswith(
            "/ground_truth/development_ground_truth.jsonl"
        )
        or not normalized["--qrel_evidence"].endswith(
            "/topics-qrels/qrel_evidence.txt"
        )
    ):
        raise ValueError("official judge command changes a frozen data path")


def _parse_official_result(
    payload: dict,
    *,
    item: OfficialJudgeBatchItem,
    ground_truth: dict[str, dict[str, str]],
    result_path: str,
    result_sha256: str,
    max_output_tokens: int,
    judge_model_path: str,
) -> OfficialJudgeObservation:
    reference = ground_truth.get(item.query_id)
    if reference is None:
        raise ValueError("official judge result query lacks ground truth")
    response = payload.get("response")
    judge_result = payload.get("judge_result")
    model_info = payload.get("model_info")
    if (
        str(payload.get("query_id")) != item.query_id
        or payload.get("is_completed") is not True
        or not isinstance(response, str)
        or sha256(response.encode("utf-8")).hexdigest() != item.prediction_sha256
        or payload.get("question") != reference["question"]
        or payload.get("correct_answer") != reference["answer"]
        or not isinstance(judge_result, dict)
        or judge_result.get("parse_error") is not False
        or not isinstance(judge_result.get("correct"), bool)
        or not isinstance(payload.get("judge_response"), str)
        or not payload.get("judge_response")
        or not isinstance(model_info, dict)
        or model_info.get("judge_model") != judge_model_path
        or model_info.get("max_output_tokens") != max_output_tokens
    ):
        raise ValueError("official judge result does not satisfy the frozen contract")
    confidence = judge_result.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
    ):
        raise ValueError("official judge confidence must be numeric when present")
    return OfficialJudgeObservation(
        trial_id=item.trial_id,
        execution_order=item.execution_order,
        variant=item.variant,
        query_id=item.query_id,
        correct=judge_result["correct"],
        confidence=confidence,
        prediction_sha256=item.prediction_sha256,
        result_path=result_path,
        result_sha256=result_sha256,
    )


def _variant_aggregate(
    variant: Literal["baseline", "candidate"],
    observations: list[OfficialJudgeObservation],
    trials: list[OfficialJudgeTrialScore],
) -> OfficialJudgeVariantAggregate:
    selected = [row for row in observations if row.variant == variant]
    correct = sum(row.correct for row in selected)
    values = [
        (
            trial.baseline_accuracy_percent
            if variant == "baseline"
            else trial.candidate_accuracy_percent
        )
        for trial in trials
    ]
    return OfficialJudgeVariantAggregate(
        variant=variant,
        correct=correct,
        evaluations=len(selected),
        pooled_accuracy_percent=round(correct / len(selected) * 100, 6),
        trial_accuracy_percent=MetricDistribution(
            trials=len(values),
            mean=round(mean(values), 6),
            sample_stddev=round(stdev(values), 6),
            minimum=round(min(values), 6),
            maximum=round(max(values), 6),
        ),
    )


def _validate_official_input(
    raw: bytes, *, query_id: str, prediction_sha256: str
) -> None:
    payload = json.loads(raw)
    result = payload.get("result")
    terminal = result[-1] if isinstance(result, list) and result else None
    response = terminal.get("output") if isinstance(terminal, dict) else None
    if (
        str(payload.get("query_id")) != query_id
        or payload.get("status") != "completed"
        or not isinstance(terminal, dict)
        or terminal.get("type") != "output_text"
        or not isinstance(response, str)
        or sha256(response.encode("utf-8")).hexdigest() != prediction_sha256
    ):
        raise ValueError("official judge input does not match its frozen prediction")


def _load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        query_id = str(payload["query_id"])
        if query_id in rows:
            raise ValueError("official judge ground truth contains duplicate IDs")
        rows[query_id] = {
            "question": payload["query"],
            "answer": payload["answer"],
        }
    return rows


def _input_set_sha256(items: list[OfficialJudgeBatchItem]) -> str:
    canonical = "\n".join(
        f"{item.batch_id}\t{item.staged_input_sha256}\t{item.prediction_sha256}"
        for item in sorted(items, key=lambda row: row.batch_id)
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _copy_audit(
    role: str, source: Path, destination: Path, batch_root: Path
) -> JudgeBatchArtifact:
    raw = source.read_bytes()
    _atomic_write_bytes(destination, raw)
    return JudgeBatchArtifact(
        role=role,
        path=destination.relative_to(batch_root).as_posix(),
        sha256=sha256(raw).hexdigest(),
    )


def _validate_execution_log(file_ref: JudgeExecutionFile, root: Path) -> None:
    path = _resolve_batch_path(file_ref.path, root)
    if sha256(path.read_bytes()).hexdigest() != file_ref.sha256:
        raise ValueError("official judge execution log hash mismatch")


def _resolve_repository_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("official judge source path must be repository-relative")
    resolved = (repository_root / path).resolve()
    _require_under_runs(resolved, repository_root)
    if not resolved.is_file():
        raise ValueError(f"official judge source artifact is missing: {value}")
    return resolved


def _resolve_batch_path(value: str, root: Path) -> Path:
    _validate_relative_path(value)
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"official judge artifact is missing or unsafe: {value}")
    return resolved


def _validate_relative_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path in {Path("."), Path("")}:
        raise ValueError("official judge artifact paths must stay relative")


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not safe or safe in {".", ".."}:
        raise ValueError("official judge ID cannot be represented safely")
    return safe


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate repository root")


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("official judge runtime artifacts must remain under ignored runs/")


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))
