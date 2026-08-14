from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_decision import (
    CandidateFailureProfile,
    DecisionSource,
    FailureAggregate,
    analyze_candidate_failures,
)
from .browsecomp_plus import normalized_text_file_sha256
from .browsecomp_repeats import (
    RepeatExperimentManifest,
    aggregate_repeat_experiment,
)
from .pi_browsecomp import PiSmokeSummary
from .repeat_development_judge import RepeatDevelopmentJudgeComparison


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DenseConfirmationGate(StrictContract):
    gate_id: Literal[
        "registration_status",
        "trial_count",
        "query_count",
        "candidate_schema_complete",
        "output_budget_overshoot",
        "evidence_recall_delta",
        "evidence_query_wins_minus_losses",
        "judge_accuracy_delta",
        "judge_parse_failures",
        "judge_request_failures",
        "candidate_to_baseline_cost_ratio",
        "combined_provider_cost",
        "final_provider_failures",
    ]
    observed: str | float | int
    operator: Literal["eq", "ge", "le"]
    threshold: str | float | int
    passed: bool


class DenseConfirmationDecision(StrictContract):
    schema_version: Literal["browsecomp-plus-dense-confirmation-decision-v1"] = (
        "browsecomp-plus-dense-confirmation-decision-v1"
    )
    created_at: str
    status: Literal["development_decision_not_leaderboard"] = (
        "development_decision_not_leaderboard"
    )
    judge_metric_status: Literal[
        "calibrated_development_diagnostic_not_official"
    ] = "calibrated_development_diagnostic_not_official"
    decision: Literal["promote", "reject"]
    candidate_layer: Literal["dense_retrieval_v0"] = "dense_retrieval_v0"
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    baseline_retriever_id: str = Field(min_length=1)
    candidate_retriever_id: str = Field(min_length=1)
    trial_count: int = Field(ge=3)
    queries_per_trial: int = Field(gt=0)
    evidence_recall_delta_pp: float
    judge_accuracy_delta_pp: float
    candidate_to_baseline_cost_ratio: float = Field(ge=0)
    combined_provider_cost_usd: float = Field(ge=0)
    gates: list[DenseConfirmationGate] = Field(min_length=13, max_length=13)
    failure_aggregate: FailureAggregate
    failure_profiles: list[CandidateFailureProfile] = Field(min_length=1)
    next_action: Literal[
        "audit_incomplete_or_structurally_invalid_grid",
        "diagnose_retrieval_policy_from_saved_replays",
        "diagnose_evidence_to_answer_control",
        "promote_dense_and_cluster_remaining_failures",
    ]
    sources: dict[str, DecisionSource]
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_registered_gates(self) -> "DenseConfirmationDecision":
        expected_ids = {
            "registration_status",
            "trial_count",
            "query_count",
            "candidate_schema_complete",
            "output_budget_overshoot",
            "evidence_recall_delta",
            "evidence_query_wins_minus_losses",
            "judge_accuracy_delta",
            "judge_parse_failures",
            "judge_request_failures",
            "candidate_to_baseline_cost_ratio",
            "combined_provider_cost",
            "final_provider_failures",
        }
        gate_ids = {gate.gate_id for gate in self.gates}
        if gate_ids != expected_ids or len(gate_ids) != len(self.gates):
            raise ValueError("dense confirmation gate set is incomplete")
        expected_decision = "promote" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected_decision:
            raise ValueError("dense confirmation decision differs from its gates")
        if self.failure_aggregate.query_count != len(self.failure_profiles):
            raise ValueError("dense confirmation failure profile count differs")
        expected_sources = {
            "preregistration",
            "repeat_experiment",
            "repeat_comparison",
            "judge_manifest",
            "judge_calibration",
            "judge_execution_registration",
            "judge_execution_result",
            "judge_comparison",
        }
        if set(self.sources) != expected_sources:
            raise ValueError("dense confirmation decision source set is incomplete")
        return self


def decide_dense_confirmation(
    *,
    preregistration_path: Path,
    repeat_experiment_path: Path,
    repeat_comparison_path: Path,
    target_manifest_path: Path,
    judge_manifest_path: Path,
    judge_calibration_path: Path,
    judge_execution_registration_path: Path,
    judge_execution_result_path: Path,
    judge_comparison_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> DenseConfirmationDecision:
    repository_root = _find_repository_root(repeat_experiment_path.resolve())
    for path in (
        repeat_experiment_path,
        repeat_comparison_path,
        judge_calibration_path,
        judge_execution_registration_path,
        judge_execution_result_path,
        judge_comparison_path,
        output_path,
    ):
        _require_under_runs(path, repository_root)
    existing = None
    if output_path.exists() and not validate_existing:
        raise ValueError("dense confirmation decision already exists")
    if output_path.exists():
        existing = DenseConfirmationDecision.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

    preregistration = _load_object(preregistration_path)
    if (
        preregistration.get("schema_version")
        != "browsecomp-plus-dense-confirmation-v1"
        or preregistration.get("status") != "preregistered_not_run"
    ):
        raise ValueError("dense confirmation preregistration is not frozen")
    repeat_bytes = repeat_experiment_path.read_bytes()
    repeat = aggregate_repeat_experiment(
        manifest_path=repeat_experiment_path,
        target_manifest_path=target_manifest_path,
        output_path=repeat_comparison_path,
        validate_existing=True,
    )
    experiment = RepeatExperimentManifest.model_validate_json(repeat_bytes)
    judge_bytes = judge_comparison_path.read_bytes()
    judge = RepeatDevelopmentJudgeComparison.model_validate_json(judge_bytes)
    registration = _load_object(judge_execution_registration_path)
    execution = _load_object(judge_execution_result_path)
    calibration_bytes = judge_calibration_path.read_bytes()

    fixed = _require_object(preregistration, "fixed_contract")
    execution_contract = _require_object(preregistration, "execution")
    gates_contract = _require_object(preregistration, "acceptance_gates")
    fresh_queries = _require_object(preregistration, "fresh_query_selection")
    registered_repeat_hash = (
        experiment.source_manifest_sha256
        if experiment.schema_version == "browsecomp-plus-repeat-experiment-v2"
        else sha256(repeat_bytes).hexdigest()
    )
    if execution_contract.get("repeat_manifest_sha256") != registered_repeat_hash:
        raise ValueError("preregistration targets another repeat manifest")
    if fixed.get("model") != repeat.model or fixed.get("control_policy") != (
        repeat.control_policy
    ):
        raise ValueError("preregistration changes frozen model controls")
    if (
        fixed.get("baseline_retriever_id") != repeat.baseline.retriever_id
        or fixed.get("candidate_retriever_id") != repeat.candidate.retriever_id
        or fixed.get("candidate_retriever_manifest_sha256")
        != repeat.candidate.retriever_manifest_sha256
    ):
        raise ValueError("preregistration changes retriever contract")
    if fresh_queries.get("limit") != repeat.queries_per_trial:
        raise ValueError("preregistration changes frozen query count")
    if fixed.get("judge_calibration_sha256") != sha256(
        calibration_bytes
    ).hexdigest():
        raise ValueError("preregistration changes Judge calibration")
    if fixed.get("judge_manifest") != judge_manifest_path.relative_to(
        repository_root
    ).as_posix():
        raise ValueError("preregistration changes Judge manifest path")
    if judge.judge_manifest_sha256 != normalized_text_file_sha256(
        judge_manifest_path
    ):
        raise ValueError("Judge comparison changes Judge manifest")
    if judge.calibration_sha256 != sha256(calibration_bytes).hexdigest():
        raise ValueError("Judge comparison changes calibration")
    if judge.repeat_experiment_sha256 != sha256(repeat_bytes).hexdigest():
        raise ValueError("Judge comparison targets another repeat experiment")
    if judge.repeat_comparison_sha256 != sha256(
        repeat_comparison_path.read_bytes()
    ).hexdigest():
        raise ValueError("Judge comparison targets another repeat comparison")
    _validate_execution_wrapper(
        registration=registration,
        registration_path=judge_execution_registration_path,
        execution=execution,
        judge_comparison_path=judge_comparison_path,
    )

    evidence_delta = round(
        repeat.candidate.evidence_recall_percent.mean
        - repeat.baseline.evidence_recall_percent.mean,
        6,
    )
    judge_delta = round(
        judge.candidate.trial_accuracy_percent.mean
        - judge.baseline.trial_accuracy_percent.mean,
        6,
    )
    evidence_paired = next(
        row for row in repeat.paired_metrics if row.metric == "evidence_recall"
    )
    evidence_wins_minus_losses = (
        evidence_paired.candidate_wins - evidence_paired.baseline_wins
    )
    combined_cost = round(
        repeat.baseline.total_cost_usd + repeat.candidate.total_cost_usd, 12
    )
    if repeat.baseline.total_cost_usd <= 0:
        raise ValueError("baseline provider cost must be positive")
    cost_ratio = round(
        repeat.candidate.total_cost_usd / repeat.baseline.total_cost_usd, 12
    )
    final_failures, overshoot = _summary_failure_totals(
        experiment=experiment, repository_root=repository_root
    )

    gate_values = [
        _gate(
            "registration_status",
            repeat.registration_status,
            "eq",
            "pre_generation",
        ),
        _gate("trial_count", repeat.trial_count, "ge", int(execution_contract["trials"])),
        _gate("query_count", repeat.queries_per_trial, "ge", int(fresh_queries["limit"])),
        _gate(
            "candidate_schema_complete",
            repeat.candidate.schema_complete_percent.mean,
            "ge",
            float(gates_contract["minimum_candidate_schema_complete_percent"]),
        ),
        _gate(
            "output_budget_overshoot",
            overshoot,
            "le",
            int(gates_contract["maximum_output_budget_overshoot_tokens"]),
        ),
        _gate(
            "evidence_recall_delta",
            evidence_delta,
            "ge",
            float(gates_contract["minimum_evidence_recall_delta_pp"]),
        ),
        _gate(
            "evidence_query_wins_minus_losses",
            evidence_wins_minus_losses,
            "ge",
            int(gates_contract["minimum_query_level_evidence_recall_wins_minus_losses"]),
        ),
        _gate(
            "judge_accuracy_delta",
            judge_delta,
            "ge",
            float(gates_contract["minimum_calibrated_judge_accuracy_delta_pp"]),
        ),
        _gate(
            "judge_parse_failures",
            judge.parse_failures,
            "le",
            int(gates_contract["maximum_judge_parse_failures"]),
        ),
        _gate(
            "judge_request_failures",
            judge.request_failures,
            "le",
            int(gates_contract["maximum_judge_request_failures"]),
        ),
        _gate(
            "candidate_to_baseline_cost_ratio",
            cost_ratio,
            "le",
            float(gates_contract["maximum_candidate_to_baseline_provider_cost_ratio"]),
        ),
        _gate(
            "combined_provider_cost",
            combined_cost,
            "le",
            float(execution_contract["maximum_combined_provider_cost_usd"]),
        ),
        _gate(
            "final_provider_failures",
            final_failures,
            "le",
            int(execution_contract["maximum_provider_failures_after_allowed_resume"]),
        ),
    ]
    all_passed = all(row.passed for row in gate_values)
    structural_ids = {
        "registration_status",
        "trial_count",
        "query_count",
        "candidate_schema_complete",
        "output_budget_overshoot",
        "judge_parse_failures",
        "judge_request_failures",
        "combined_provider_cost",
        "final_provider_failures",
    }
    gate_by_id = {row.gate_id: row for row in gate_values}
    if not gate_by_id["evidence_recall_delta"].passed or not gate_by_id[
        "evidence_query_wins_minus_losses"
    ].passed:
        next_action = "diagnose_retrieval_policy_from_saved_replays"
    elif not gate_by_id["judge_accuracy_delta"].passed:
        next_action = "diagnose_evidence_to_answer_control"
    elif not all(gate_by_id[gate_id].passed for gate_id in structural_ids):
        next_action = "audit_incomplete_or_structurally_invalid_grid"
    else:
        next_action = "promote_dense_and_cluster_remaining_failures"

    profiles, failures = analyze_candidate_failures(
        repeat=repeat,
        judge=judge,
        repository_root=repository_root,
    )
    artifact = DenseConfirmationDecision(
        created_at=(
            existing.created_at
            if existing is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        decision="promote" if all_passed else "reject",
        model=repeat.model,
        baseline_retriever_id=repeat.baseline.retriever_id,
        candidate_retriever_id=repeat.candidate.retriever_id,
        trial_count=repeat.trial_count,
        queries_per_trial=repeat.queries_per_trial,
        evidence_recall_delta_pp=evidence_delta,
        judge_accuracy_delta_pp=judge_delta,
        candidate_to_baseline_cost_ratio=cost_ratio,
        combined_provider_cost_usd=combined_cost,
        gates=gate_values,
        failure_aggregate=failures,
        failure_profiles=profiles,
        next_action=next_action,
        sources={
            "preregistration": _source(preregistration_path, repository_root),
            "repeat_experiment": _source(repeat_experiment_path, repository_root),
            "repeat_comparison": _source(repeat_comparison_path, repository_root),
            "judge_manifest": _source(judge_manifest_path, repository_root),
            "judge_calibration": _source(judge_calibration_path, repository_root),
            "judge_execution_registration": _source(
                judge_execution_registration_path, repository_root
            ),
            "judge_execution_result": _source(
                judge_execution_result_path, repository_root
            ),
            "judge_comparison": _source(judge_comparison_path, repository_root),
        },
        limitations=(
            "Fresh development slice only; sealed holdout was not accessed and no "
            "leaderboard submission was made.",
            "Three trials repeat one frozen 25-query set; they measure runtime "
            "stability, not 75 independent questions.",
            "Judge labels come from a calibrated persistent service matching the "
            "frozen grader/model contract, not the official evaluator process.",
            "Observed differences are harness effects with frozen model parameters, "
            "not model-capability improvements.",
        ),
    )
    if existing is not None:
        if artifact != existing:
            raise ValueError("existing dense confirmation decision changed with sources")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return artifact


def _gate(
    gate_id: str,
    observed: str | float | int,
    operator: Literal["eq", "ge", "le"],
    threshold: str | float | int,
) -> DenseConfirmationGate:
    passed = (
        observed == threshold
        if operator == "eq"
        else observed >= threshold
        if operator == "ge"
        else observed <= threshold
    )
    return DenseConfirmationGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _summary_failure_totals(
    *, experiment: RepeatExperimentManifest, repository_root: Path
) -> tuple[int, int]:
    failures = 0
    overshoot = 0
    for pair in experiment.pairs:
        for variant in ("baseline", "candidate"):
            summary_path = _resolve_repository_file(
                getattr(pair, variant).summary_path, repository_root
            )
            summary = PiSmokeSummary.model_validate_json(
                summary_path.read_text(encoding="utf-8")
            )
            failures += summary.failed
            overshoot += summary.total_output_budget_overshoot_tokens
    return failures, overshoot


def _validate_execution_wrapper(
    *,
    registration: dict[str, object],
    registration_path: Path,
    execution: dict[str, object],
    judge_comparison_path: Path,
) -> None:
    if (
        registration.get("status")
        not in {
            "registered_pre_inference",
            "registered_pre_aggregation_recovery",
        }
        or registration.get("metric_status")
        != "calibrated_development_diagnostic_not_official"
        or execution.get("status") != "succeeded"
        or execution.get("error") is not None
        or execution.get("registration_sha256")
        != sha256(registration_path.read_bytes()).hexdigest()
    ):
        raise ValueError("repeat Judge execution wrapper did not succeed")
    if registration.get("status") == "registered_pre_aggregation_recovery" and (
        registration.get("provider_calls") != 0
        or execution.get("provider_calls") != 0
    ):
        raise ValueError("repeat Judge aggregation recovery made provider calls")
    comparison = execution.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("sha256") != sha256(
        judge_comparison_path.read_bytes()
    ).hexdigest():
        raise ValueError("repeat Judge execution wrapper changes comparison")


def _require_object(
    payload: dict[str, object], field_name: str
) -> dict[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"dense confirmation preregistration lacks {field_name}")
    return value


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dense confirmation source must be a JSON object")
    return value


def _resolve_repository_file(value: str, repository_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("dense confirmation source path must be repository-relative")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("dense confirmation runtime source must stay under runs/")
    if not resolved.is_file():
        raise ValueError(f"dense confirmation source is missing: {value}")
    return resolved


def _source(path: Path, repository_root: Path) -> DecisionSource:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return DecisionSource(
        path=display,
        sha256=sha256(resolved.read_bytes()).hexdigest(),
    )


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "runs"
        ).is_dir():
            return candidate
    raise ValueError("could not locate repository root for dense confirmation")


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("dense confirmation artifacts must stay under runs/")
