from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DiagnosticSummary
from .browsecomp_plus import PiBrowseCompRun, load_pi_browsecomp_run
from .development_judge import DevelopmentJudgeResult
from .evidence_span_oracle import ArtifactReference
from .obligation_span_fresh import (
    FreshVariant,
    ObligationSpanFreshRegistration,
    load_obligation_span_fresh_registration,
)
from .pi_browsecomp import PiSmokeSummary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepeatArmPaths(StrictContract):
    run_root: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    diagnostic: str = Field(min_length=1)
    judge: str = Field(min_length=1)


class ObligationSpanRepeatTrial(StrictContract):
    trial_id: str = Field(pattern=r"^trial-[1-3]$")
    execution_order: Literal["baseline_first", "candidate_first"]
    observed_before_registration: bool
    baseline: RepeatArmPaths
    candidate: RepeatArmPaths


class ObligationSpanRepeatAcceptance(StrictContract):
    trial_count_must_equal: Literal[3] = 3
    total_generation_failures_must_equal: Literal[0] = 0
    total_judge_failures_must_equal: Literal[0] = 0
    maximum_provider_attempt_policy_violations: Literal[0] = 0
    minimum_candidate_minus_baseline_schema_complete: int = Field(ge=0)
    minimum_aggregate_judge_correct_delta: int = Field(ge=1)
    minimum_noninferior_judge_trials: int = Field(ge=2, le=3)
    minimum_aggregate_normalized_exact_delta: int = Field(ge=0)
    minimum_mean_evidence_recall_delta_pp: float
    maximum_total_token_ratio: float = Field(ge=1)
    maximum_recorded_provider_cost_ratio: float = Field(ge=1)
    maximum_new_recorded_provider_cost_usd: float = Field(gt=0)


class ObligationSpanRepeatRecovery(StrictContract):
    reason: Literal["post_provider_trace_validation_failure"]
    original_repeat_registration: ArtifactReference
    failed_summary: ArtifactReference
    failed_query_ids: tuple[str, ...] = Field(min_length=1)
    provider_attempt_policy_violations: Literal[1] = 1
    effectiveness_gate_already_failed: Literal[True] = True


class ObligationSpanRepeatRegistration(StrictContract):
    schema_version: Literal["obligation-span-repeat-registration-v0"] = (
        "obligation-span-repeat-registration-v0"
    )
    status: Literal["preregistered_before_trials_2_and_3"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    reference_fresh_registration: ArtifactReference
    first_trial_decision: ArtifactReference
    first_trial_bad_case_audit: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    variants: tuple[FreshVariant, FreshVariant]
    trials: tuple[
        ObligationSpanRepeatTrial,
        ObligationSpanRepeatTrial,
        ObligationSpanRepeatTrial,
    ]
    provider_attempts_per_new_case: int = Field(default=1, ge=1, le=2)
    known_unobservable_provider_attempts: int = Field(default=1, ge=1, le=2)
    maximum_new_provider_cases: Literal[100] = 100
    acceptance: ObligationSpanRepeatAcceptance
    recovery: ObligationSpanRepeatRecovery | None = None
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def repeat_grid_is_fixed(self) -> "ObligationSpanRepeatRegistration":
        if [variant.name for variant in self.variants] != ["baseline", "candidate"]:
            raise ValueError("repeat variants must be baseline then candidate")
        if [trial.trial_id for trial in self.trials] != [
            "trial-1",
            "trial-2",
            "trial-3",
        ]:
            raise ValueError("repeat trials must be ordered trial-1 through trial-3")
        if [trial.execution_order for trial in self.trials] != [
            "baseline_first",
            "candidate_first",
            "baseline_first",
        ]:
            raise ValueError("repeat execution order must alternate")
        if [trial.observed_before_registration for trial in self.trials] != [
            True,
            False,
            False,
        ]:
            raise ValueError("only trial-1 may predate repeat registration")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("repeat query IDs must be unique")
        all_paths = [
            value
            for trial in self.trials
            for arm in (trial.baseline, trial.candidate)
            for value in (arm.run_root, arm.summary, arm.diagnostic, arm.judge)
        ]
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("repeat trial paths must be unique")
        if self.recovery is None:
            if self.provider_attempts_per_new_case != 1:
                raise ValueError("initial repeats permit one provider attempt per case")
            if self.known_unobservable_provider_attempts != 1:
                raise ValueError("initial repeats inherit one unobservable attempt")
        else:
            if self.provider_attempts_per_new_case != 2:
                raise ValueError("repeat recovery must cap provider attempts at two")
            if self.known_unobservable_provider_attempts != 2:
                raise ValueError("repeat recovery must count both failed attempts")
            if not set(self.recovery.failed_query_ids).issubset(self.query_ids):
                raise ValueError("repeat recovery query IDs escape the frozen slice")
        return self


class RepeatArmMetrics(StrictContract):
    schema_complete: int = Field(ge=0)
    judge_correct: int = Field(ge=0)
    normalized_exact: int = Field(ge=0)
    evidence_recall_percent: float = Field(ge=0, le=100)
    search_calls: int = Field(ge=0)
    successful_open_calls: int = Field(ge=0)
    total_tokens: int = Field(gt=0)
    recorded_cost_usd: float = Field(gt=0)


class ObligationSpanRepeatTrialResult(StrictContract):
    trial_id: str
    execution_order: Literal["baseline_first", "candidate_first"]
    observed_before_registration: bool
    baseline: RepeatArmMetrics
    candidate: RepeatArmMetrics
    judge_correct_delta: int
    normalized_exact_delta: int
    evidence_recall_delta_pp: float
    paired_improvements: int = Field(ge=0)
    paired_regressions: int = Field(ge=0)
    sources: dict[str, ArtifactReference]


class RepeatGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class ObligationSpanRepeatDecision(StrictContract):
    schema_version: Literal["obligation-span-repeat-decision-v0"] = (
        "obligation-span-repeat-decision-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["development_repeat_decision_not_official"] = (
        "development_repeat_decision_not_official"
    )
    decision: Literal["accept", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_count: Literal[3] = 3
    query_count_per_trial: int = Field(gt=0)
    baseline_judge_correct: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    judge_correct_delta: int
    noninferior_judge_trials: int = Field(ge=0, le=3)
    baseline_schema_complete: int = Field(ge=0)
    candidate_schema_complete: int = Field(ge=0)
    schema_complete_delta: int
    baseline_normalized_exact: int = Field(ge=0)
    candidate_normalized_exact: int = Field(ge=0)
    normalized_exact_delta: int
    baseline_mean_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_mean_evidence_recall_percent: float = Field(ge=0, le=100)
    mean_evidence_recall_delta_pp: float
    paired_improvements: int = Field(ge=0)
    paired_regressions: int = Field(ge=0)
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0)
    candidate_successful_open_calls: int = Field(ge=0)
    baseline_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(gt=0)
    total_token_ratio: float = Field(ge=0)
    baseline_recorded_cost_usd: float = Field(gt=0)
    candidate_recorded_cost_usd: float = Field(gt=0)
    recorded_provider_cost_ratio: float = Field(ge=0)
    new_recorded_provider_cost_usd: float = Field(gt=0)
    known_unobservable_provider_attempts: int = Field(ge=1)
    provider_attempt_policy_violations: int = Field(ge=0)
    cost_observability: Literal[
        "recorded_lower_bound_with_unobservable_failed_attempts"
    ] = "recorded_lower_bound_with_unobservable_failed_attempts"
    trials: tuple[
        ObligationSpanRepeatTrialResult,
        ObligationSpanRepeatTrialResult,
        ObligationSpanRepeatTrialResult,
    ]
    gates: tuple[RepeatGate, ...] = Field(min_length=1)
    next_action: Literal[
        "promote_span_opening_to_official_evaluator_gate",
        "freeze_span_opening_and_pivot_layer",
    ]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "ObligationSpanRepeatDecision":
        expected = "accept" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected:
            raise ValueError("repeat decision differs from gates")
        expected_next = (
            "promote_span_opening_to_official_evaluator_gate"
            if self.decision == "accept"
            else "freeze_span_opening_and_pivot_layer"
        )
        if self.next_action != expected_next:
            raise ValueError("repeat next action differs from decision")
        return self


def load_obligation_span_repeat_registration(
    path: Path,
) -> ObligationSpanRepeatRegistration:
    registration = ObligationSpanRepeatRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    for artifact in (
        registration.reference_fresh_registration,
        registration.first_trial_decision,
        registration.first_trial_bad_case_audit,
        *(variant.adapter_runner for variant in registration.variants),
        *(
            (
                registration.recovery.original_repeat_registration,
                registration.recovery.failed_summary,
            )
            if registration.recovery is not None
            else ()
        ),
        *registration.frozen_artifacts,
    ):
        artifact_path = (root / artifact.path).resolve()
        if not artifact_path.is_relative_to(root) or not artifact_path.is_file():
            raise ValueError(f"repeat artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"repeat artifact hash changed: {artifact.path}")
    reference = load_obligation_span_fresh_registration(
        root / registration.reference_fresh_registration.path
    )
    if registration.query_ids != reference.query_ids:
        raise ValueError("repeat query IDs differ from fresh registration")
    if registration.variants != reference.variants:
        raise ValueError("repeat variants differ from fresh registration")
    if registration.recovery is not None:
        failed_summary = PiSmokeSummary.model_validate_json(
            (root / registration.recovery.failed_summary.path).read_text(
                encoding="utf-8"
            )
        )
        failed_ids = tuple(
            item.query_id
            for item in failed_summary.items
            if item.status != "succeeded"
        )
        if failed_ids != registration.recovery.failed_query_ids:
            raise ValueError("repeat recovery IDs differ from the frozen failure")
    runs_root = (root / "runs").resolve()
    for trial in registration.trials:
        for arm in (trial.baseline, trial.candidate):
            for value in (arm.run_root, arm.summary, arm.diagnostic, arm.judge):
                resolved = (root / value).resolve()
                if not resolved.is_relative_to(runs_root):
                    raise ValueError("repeat output path escapes ignored runs/")
    return registration


def decide_obligation_span_repeats(
    *, registration_path: Path, output_path: Path
) -> ObligationSpanRepeatDecision:
    registration = load_obligation_span_repeat_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("repeat decision output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("repeat decision output already exists")
    reference = ObligationSpanFreshRegistration.model_validate_json(
        (root / registration.reference_fresh_registration.path).read_text(
            encoding="utf-8"
        )
    )
    results = tuple(
        _load_trial(root, registration, reference, trial)
        for trial in registration.trials
    )
    baseline_judge = sum(row.baseline.judge_correct for row in results)
    candidate_judge = sum(row.candidate.judge_correct for row in results)
    baseline_schema = sum(row.baseline.schema_complete for row in results)
    candidate_schema = sum(row.candidate.schema_complete for row in results)
    baseline_exact = sum(row.baseline.normalized_exact for row in results)
    candidate_exact = sum(row.candidate.normalized_exact for row in results)
    baseline_recall = sum(row.baseline.evidence_recall_percent for row in results) / 3
    candidate_recall = sum(row.candidate.evidence_recall_percent for row in results) / 3
    baseline_tokens = sum(row.baseline.total_tokens for row in results)
    candidate_tokens = sum(row.candidate.total_tokens for row in results)
    baseline_cost = sum(row.baseline.recorded_cost_usd for row in results)
    candidate_cost = sum(row.candidate.recorded_cost_usd for row in results)
    new_cost = sum(
        row.baseline.recorded_cost_usd + row.candidate.recorded_cost_usd
        for row in results
        if not row.observed_before_registration
    )
    generation_failures = sum(
        _summary_failures(root / arm.summary)
        for trial in registration.trials
        for arm in (trial.baseline, trial.candidate)
    )
    judge_failures = sum(
        _judge_failures(root / arm.judge)
        for trial in registration.trials
        for arm in (trial.baseline, trial.candidate)
    )
    judge_delta = candidate_judge - baseline_judge
    schema_delta = candidate_schema - baseline_schema
    exact_delta = candidate_exact - baseline_exact
    recall_delta = round(candidate_recall - baseline_recall, 8)
    token_ratio = candidate_tokens / baseline_tokens
    cost_ratio = candidate_cost / baseline_cost
    noninferior = sum(row.judge_correct_delta >= 0 for row in results)
    acceptance = registration.acceptance
    policy_violations = (
        registration.recovery.provider_attempt_policy_violations
        if registration.recovery is not None
        else 0
    )
    gates = (
        _gate("trial_count", len(results), "eq", acceptance.trial_count_must_equal),
        _gate("generation_failures", generation_failures, "eq", acceptance.total_generation_failures_must_equal),
        _gate("judge_failures", judge_failures, "eq", acceptance.total_judge_failures_must_equal),
        _gate("provider_attempt_policy_violations", policy_violations, "le", acceptance.maximum_provider_attempt_policy_violations),
        _gate("schema_complete_delta", schema_delta, "ge", acceptance.minimum_candidate_minus_baseline_schema_complete),
        _gate("aggregate_judge_correct_delta", judge_delta, "ge", acceptance.minimum_aggregate_judge_correct_delta),
        _gate("noninferior_judge_trials", noninferior, "ge", acceptance.minimum_noninferior_judge_trials),
        _gate("aggregate_normalized_exact_delta", exact_delta, "ge", acceptance.minimum_aggregate_normalized_exact_delta),
        _gate("mean_evidence_recall_delta_pp", recall_delta, "ge", acceptance.minimum_mean_evidence_recall_delta_pp),
        _gate("total_token_ratio", token_ratio, "le", acceptance.maximum_total_token_ratio),
        _gate("recorded_provider_cost_ratio", cost_ratio, "le", acceptance.maximum_recorded_provider_cost_ratio),
        _gate("new_recorded_provider_cost_usd", new_cost, "le", acceptance.maximum_new_recorded_provider_cost_usd),
    )
    decision: Literal["accept", "reject"] = (
        "accept" if all(gate.passed for gate in gates) else "reject"
    )
    result = ObligationSpanRepeatDecision(
        created_at=_utc_now(),
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count_per_trial=len(registration.query_ids),
        baseline_judge_correct=baseline_judge,
        candidate_judge_correct=candidate_judge,
        judge_correct_delta=judge_delta,
        noninferior_judge_trials=noninferior,
        baseline_schema_complete=baseline_schema,
        candidate_schema_complete=candidate_schema,
        schema_complete_delta=schema_delta,
        baseline_normalized_exact=baseline_exact,
        candidate_normalized_exact=candidate_exact,
        normalized_exact_delta=exact_delta,
        baseline_mean_evidence_recall_percent=round(baseline_recall, 8),
        candidate_mean_evidence_recall_percent=round(candidate_recall, 8),
        mean_evidence_recall_delta_pp=recall_delta,
        paired_improvements=sum(row.paired_improvements for row in results),
        paired_regressions=sum(row.paired_regressions for row in results),
        baseline_search_calls=sum(row.baseline.search_calls for row in results),
        candidate_search_calls=sum(row.candidate.search_calls for row in results),
        candidate_successful_open_calls=sum(
            row.candidate.successful_open_calls for row in results
        ),
        baseline_total_tokens=baseline_tokens,
        candidate_total_tokens=candidate_tokens,
        total_token_ratio=round(token_ratio, 12),
        baseline_recorded_cost_usd=baseline_cost,
        candidate_recorded_cost_usd=candidate_cost,
        recorded_provider_cost_ratio=round(cost_ratio, 12),
        new_recorded_provider_cost_usd=new_cost,
        known_unobservable_provider_attempts=(
            registration.known_unobservable_provider_attempts
        ),
        provider_attempt_policy_violations=policy_violations,
        trials=results,
        gates=gates,
        next_action=(
            "promote_span_opening_to_official_evaluator_gate"
            if decision == "accept"
            else "freeze_span_opening_and_pivot_layer"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_trial(
    root: Path,
    registration: ObligationSpanRepeatRegistration,
    reference: ObligationSpanFreshRegistration,
    trial: ObligationSpanRepeatTrial,
) -> ObligationSpanRepeatTrialResult:
    baseline = _load_arm(
        root,
        trial.baseline,
        registration.variants[0],
        reference,
        registration.query_ids,
        attempt_cap=(
            2
            if trial.observed_before_registration
            else registration.provider_attempts_per_new_case
        ),
    )
    candidate = _load_arm(
        root,
        trial.candidate,
        registration.variants[1],
        reference,
        registration.query_ids,
        attempt_cap=(
            2
            if trial.observed_before_registration
            else registration.provider_attempts_per_new_case
        ),
    )
    baseline_summary, baseline_diagnostic, baseline_judge, baseline_runs = baseline
    candidate_summary, candidate_diagnostic, candidate_judge, candidate_runs = candidate
    base_labels = {row.query_id: bool(row.correct) for row in baseline_judge.observations}
    cand_labels = {row.query_id: bool(row.correct) for row in candidate_judge.observations}
    improvements = sum(
        not base_labels[query_id] and cand_labels[query_id]
        for query_id in registration.query_ids
    )
    regressions = sum(
        base_labels[query_id] and not cand_labels[query_id]
        for query_id in registration.query_ids
    )
    return ObligationSpanRepeatTrialResult(
        trial_id=trial.trial_id,
        execution_order=trial.execution_order,
        observed_before_registration=trial.observed_before_registration,
        baseline=_metrics(baseline_summary, baseline_diagnostic, baseline_judge, baseline_runs),
        candidate=_metrics(candidate_summary, candidate_diagnostic, candidate_judge, candidate_runs),
        judge_correct_delta=candidate_judge.correct - baseline_judge.correct,
        normalized_exact_delta=(
            candidate_diagnostic.normalized_exact_match
            - baseline_diagnostic.normalized_exact_match
        ),
        evidence_recall_delta_pp=round(
            candidate_diagnostic.evidence_recall_percent
            - baseline_diagnostic.evidence_recall_percent,
            8,
        ),
        paired_improvements=improvements,
        paired_regressions=regressions,
        sources={
            "baseline_summary": _source(root / trial.baseline.summary, root),
            "baseline_diagnostic": _source(root / trial.baseline.diagnostic, root),
            "baseline_judge": _source(root / trial.baseline.judge, root),
            "candidate_summary": _source(root / trial.candidate.summary, root),
            "candidate_diagnostic": _source(root / trial.candidate.diagnostic, root),
            "candidate_judge": _source(root / trial.candidate.judge, root),
        },
    )


def _load_arm(
    root: Path,
    paths: RepeatArmPaths,
    variant: FreshVariant,
    reference: ObligationSpanFreshRegistration,
    query_ids: tuple[str, ...],
    *,
    attempt_cap: int,
) -> tuple[
    PiSmokeSummary,
    DiagnosticSummary,
    DevelopmentJudgeResult,
    tuple[PiBrowseCompRun, ...],
]:
    summary_path = root / paths.summary
    diagnostic_path = root / paths.diagnostic
    judge_path = root / paths.judge
    summary = PiSmokeSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    diagnostic = DiagnosticSummary.model_validate_json(
        diagnostic_path.read_text(encoding="utf-8")
    )
    judge = DevelopmentJudgeResult.model_validate_json(judge_path.read_text(encoding="utf-8"))
    summary_sha = _sha256_file(summary_path)
    if diagnostic.source_summary_sha256 != summary_sha or judge.source_summary_sha256 != summary_sha:
        raise ValueError("repeat derived artifact targets a different summary")
    if tuple(item.query_id for item in summary.items) != query_ids:
        raise ValueError("repeat summary query order changed")
    if {row.query_id for row in diagnostic.rows} != set(query_ids):
        raise ValueError("repeat diagnostic query IDs changed")
    if {row.query_id for row in judge.observations} != set(query_ids):
        raise ValueError("repeat Judge query IDs changed")
    fixed = reference.fixed_contract
    expected_summary = {
        "target_manifest_sha256": fixed.target_manifest.sha256,
        "development_queries_sha256": fixed.query_artifact_normalized_sha256,
        "model": fixed.model,
        "thinking_level": fixed.thinking_level,
        "control_policy": fixed.control_policy,
        "system_prompt_policy": fixed.system_prompt_policy,
        "max_search_results": fixed.max_search_results,
        "retriever_id": variant.retriever_id,
        "retriever_manifest_sha256": fixed.retriever_manifest.sha256,
    }
    for field_name, expected in expected_summary.items():
        if getattr(summary, field_name) != expected:
            raise ValueError(f"repeat summary changed frozen {field_name}")
    runs: list[PiBrowseCompRun] = []
    run_root = (root / paths.run_root).resolve()
    if not run_root.is_relative_to((root / "runs").resolve()):
        raise ValueError("repeat run root escapes ignored runs/")
    for item in summary.items:
        if item.attempt_count > attempt_cap:
            raise ValueError("repeat item exceeded its attempt cap")
        if item.run_path is None or item.run_sha256 is None:
            raise ValueError("repeat item lacks a successful run artifact")
        run_path = root / item.run_path
        if not run_path.resolve().is_relative_to(run_root):
            raise ValueError("repeat item run escapes its registered root")
        if _sha256_file(run_path) != item.run_sha256:
            raise ValueError("repeat item run hash changed")
        run = load_pi_browsecomp_run(run_path)
        if run.adapter_version != variant.adapter_version:
            raise ValueError("repeat run adapter changed")
        if (
            run.model != fixed.model
            or run.thinking_level != fixed.thinking_level
            or run.control_policy != fixed.control_policy
            or run.max_search_results != fixed.max_search_results
        ):
            raise ValueError("repeat run changed frozen generation controls")
        runs.append(run)
    return summary, diagnostic, judge, tuple(runs)


def _metrics(
    summary: PiSmokeSummary,
    diagnostic: DiagnosticSummary,
    judge: DevelopmentJudgeResult,
    runs: tuple[PiBrowseCompRun, ...],
) -> RepeatArmMetrics:
    successful_opens = sum(
        call.outcome == "ok"
        and call.result is not None
        and call.result.outcome == "opened"
        for run in runs
        for call in run.evidence_open_calls
    )
    return RepeatArmMetrics(
        schema_complete=summary.schema_complete or 0,
        judge_correct=judge.correct,
        normalized_exact=diagnostic.normalized_exact_match,
        evidence_recall_percent=diagnostic.evidence_recall_percent,
        search_calls=summary.total_search_calls,
        successful_open_calls=successful_opens,
        total_tokens=summary.total_tokens,
        recorded_cost_usd=summary.total_cost_usd,
    )


def _summary_failures(path: Path) -> int:
    summary = PiSmokeSummary.model_validate_json(path.read_text(encoding="utf-8"))
    return summary.failed + summary.budget_exhausted


def _judge_failures(path: Path) -> int:
    judge = DevelopmentJudgeResult.model_validate_json(path.read_text(encoding="utf-8"))
    return judge.parse_failures + judge.request_failures


def _gate(
    gate_id: str,
    observed: int | float,
    operator: Literal["eq", "ge", "le"],
    threshold: int | float,
) -> RepeatGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return RepeatGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _source(path: Path, root: Path) -> ArtifactReference:
    return ArtifactReference(
        path=path.resolve().relative_to(root.resolve()).as_posix(),
        sha256=_sha256_file(path),
    )
