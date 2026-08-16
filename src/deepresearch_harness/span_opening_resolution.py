from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DiagnosticSummary
from .development_judge import DevelopmentJudgeResult
from .evidence_span_oracle import ArtifactReference
from .obligation_span_fresh import ObligationSpanFreshDecision
from .pi_browsecomp import PiSmokeSummary
from .repeat_execution_audit import RepeatExecutionAuditResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidTrialMetrics(StrictContract):
    trial_id: Literal["trial-1", "trial-2"]
    baseline_judge_correct: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    judge_correct_delta: int
    baseline_normalized_exact: int = Field(ge=0)
    candidate_normalized_exact: int = Field(ge=0)
    normalized_exact_delta: int
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_evidence_recall_percent: float = Field(ge=0, le=100)
    evidence_recall_delta_pp: float
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0)
    baseline_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(gt=0)
    baseline_recorded_cost_usd: float = Field(gt=0)
    candidate_recorded_cost_usd: float = Field(gt=0)


class ResolutionGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class SpanOpeningResolution(StrictContract):
    schema_version: Literal["obligation-span-opening-valid-trial-resolution-v0"] = (
        "obligation-span-opening-valid-trial-resolution-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["post_execution_resolution_not_official"] = (
        "post_execution_resolution_not_official"
    )
    decision: Literal[
        "continue_clean_repetition",
        "freeze_obligation_span_opening",
    ]
    valid_trial_count: Literal[2] = 2
    invalid_trial_count: Literal[1] = 1
    query_count_per_trial: int = Field(gt=0)
    trials: tuple[ValidTrialMetrics, ValidTrialMetrics]
    aggregate_baseline_judge_correct: int = Field(ge=0)
    aggregate_candidate_judge_correct: int = Field(ge=0)
    aggregate_judge_correct_delta: int
    aggregate_baseline_normalized_exact: int = Field(ge=0)
    aggregate_candidate_normalized_exact: int = Field(ge=0)
    aggregate_normalized_exact_delta: int
    baseline_mean_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_mean_evidence_recall_percent: float = Field(ge=0, le=100)
    mean_evidence_recall_delta_pp: float
    total_token_ratio: float = Field(ge=0)
    recorded_provider_cost_ratio: float = Field(ge=0)
    combined_recorded_provider_cost_usd: float = Field(gt=0)
    known_unobservable_provider_attempts: int = Field(ge=1)
    cost_observability: Literal[
        "recorded_lower_bound_with_unobservable_failed_attempts"
    ] = "recorded_lower_bound_with_unobservable_failed_attempts"
    invalid_trial_judge_calls: Literal[0] = 0
    resolution_provider_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    gates: tuple[ResolutionGate, ...] = Field(min_length=1)
    sources: dict[str, ArtifactReference]
    next_action: Literal["diagnose_v10_cross_trial_failure_stability"] = (
        "diagnose_v10_cross_trial_failure_stability"
    )
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "SpanOpeningResolution":
        expected = (
            "continue_clean_repetition"
            if all(gate.passed for gate in self.gates)
            else "freeze_obligation_span_opening"
        )
        if self.decision != expected:
            raise ValueError("span-opening resolution differs from gates")
        if [trial.trial_id for trial in self.trials] != ["trial-1", "trial-2"]:
            raise ValueError("valid trials must be ordered trial-1 then trial-2")
        return self


def resolve_span_opening_valid_trials(
    *,
    fresh_decision_path: Path,
    trial2_baseline_summary_path: Path,
    trial2_candidate_summary_path: Path,
    trial2_baseline_diagnostic_path: Path,
    trial2_candidate_diagnostic_path: Path,
    trial2_baseline_judge_path: Path,
    trial2_candidate_judge_path: Path,
    execution_audit_path: Path,
    output_path: Path,
) -> SpanOpeningResolution:
    paths = {
        "fresh_decision": fresh_decision_path.resolve(),
        "trial2_baseline_summary": trial2_baseline_summary_path.resolve(),
        "trial2_candidate_summary": trial2_candidate_summary_path.resolve(),
        "trial2_baseline_diagnostic": trial2_baseline_diagnostic_path.resolve(),
        "trial2_candidate_diagnostic": trial2_candidate_diagnostic_path.resolve(),
        "trial2_baseline_judge": trial2_baseline_judge_path.resolve(),
        "trial2_candidate_judge": trial2_candidate_judge_path.resolve(),
        "execution_audit": execution_audit_path.resolve(),
    }
    root = _common_repository_root(paths.values())
    output_path = output_path.resolve()
    if not output_path.is_relative_to((root / "runs").resolve()):
        raise ValueError("span-opening resolution output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("span-opening resolution output already exists")

    fresh = ObligationSpanFreshDecision.model_validate_json(
        paths["fresh_decision"].read_text(encoding="utf-8")
    )
    baseline_summary = PiSmokeSummary.model_validate_json(
        paths["trial2_baseline_summary"].read_text(encoding="utf-8")
    )
    candidate_summary = PiSmokeSummary.model_validate_json(
        paths["trial2_candidate_summary"].read_text(encoding="utf-8")
    )
    baseline_diagnostic = DiagnosticSummary.model_validate_json(
        paths["trial2_baseline_diagnostic"].read_text(encoding="utf-8")
    )
    candidate_diagnostic = DiagnosticSummary.model_validate_json(
        paths["trial2_candidate_diagnostic"].read_text(encoding="utf-8")
    )
    baseline_judge = DevelopmentJudgeResult.model_validate_json(
        paths["trial2_baseline_judge"].read_text(encoding="utf-8")
    )
    candidate_judge = DevelopmentJudgeResult.model_validate_json(
        paths["trial2_candidate_judge"].read_text(encoding="utf-8")
    )
    execution_audit = RepeatExecutionAuditResult.model_validate_json(
        paths["execution_audit"].read_text(encoding="utf-8")
    )
    if execution_audit.decision != "reject_execution":
        raise ValueError("resolution requires a rejected invalid execution")
    _validate_trial2_identity(
        baseline_summary,
        candidate_summary,
        baseline_diagnostic,
        candidate_diagnostic,
        baseline_judge,
        candidate_judge,
    )
    if baseline_judge.parse_failures or baseline_judge.request_failures:
        raise ValueError("trial-2 baseline Judge is incomplete")
    if candidate_judge.parse_failures or candidate_judge.request_failures:
        raise ValueError("trial-2 candidate Judge is incomplete")

    trial1 = ValidTrialMetrics(
        trial_id="trial-1",
        baseline_judge_correct=fresh.baseline_judge_correct,
        candidate_judge_correct=fresh.candidate_judge_correct,
        judge_correct_delta=fresh.judge_correct_delta,
        baseline_normalized_exact=fresh.baseline_normalized_exact,
        candidate_normalized_exact=fresh.candidate_normalized_exact,
        normalized_exact_delta=fresh.normalized_exact_delta,
        baseline_evidence_recall_percent=fresh.baseline_evidence_recall_percent,
        candidate_evidence_recall_percent=fresh.candidate_evidence_recall_percent,
        evidence_recall_delta_pp=fresh.evidence_recall_delta_pp,
        baseline_search_calls=fresh.baseline_search_calls,
        candidate_search_calls=fresh.candidate_search_calls,
        baseline_total_tokens=fresh.baseline_total_tokens,
        candidate_total_tokens=fresh.candidate_total_tokens,
        baseline_recorded_cost_usd=fresh.baseline_cost_usd,
        candidate_recorded_cost_usd=fresh.candidate_cost_usd,
    )
    trial2 = ValidTrialMetrics(
        trial_id="trial-2",
        baseline_judge_correct=baseline_judge.correct,
        candidate_judge_correct=candidate_judge.correct,
        judge_correct_delta=candidate_judge.correct - baseline_judge.correct,
        baseline_normalized_exact=baseline_diagnostic.normalized_exact_match,
        candidate_normalized_exact=candidate_diagnostic.normalized_exact_match,
        normalized_exact_delta=(
            candidate_diagnostic.normalized_exact_match
            - baseline_diagnostic.normalized_exact_match
        ),
        baseline_evidence_recall_percent=baseline_diagnostic.evidence_recall_percent,
        candidate_evidence_recall_percent=candidate_diagnostic.evidence_recall_percent,
        evidence_recall_delta_pp=round(
            candidate_diagnostic.evidence_recall_percent
            - baseline_diagnostic.evidence_recall_percent,
            8,
        ),
        baseline_search_calls=baseline_summary.total_search_calls,
        candidate_search_calls=candidate_summary.total_search_calls,
        baseline_total_tokens=baseline_summary.total_tokens,
        candidate_total_tokens=candidate_summary.total_tokens,
        baseline_recorded_cost_usd=baseline_summary.total_cost_usd,
        candidate_recorded_cost_usd=candidate_summary.total_cost_usd,
    )
    trials = (trial1, trial2)
    baseline_judge_correct = sum(trial.baseline_judge_correct for trial in trials)
    candidate_judge_correct = sum(trial.candidate_judge_correct for trial in trials)
    baseline_exact = sum(trial.baseline_normalized_exact for trial in trials)
    candidate_exact = sum(trial.candidate_normalized_exact for trial in trials)
    baseline_recall = sum(
        trial.baseline_evidence_recall_percent for trial in trials
    ) / 2
    candidate_recall = sum(
        trial.candidate_evidence_recall_percent for trial in trials
    ) / 2
    baseline_tokens = sum(trial.baseline_total_tokens for trial in trials)
    candidate_tokens = sum(trial.candidate_total_tokens for trial in trials)
    baseline_cost = sum(trial.baseline_recorded_cost_usd for trial in trials)
    candidate_cost = sum(trial.candidate_recorded_cost_usd for trial in trials)
    judge_delta = candidate_judge_correct - baseline_judge_correct
    exact_delta = candidate_exact - baseline_exact
    recall_delta = round(candidate_recall - baseline_recall, 8)
    token_ratio = candidate_tokens / baseline_tokens
    cost_ratio = candidate_cost / baseline_cost
    gates = (
        _gate("valid_trial_count", len(trials), "eq", 2),
        _gate("aggregate_judge_correct_delta", judge_delta, "ge", 0),
        _gate("aggregate_normalized_exact_delta", exact_delta, "ge", 0),
        _gate("mean_evidence_recall_delta_pp", recall_delta, "ge", -1.0),
        _gate("total_token_ratio", token_ratio, "le", 1.15),
        _gate("recorded_provider_cost_ratio", cost_ratio, "le", 1.25),
        _gate("invalid_trial_judge_calls", 0, "eq", 0),
    )
    decision: Literal[
        "continue_clean_repetition", "freeze_obligation_span_opening"
    ] = (
        "continue_clean_repetition"
        if all(gate.passed for gate in gates)
        else "freeze_obligation_span_opening"
    )
    result = SpanOpeningResolution(
        created_at=_utc_now(),
        decision=decision,
        query_count_per_trial=baseline_summary.query_count,
        trials=trials,
        aggregate_baseline_judge_correct=baseline_judge_correct,
        aggregate_candidate_judge_correct=candidate_judge_correct,
        aggregate_judge_correct_delta=judge_delta,
        aggregate_baseline_normalized_exact=baseline_exact,
        aggregate_candidate_normalized_exact=candidate_exact,
        aggregate_normalized_exact_delta=exact_delta,
        baseline_mean_evidence_recall_percent=round(baseline_recall, 8),
        candidate_mean_evidence_recall_percent=round(candidate_recall, 8),
        mean_evidence_recall_delta_pp=recall_delta,
        total_token_ratio=round(token_ratio, 12),
        recorded_provider_cost_ratio=round(cost_ratio, 12),
        combined_recorded_provider_cost_usd=baseline_cost + candidate_cost,
        known_unobservable_provider_attempts=(
            execution_audit.known_unobservable_provider_attempts
        ),
        gates=gates,
        sources={
            name: ArtifactReference(
                path=path.relative_to(root).as_posix(), sha256=_sha256_file(path)
            )
            for name, path in paths.items()
        },
        claim_boundary=(
            "This post-execution resolution uses only the two complete paired development "
            "trials. The all-search-failed trial-3 candidate is excluded and received no Judge "
            "call. It supports freezing this harness mechanism, not a zero score for the model, "
            "an official benchmark result, or a frozen-model capability claim. Recorded provider "
            "cost is a lower bound because two validation-failure attempts are unobservable."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _common_repository_root(paths: object) -> Path:
    resolved = list(paths)
    for path in resolved:
        for parent in path.parents:
            if (parent / "pyproject.toml").is_file() and (parent / "runs").is_dir():
                if all(item.is_relative_to(parent) for item in resolved):
                    return parent
    raise ValueError("input artifacts do not share a repository root")


def _validate_trial2_identity(
    baseline_summary: PiSmokeSummary,
    candidate_summary: PiSmokeSummary,
    baseline_diagnostic: DiagnosticSummary,
    candidate_diagnostic: DiagnosticSummary,
    baseline_judge: DevelopmentJudgeResult,
    candidate_judge: DevelopmentJudgeResult,
) -> None:
    sequences = (
        [item.query_id for item in baseline_summary.items],
        [item.query_id for item in candidate_summary.items],
        [row.query_id for row in baseline_diagnostic.rows],
        [row.query_id for row in candidate_diagnostic.rows],
        [row.query_id for row in baseline_judge.observations],
        [row.query_id for row in candidate_judge.observations],
    )
    expected = set(sequences[0])
    if len(expected) != len(sequences[0]) or any(
        len(sequence) != len(expected) or set(sequence) != expected
        for sequence in sequences[1:]
    ):
        raise ValueError("trial-2 artifacts differ in query identity")
    if baseline_summary.failed or candidate_summary.failed:
        raise ValueError("trial-2 generation contains failed cases")


def _gate(
    gate_id: str,
    observed: int | float,
    operator: Literal["eq", "ge", "le"],
    threshold: int | float,
) -> ResolutionGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return ResolutionGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )
