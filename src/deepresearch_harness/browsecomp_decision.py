from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DiagnosticSummary
from .browsecomp_judge import (
    OfficialJudgeBatchManifest,
    OfficialJudgeComparison,
    aggregate_official_judge_results,
    validate_official_judge_batch,
)
from .browsecomp_plus import normalized_text_file_sha256
from .browsecomp_repeats import (
    RepeatComparisonSummary,
    aggregate_repeat_experiment,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromotionGateManifest(StrictContract):
    schema_version: Literal["browsecomp-plus-promotion-gates-v0"] = (
        "browsecomp-plus-promotion-gates-v0"
    )
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_layer: Literal["dense_retrieval_v0"]
    baseline_retriever_id: Literal["bm25"]
    candidate_retriever_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    candidate_retriever_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold_registration_status: Literal[
        "formalized_after_generation_from_precommitted_thresholds"
    ]
    threshold_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    threshold_source_documents: tuple[str, str]
    required_registration_status: Literal["pre_generation"]
    minimum_trials: int = Field(ge=3)
    minimum_queries_per_trial: int = Field(gt=0)
    minimum_evidence_recall_delta_pp: float = Field(ge=0, le=100)
    minimum_official_accuracy_delta_pp: float = Field(ge=0, le=100)
    require_zero_judge_parse_failures: Literal[True] = True
    sealed_holdout_eligible: Literal[False] = False
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def provenance_is_repository_relative(self) -> "PromotionGateManifest":
        if len(set(self.threshold_source_documents)) != 2:
            raise ValueError("promotion thresholds require two unique source documents")
        for value in self.threshold_source_documents:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("promotion threshold sources must be repository-relative")
        return self


class DecisionSource(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GateResult(StrictContract):
    gate_id: Literal[
        "registration",
        "trial_count",
        "query_count",
        "evidence_recall_delta",
        "official_accuracy_delta",
        "judge_parse_failures",
    ]
    observed: str | float | int
    operator: Literal["eq", "ge"]
    threshold: str | float | int
    passed: bool


class ResourceDelta(StrictContract):
    metric: Literal[
        "search_calls_per_query",
        "output_tokens_per_query",
        "total_tokens_per_query",
        "cost_usd_per_query",
        "latency_ms_per_query",
    ]
    baseline_mean: float = Field(ge=0)
    candidate_mean: float = Field(ge=0)
    candidate_minus_baseline: float


class CandidateFailureProfile(StrictContract):
    query_id: str = Field(min_length=1)
    trials: int = Field(gt=0)
    candidate_correct_trials: int = Field(ge=0)
    baseline_correct_trials: int = Field(ge=0)
    stability: Literal["persistent_failure", "unstable", "stable_success"]
    format_failures: int = Field(ge=0)
    no_relevant_doc_retrieved: int = Field(ge=0)
    relevant_doc_retrieved_but_incorrect: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_fit_trials(self) -> "CandidateFailureProfile":
        if self.candidate_correct_trials > self.trials:
            raise ValueError("candidate correctness exceeds trial count")
        if self.baseline_correct_trials > self.trials:
            raise ValueError("baseline correctness exceeds trial count")
        failures = (
            self.format_failures
            + self.no_relevant_doc_retrieved
            + self.relevant_doc_retrieved_but_incorrect
        )
        if self.candidate_correct_trials + failures != self.trials:
            raise ValueError("candidate failure categories do not cover every trial")
        expected_stability = (
            "persistent_failure"
            if self.candidate_correct_trials == 0
            else "stable_success"
            if self.candidate_correct_trials == self.trials
            else "unstable"
        )
        if self.stability != expected_stability:
            raise ValueError("candidate stability does not match correctness")
        return self


class FailureAggregate(StrictContract):
    query_count: int = Field(gt=0)
    candidate_evaluations: int = Field(gt=0)
    candidate_correct: int = Field(ge=0)
    candidate_incorrect: int = Field(ge=0)
    format_failures: int = Field(ge=0)
    no_relevant_doc_retrieved: int = Field(ge=0)
    relevant_doc_retrieved_but_incorrect: int = Field(ge=0)
    persistent_failure_queries: int = Field(ge=0)
    unstable_queries: int = Field(ge=0)
    stable_success_queries: int = Field(ge=0)
    candidate_improvements: int = Field(ge=0)
    candidate_regressions: int = Field(ge=0)
    paired_ties: int = Field(ge=0)

    @model_validator(mode="after")
    def totals_are_consistent(self) -> "FailureAggregate":
        if self.candidate_correct + self.candidate_incorrect != self.candidate_evaluations:
            raise ValueError("candidate correctness totals do not match evaluations")
        categorized = (
            self.format_failures
            + self.no_relevant_doc_retrieved
            + self.relevant_doc_retrieved_but_incorrect
        )
        if categorized != self.candidate_incorrect:
            raise ValueError("candidate failure categories do not match incorrect count")
        if (
            self.candidate_improvements
            + self.candidate_regressions
            + self.paired_ties
            != self.candidate_evaluations
        ):
            raise ValueError("paired correctness outcomes do not cover evaluations")
        if (
            self.persistent_failure_queries
            + self.unstable_queries
            + self.stable_success_queries
            != self.query_count
        ):
            raise ValueError("query stability counts do not match query_count")
        return self


class LayerPromotionDecision(StrictContract):
    schema_version: Literal["browsecomp-plus-layer-promotion-decision-v0"] = (
        "browsecomp-plus-layer-promotion-decision-v0"
    )
    created_at: str
    status: Literal["development_decision_not_leaderboard"] = (
        "development_decision_not_leaderboard"
    )
    decision: Literal["promote", "reject", "insufficient_scope"]
    candidate_layer: Literal["dense_retrieval_v0"]
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    baseline_retriever_id: str = Field(min_length=1)
    candidate_retriever_id: str = Field(min_length=1)
    trial_count: int = Field(ge=3)
    queries_per_trial: int = Field(gt=0)
    registration_status: Literal[
        "pre_generation", "reconstructed_after_interruption"
    ]
    threshold_registration_status: Literal[
        "formalized_after_generation_from_precommitted_thresholds"
    ]
    recovery_policy_status: Literal[
        "none",
        "preregistered",
        "post_failure_operational_amendment",
    ]
    claim_qualifier: Literal[
        "diagnostic_only",
        "development_gate_clean",
        "development_gate_with_operational_amendment",
    ]
    evidence_recall_delta_pp: float
    official_accuracy_delta_pp: float
    gates: list[GateResult] = Field(min_length=6, max_length=6)
    resource_deltas: list[ResourceDelta] = Field(min_length=5, max_length=5)
    failure_aggregate: FailureAggregate
    failure_profiles: list[CandidateFailureProfile] = Field(min_length=1)
    next_action: Literal[
        "complete_25_query_gate",
        "diagnose_query_compilation_with_replay",
        "diagnose_evidence_to_answer_control",
        "promote_dense_and_cluster_remaining_failures",
    ]
    sources: dict[str, DecisionSource]
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "LayerPromotionDecision":
        gate_by_id = {gate.gate_id: gate for gate in self.gates}
        if len(gate_by_id) != 6:
            raise ValueError("promotion decision must contain each gate exactly once")
        scope_ids = {"registration", "trial_count", "query_count", "judge_parse_failures"}
        scope_passed = all(gate_by_id[gate_id].passed for gate_id in scope_ids)
        mechanism_passed = all(
            gate_by_id[gate_id].passed
            for gate_id in {"evidence_recall_delta", "official_accuracy_delta"}
        )
        expected = (
            "insufficient_scope"
            if not scope_passed
            else "promote"
            if mechanism_passed
            else "reject"
        )
        if self.decision != expected:
            raise ValueError("decision does not match promotion gates")
        if self.failure_aggregate.query_count != len(self.failure_profiles):
            raise ValueError("failure aggregate does not match query profiles")
        expected_sources = {
            "promotion_gates",
            "repeat_experiment",
            "repeat_comparison",
            "judge_batch",
            "judge_execution_registration",
            "judge_execution_result",
            "judge_comparison",
        }
        if set(self.sources) != expected_sources:
            raise ValueError("layer decision source set is incomplete")
        return self


def decide_browsecomp_layer_promotion(
    *,
    repeat_experiment_path: Path,
    repeat_comparison_path: Path,
    target_manifest_path: Path,
    promotion_gates_path: Path,
    judge_batch_manifest_path: Path,
    judge_execution_registration_path: Path,
    judge_execution_result_path: Path,
    judge_comparison_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> LayerPromotionDecision:
    repository_root = _find_repository_root(repeat_experiment_path.resolve())
    _require_under_runs(repeat_experiment_path, repository_root)
    _require_under_runs(repeat_comparison_path, repository_root)
    _require_under_runs(judge_batch_manifest_path, repository_root)
    _require_under_runs(judge_execution_registration_path, repository_root)
    _require_under_runs(judge_execution_result_path, repository_root)
    _require_under_runs(judge_comparison_path, repository_root)
    _require_under_runs(output_path, repository_root)
    existing: LayerPromotionDecision | None = None
    if output_path.exists() and not validate_existing:
        raise ValueError("layer-promotion decision output already exists")
    if output_path.exists():
        existing = LayerPromotionDecision.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

    gates_manifest = PromotionGateManifest.model_validate_json(
        promotion_gates_path.read_text(encoding="utf-8")
    )
    target_hash = normalized_text_file_sha256(target_manifest_path)
    if gates_manifest.target_manifest_sha256 != target_hash:
        raise ValueError("promotion gates target a different benchmark manifest")

    repeat = aggregate_repeat_experiment(
        manifest_path=repeat_experiment_path,
        target_manifest_path=target_manifest_path,
        output_path=repeat_comparison_path,
        validate_existing=True,
    )
    if (
        repeat.baseline.retriever_id != gates_manifest.baseline_retriever_id
        or repeat.candidate.retriever_id != gates_manifest.candidate_retriever_id
        or repeat.candidate.retriever_manifest_sha256
        != gates_manifest.candidate_retriever_manifest_sha256
    ):
        raise ValueError("promotion gates target different retriever variants")
    judge = aggregate_official_judge_results(
        batch_manifest_path=judge_batch_manifest_path,
        execution_registration_path=judge_execution_registration_path,
        execution_result_path=judge_execution_result_path,
        output_path=judge_comparison_path,
        validate_existing=True,
    )
    batch = validate_official_judge_batch(judge_batch_manifest_path)
    _validate_cross_artifact_contract(
        repeat=repeat,
        judge=judge,
        batch=batch,
        repeat_experiment_path=repeat_experiment_path,
        repeat_comparison_path=repeat_comparison_path,
        judge_batch_manifest_path=judge_batch_manifest_path,
    )

    evidence_delta = round(
        repeat.candidate.evidence_recall_percent.mean
        - repeat.baseline.evidence_recall_percent.mean,
        6,
    )
    accuracy_delta = round(
        judge.candidate.trial_accuracy_percent.mean
        - judge.baseline.trial_accuracy_percent.mean,
        6,
    )
    gate_results = _gate_results(
        repeat=repeat,
        judge=judge,
        manifest=gates_manifest,
        evidence_delta=evidence_delta,
        accuracy_delta=accuracy_delta,
    )
    gate_by_id = {gate.gate_id: gate for gate in gate_results}
    scope_passed = all(
        gate_by_id[gate_id].passed
        for gate_id in {"registration", "trial_count", "query_count", "judge_parse_failures"}
    )
    mechanism_passed = (
        gate_by_id["evidence_recall_delta"].passed
        and gate_by_id["official_accuracy_delta"].passed
    )
    decision: Literal["promote", "reject", "insufficient_scope"] = (
        "insufficient_scope"
        if not scope_passed
        else "promote"
        if mechanism_passed
        else "reject"
    )

    profiles, failure_aggregate = analyze_candidate_failures(
        repeat=repeat,
        judge=judge,
        repository_root=repository_root,
    )
    if decision == "insufficient_scope":
        next_action = "complete_25_query_gate"
        claim_qualifier = "diagnostic_only"
    elif not gate_by_id["evidence_recall_delta"].passed:
        next_action = "diagnose_query_compilation_with_replay"
        claim_qualifier = _development_claim_qualifier(repeat)
    elif not gate_by_id["official_accuracy_delta"].passed:
        next_action = "diagnose_evidence_to_answer_control"
        claim_qualifier = _development_claim_qualifier(repeat)
    else:
        next_action = "promote_dense_and_cluster_remaining_failures"
        claim_qualifier = _development_claim_qualifier(repeat)

    artifact = LayerPromotionDecision(
        created_at=(
            existing.created_at
            if existing is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        decision=decision,
        candidate_layer=gates_manifest.candidate_layer,
        model=repeat.model,
        baseline_retriever_id=repeat.baseline.retriever_id,
        candidate_retriever_id=repeat.candidate.retriever_id,
        trial_count=repeat.trial_count,
        queries_per_trial=repeat.queries_per_trial,
        registration_status=repeat.registration_status,
        threshold_registration_status=gates_manifest.threshold_registration_status,
        recovery_policy_status=repeat.recovery_policy_status,
        claim_qualifier=claim_qualifier,
        evidence_recall_delta_pp=evidence_delta,
        official_accuracy_delta_pp=accuracy_delta,
        gates=gate_results,
        resource_deltas=_resource_deltas(repeat),
        failure_aggregate=failure_aggregate,
        failure_profiles=profiles,
        next_action=next_action,
        sources={
            "promotion_gates": _source(promotion_gates_path, repository_root),
            "repeat_experiment": _source(repeat_experiment_path, repository_root),
            "repeat_comparison": _source(repeat_comparison_path, repository_root),
            "judge_batch": _source(judge_batch_manifest_path, repository_root),
            "judge_execution_registration": _source(
                judge_execution_registration_path, repository_root
            ),
            "judge_execution_result": _source(
                judge_execution_result_path, repository_root
            ),
            "judge_comparison": _source(judge_comparison_path, repository_root),
        },
        limitations=(
            "Development questions only; no sealed-holdout access or leaderboard submission.",
            "Repeated trials measure harness reliability on one fixed query set, not additional independent questions.",
            "Observed differences belong to the frozen-model harness configuration, not intrinsic model capability.",
        ),
    )
    if existing is not None:
        if artifact != existing:
            raise ValueError("existing layer decision no longer matches its sources")
        return existing
    _atomic_write(output_path, artifact.model_dump_json(indent=2))
    return artifact


def _validate_cross_artifact_contract(
    *,
    repeat: RepeatComparisonSummary,
    judge: OfficialJudgeComparison,
    batch: OfficialJudgeBatchManifest,
    repeat_experiment_path: Path,
    repeat_comparison_path: Path,
    judge_batch_manifest_path: Path,
) -> None:
    if batch.repeat_experiment_sha256 != normalized_text_file_sha256(
        repeat_experiment_path
    ):
        raise ValueError("judge batch targets a different repeat experiment")
    if batch.repeat_comparison_sha256 != normalized_text_file_sha256(
        repeat_comparison_path
    ):
        raise ValueError("judge batch targets a different repeat comparison")
    if judge.batch_manifest_sha256 != sha256(
        judge_batch_manifest_path.read_bytes()
    ).hexdigest():
        raise ValueError("judge comparison targets a different batch manifest")
    if (
        batch.generator_model != repeat.model
        or batch.registration_status != repeat.registration_status
        or batch.trial_count != repeat.trial_count
        or batch.queries_per_trial != repeat.queries_per_trial
        or judge.trial_count != repeat.trial_count
        or judge.queries_per_trial != repeat.queries_per_trial
    ):
        raise ValueError("judge and repeat comparison contracts differ")


def _gate_results(
    *,
    repeat: RepeatComparisonSummary,
    judge: OfficialJudgeComparison,
    manifest: PromotionGateManifest,
    evidence_delta: float,
    accuracy_delta: float,
) -> list[GateResult]:
    return [
        GateResult(
            gate_id="registration",
            observed=repeat.registration_status,
            operator="eq",
            threshold=manifest.required_registration_status,
            passed=repeat.registration_status == manifest.required_registration_status,
        ),
        GateResult(
            gate_id="trial_count",
            observed=repeat.trial_count,
            operator="ge",
            threshold=manifest.minimum_trials,
            passed=repeat.trial_count >= manifest.minimum_trials,
        ),
        GateResult(
            gate_id="query_count",
            observed=repeat.queries_per_trial,
            operator="ge",
            threshold=manifest.minimum_queries_per_trial,
            passed=repeat.queries_per_trial >= manifest.minimum_queries_per_trial,
        ),
        GateResult(
            gate_id="evidence_recall_delta",
            observed=evidence_delta,
            operator="ge",
            threshold=manifest.minimum_evidence_recall_delta_pp,
            passed=evidence_delta >= manifest.minimum_evidence_recall_delta_pp,
        ),
        GateResult(
            gate_id="official_accuracy_delta",
            observed=accuracy_delta,
            operator="ge",
            threshold=manifest.minimum_official_accuracy_delta_pp,
            passed=accuracy_delta >= manifest.minimum_official_accuracy_delta_pp,
        ),
        GateResult(
            gate_id="judge_parse_failures",
            observed=judge.parse_failures,
            operator="eq",
            threshold=0,
            passed=judge.parse_failures == 0,
        ),
    ]


def analyze_candidate_failures(
    *,
    repeat: RepeatComparisonSummary,
    judge: OfficialJudgeComparison,
    repository_root: Path,
) -> tuple[list[CandidateFailureProfile], FailureAggregate]:
    candidate_diagnostics: dict[tuple[str, str], object] = {}
    for trial in repeat.trials:
        diagnostic_path = _resolve_repository_path(
            trial.candidate_diagnostic.path, repository_root, require_runs=True
        )
        diagnostic_bytes = diagnostic_path.read_bytes()
        if sha256(diagnostic_bytes).hexdigest() != trial.candidate_diagnostic.sha256:
            raise ValueError("candidate diagnostic hash changed after repeat aggregation")
        diagnostic = DiagnosticSummary.model_validate_json(diagnostic_bytes)
        for row in diagnostic.rows:
            key = (trial.trial_id, row.query_id)
            if key in candidate_diagnostics:
                raise ValueError("candidate diagnostic rows are duplicated")
            candidate_diagnostics[key] = row

    official_by_key = {
        (observation.trial_id, observation.query_id, observation.variant): observation
        for observation in judge.observations
    }
    expected_keys = {
        (trial.trial_id, query_id)
        for trial in repeat.trials
        for query_id in {
            observation.query_id
            for observation in judge.observations
            if observation.trial_id == trial.trial_id
        }
    }
    if set(candidate_diagnostics) != expected_keys:
        raise ValueError("candidate diagnostics and official observations differ")

    query_ids = sorted({query_id for _, query_id in candidate_diagnostics})
    profiles: list[CandidateFailureProfile] = []
    category_totals = {
        "format_failures": 0,
        "no_relevant_doc_retrieved": 0,
        "relevant_doc_retrieved_but_incorrect": 0,
    }
    candidate_correct_total = 0
    improvements = 0
    regressions = 0
    ties = 0
    for query_id in query_ids:
        candidate_correct = 0
        baseline_correct = 0
        query_categories = {key: 0 for key in category_totals}
        for trial in repeat.trials:
            key = (trial.trial_id, query_id)
            candidate_observation = official_by_key.get((*key, "candidate"))
            baseline_observation = official_by_key.get((*key, "baseline"))
            diagnostic = candidate_diagnostics.get(key)
            if candidate_observation is None or baseline_observation is None or diagnostic is None:
                raise ValueError("missing paired official or diagnostic observation")
            if candidate_observation.prediction_sha256 != diagnostic.prediction_sha256:
                raise ValueError("candidate judge and diagnostic predictions differ")
            candidate_correct += int(candidate_observation.correct)
            baseline_correct += int(baseline_observation.correct)
            if candidate_observation.correct and not baseline_observation.correct:
                improvements += 1
            elif baseline_observation.correct and not candidate_observation.correct:
                regressions += 1
            else:
                ties += 1
            if candidate_observation.correct:
                continue
            if not diagnostic.answer_schema_complete or not diagnostic.exact_answer_extracted:
                category = "format_failures"
            elif max(diagnostic.evidence_recall or 0.0, diagnostic.gold_recall or 0.0) == 0:
                category = "no_relevant_doc_retrieved"
            else:
                category = "relevant_doc_retrieved_but_incorrect"
            query_categories[category] += 1
            category_totals[category] += 1
        candidate_correct_total += candidate_correct
        stability: Literal["persistent_failure", "unstable", "stable_success"] = (
            "persistent_failure"
            if candidate_correct == 0
            else "stable_success"
            if candidate_correct == repeat.trial_count
            else "unstable"
        )
        profiles.append(
            CandidateFailureProfile(
                query_id=query_id,
                trials=repeat.trial_count,
                candidate_correct_trials=candidate_correct,
                baseline_correct_trials=baseline_correct,
                stability=stability,
                **query_categories,
            )
        )

    candidate_evaluations = repeat.trial_count * repeat.queries_per_trial
    aggregate = FailureAggregate(
        query_count=len(profiles),
        candidate_evaluations=candidate_evaluations,
        candidate_correct=candidate_correct_total,
        candidate_incorrect=candidate_evaluations - candidate_correct_total,
        **category_totals,
        persistent_failure_queries=sum(
            profile.stability == "persistent_failure" for profile in profiles
        ),
        unstable_queries=sum(profile.stability == "unstable" for profile in profiles),
        stable_success_queries=sum(
            profile.stability == "stable_success" for profile in profiles
        ),
        candidate_improvements=improvements,
        candidate_regressions=regressions,
        paired_ties=ties,
    )
    return profiles, aggregate


def _resource_deltas(repeat: RepeatComparisonSummary) -> list[ResourceDelta]:
    metrics = (
        "search_calls_per_query",
        "output_tokens_per_query",
        "total_tokens_per_query",
        "cost_usd_per_query",
        "latency_ms_per_query",
    )
    rows = []
    for metric in metrics:
        baseline = getattr(repeat.baseline, metric).mean
        candidate = getattr(repeat.candidate, metric).mean
        rows.append(
            ResourceDelta(
                metric=metric,
                baseline_mean=baseline,
                candidate_mean=candidate,
                candidate_minus_baseline=round(candidate - baseline, 12),
            )
        )
    return rows


def _development_claim_qualifier(
    repeat: RepeatComparisonSummary,
) -> Literal[
    "development_gate_clean", "development_gate_with_operational_amendment"
]:
    return (
        "development_gate_with_operational_amendment"
        if repeat.recovery_policy_status == "post_failure_operational_amendment"
        else "development_gate_clean"
    )


def _source(path: Path, repository_root: Path) -> DecisionSource:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return DecisionSource(
        path=display_path,
        sha256=sha256(resolved.read_bytes()).hexdigest(),
    )


def _resolve_repository_path(
    value: str, repository_root: Path, *, require_runs: bool
) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("decision source paths must be repository-relative")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise ValueError("decision source path escapes the repository")
    if require_runs and not resolved.is_relative_to(
        (repository_root / "runs").resolve()
    ):
        raise ValueError("benchmark-derived decision sources must remain under runs/")
    if not resolved.is_file():
        raise ValueError(f"decision source artifact is missing: {value}")
    return resolved


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate repository root for layer decision")


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("benchmark decision artifacts must remain under ignored runs/")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)
