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
from .browsecomp_evaluation import DiagnosticSummary
from .browsecomp_plus import normalized_text_file_sha256
from .browsecomp_repeats import (
    RepeatComparisonSummary,
    RepeatExperimentManifest,
    aggregate_repeat_experiment,
)
from .pi_browsecomp import PiSmokeSummary
from .repeat_development_judge import RepeatDevelopmentJudgeComparison


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceBandwidthGate(StrictContract):
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
        "candidate_to_baseline_search_call_ratio_min",
        "candidate_to_baseline_search_call_ratio_max",
        "candidate_to_baseline_total_token_ratio",
        "candidate_to_baseline_provider_cost_ratio",
        "combined_provider_cost",
        "final_provider_failures",
    ]
    observed: str | float | int
    operator: Literal["eq", "ge", "le"]
    threshold: str | float | int
    passed: bool


class SynthesisLossDiagnostic(StrictContract):
    paired_comparisons: int = Field(gt=0)
    candidate_improvements: int = Field(ge=0)
    candidate_regressions: int = Field(ge=0)
    paired_ties: int = Field(ge=0)
    regressions_with_candidate_relevant_evidence: int = Field(ge=0)
    regressions_with_candidate_evidence_advantage: int = Field(ge=0)
    improvements_with_candidate_evidence_advantage: int = Field(ge=0)
    baseline_incorrect: int = Field(ge=0)
    baseline_format_failures: int = Field(ge=0)
    baseline_no_relevant_doc_retrieved: int = Field(ge=0)
    baseline_relevant_evidence_but_incorrect: int = Field(ge=0)
    candidate_incorrect: int = Field(ge=0)
    candidate_format_failures: int = Field(ge=0)
    candidate_no_relevant_doc_retrieved: int = Field(ge=0)
    candidate_relevant_evidence_but_incorrect: int = Field(ge=0)
    interpretation: Literal[
        "retrieval_gain_did_not_transfer_to_answer_accuracy"
    ] = "retrieval_gain_did_not_transfer_to_answer_accuracy"

    @model_validator(mode="after")
    def counts_are_consistent(self) -> "SynthesisLossDiagnostic":
        if (
            self.candidate_improvements
            + self.candidate_regressions
            + self.paired_ties
            != self.paired_comparisons
        ):
            raise ValueError("paired synthesis outcomes do not cover comparisons")
        if self.regressions_with_candidate_relevant_evidence > self.candidate_regressions:
            raise ValueError("relevant-evidence regressions exceed regressions")
        if self.regressions_with_candidate_evidence_advantage > (
            self.regressions_with_candidate_relevant_evidence
        ):
            raise ValueError("evidence-advantage regressions exceed relevant regressions")
        if self.improvements_with_candidate_evidence_advantage > (
            self.candidate_improvements
        ):
            raise ValueError("evidence-advantage improvements exceed improvements")
        if (
            self.baseline_format_failures
            + self.baseline_no_relevant_doc_retrieved
            + self.baseline_relevant_evidence_but_incorrect
            != self.baseline_incorrect
        ):
            raise ValueError("baseline failure categories do not cover failures")
        if (
            self.candidate_format_failures
            + self.candidate_no_relevant_doc_retrieved
            + self.candidate_relevant_evidence_but_incorrect
            != self.candidate_incorrect
        ):
            raise ValueError("candidate failure categories do not cover failures")
        return self


class EvidenceBandwidthDecision(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-evidence-bandwidth-decision-v0"
    ] = "browsecomp-plus-evidence-bandwidth-decision-v0"
    created_at: str
    status: Literal[
        "development_decision_not_leaderboard"
    ] = "development_decision_not_leaderboard"
    judge_metric_status: Literal[
        "calibrated_development_diagnostic_not_official"
    ] = "calibrated_development_diagnostic_not_official"
    decision: Literal["promote", "reject"]
    candidate_layer: Literal[
        "evidence_bandwidth_exchange_v0"
    ] = "evidence_bandwidth_exchange_v0"
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    baseline_retriever_id: str = Field(min_length=1)
    candidate_retriever_id: str = Field(min_length=1)
    baseline_result_count: int = Field(gt=0)
    candidate_result_count: int = Field(gt=0)
    trial_count: int = Field(ge=3)
    queries_per_trial: int = Field(gt=0)
    strict_exact_delta_pp: float
    evidence_recall_delta_pp: float
    gold_recall_delta_pp: float
    judge_accuracy_delta_pp: float
    candidate_to_baseline_search_call_ratio: float = Field(ge=0)
    candidate_to_baseline_total_token_ratio: float = Field(ge=0)
    candidate_to_baseline_provider_cost_ratio: float = Field(ge=0)
    candidate_to_baseline_latency_ratio: float = Field(ge=0)
    combined_provider_cost_usd: float = Field(ge=0)
    gates: list[EvidenceBandwidthGate] = Field(min_length=16, max_length=16)
    failure_aggregate: FailureAggregate
    failure_profiles: list[CandidateFailureProfile] = Field(min_length=1)
    synthesis_loss: SynthesisLossDiagnostic
    next_action: Literal[
        "audit_incomplete_or_structurally_invalid_grid",
        "diagnose_retrieval_bandwidth_from_saved_replays",
        "diagnose_evidence_selection_and_synthesis_from_saved_runs",
        "calibrate_adaptive_evidence_bandwidth_budget",
        "promote_evidence_bandwidth_and_cluster_remaining_failures",
    ]
    sources: dict[str, DecisionSource]
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_registered_gates(self) -> "EvidenceBandwidthDecision":
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
            "candidate_to_baseline_search_call_ratio_min",
            "candidate_to_baseline_search_call_ratio_max",
            "candidate_to_baseline_total_token_ratio",
            "candidate_to_baseline_provider_cost_ratio",
            "combined_provider_cost",
            "final_provider_failures",
        }
        gate_ids = {gate.gate_id for gate in self.gates}
        if gate_ids != expected_ids or len(gate_ids) != len(self.gates):
            raise ValueError("evidence bandwidth gate set is incomplete")
        expected_decision = "promote" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected_decision:
            raise ValueError("evidence bandwidth decision differs from its gates")
        if self.failure_aggregate.query_count != len(self.failure_profiles):
            raise ValueError("evidence bandwidth failure profile count differs")
        expected_sources = {
            "preregistration",
            "candidate_manifest",
            "repeat_experiment",
            "repeat_comparison",
            "target_manifest",
            "judge_manifest",
            "judge_calibration",
            "judge_execution_registration",
            "judge_execution_result",
            "judge_comparison",
        }
        if set(self.sources) != expected_sources:
            raise ValueError("evidence bandwidth decision source set is incomplete")
        return self


def decide_evidence_bandwidth(
    *,
    preregistration_path: Path,
    candidate_manifest_path: Path,
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
) -> EvidenceBandwidthDecision:
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
        raise ValueError("evidence bandwidth decision already exists")
    if output_path.exists():
        existing = EvidenceBandwidthDecision.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

    preregistration = _load_object(preregistration_path)
    if (
        preregistration.get("schema_version")
        != "browsecomp-plus-evidence-bandwidth-confirmation-v0"
        or preregistration.get("status") != "preregistered_not_run"
    ):
        raise ValueError("evidence bandwidth preregistration is not frozen")
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
    execution_result = _load_object(judge_execution_result_path)
    calibration_bytes = judge_calibration_path.read_bytes()

    fixed = _require_object(preregistration, "fixed_contract")
    execution = _require_object(preregistration, "execution")
    gates_contract = _require_object(preregistration, "acceptance_gates")
    fresh_queries = _require_object(preregistration, "fresh_query_selection")
    baseline_contract = _require_object(fixed, "baseline")
    candidate_contract = _require_object(fixed, "candidate")

    if execution.get("repeat_manifest_sha256") != sha256(repeat_bytes).hexdigest():
        raise ValueError("preregistration targets another repeat manifest")
    if fixed.get("model") != repeat.model or fixed.get("control_policy") != (
        repeat.control_policy
    ):
        raise ValueError("preregistration changes frozen model controls")
    if fixed.get("adapter_version") != repeat.adapter_version:
        raise ValueError("preregistration changes frozen adapter version")
    if baseline_contract.get("retriever_id") != repeat.baseline.retriever_id:
        raise ValueError("preregistration changes baseline retriever")
    if candidate_contract.get("retriever_id") != repeat.candidate.retriever_id:
        raise ValueError("preregistration changes candidate retriever")
    if baseline_contract.get("result_count") != repeat.baseline_max_search_results:
        raise ValueError("preregistration changes baseline result count")
    if candidate_contract.get("result_count") != repeat.candidate_max_search_results:
        raise ValueError("preregistration changes candidate result count")
    candidate_hash = normalized_text_file_sha256(candidate_manifest_path)
    if (
        candidate_contract.get("manifest_sha256") != candidate_hash
        or repeat.candidate.retriever_manifest_sha256 != candidate_hash
    ):
        raise ValueError("candidate manifest differs from frozen contract")
    if fixed.get("target_manifest_sha256") != normalized_text_file_sha256(
        target_manifest_path
    ):
        raise ValueError("target manifest differs from frozen contract")
    if fresh_queries.get("normalized_query_artifact_sha256") != (
        repeat.development_queries_sha256
    ):
        raise ValueError("repeat uses another development query slice")
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
        execution=execution_result,
        judge_comparison_path=judge_comparison_path,
    )

    strict_delta = _delta(
        repeat.candidate.strict_exact_percent.mean,
        repeat.baseline.strict_exact_percent.mean,
    )
    evidence_delta = _delta(
        repeat.candidate.evidence_recall_percent.mean,
        repeat.baseline.evidence_recall_percent.mean,
    )
    gold_delta = _delta(
        repeat.candidate.gold_recall_percent.mean,
        repeat.baseline.gold_recall_percent.mean,
    )
    judge_delta = _delta(
        judge.candidate.trial_accuracy_percent.mean,
        judge.baseline.trial_accuracy_percent.mean,
    )
    evidence_paired = next(
        row for row in repeat.paired_metrics if row.metric == "evidence_recall"
    )
    evidence_wins_minus_losses = (
        evidence_paired.candidate_wins - evidence_paired.baseline_wins
    )
    search_ratio = _ratio(
        repeat.candidate.search_calls_per_query.mean,
        repeat.baseline.search_calls_per_query.mean,
        "baseline search calls",
    )
    token_ratio = _ratio(
        repeat.candidate.total_tokens_per_query.mean,
        repeat.baseline.total_tokens_per_query.mean,
        "baseline total tokens",
    )
    cost_ratio = _ratio(
        repeat.candidate.total_cost_usd,
        repeat.baseline.total_cost_usd,
        "baseline provider cost",
    )
    latency_ratio = _ratio(
        repeat.candidate.latency_ms_per_query.mean,
        repeat.baseline.latency_ms_per_query.mean,
        "baseline latency",
    )
    combined_cost = round(
        repeat.baseline.total_cost_usd + repeat.candidate.total_cost_usd, 12
    )
    final_failures, overshoot = _summary_failure_totals(
        experiment=experiment, repository_root=repository_root
    )

    gate_values = [
        _gate("registration_status", repeat.registration_status, "eq", "pre_generation"),
        _gate("trial_count", repeat.trial_count, "ge", int(execution["trials"])),
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
            "candidate_to_baseline_search_call_ratio_min",
            search_ratio,
            "ge",
            float(gates_contract["minimum_candidate_to_baseline_search_call_ratio"]),
        ),
        _gate(
            "candidate_to_baseline_search_call_ratio_max",
            search_ratio,
            "le",
            float(gates_contract["maximum_candidate_to_baseline_search_call_ratio"]),
        ),
        _gate(
            "candidate_to_baseline_total_token_ratio",
            token_ratio,
            "le",
            float(gates_contract["maximum_candidate_to_baseline_total_token_ratio"]),
        ),
        _gate(
            "candidate_to_baseline_provider_cost_ratio",
            cost_ratio,
            "le",
            float(gates_contract["maximum_candidate_to_baseline_provider_cost_ratio"]),
        ),
        _gate(
            "combined_provider_cost",
            combined_cost,
            "le",
            float(execution["maximum_combined_provider_cost_usd"]),
        ),
        _gate(
            "final_provider_failures",
            final_failures,
            "le",
            int(gates_contract["maximum_final_provider_failures"]),
        ),
    ]
    next_action = choose_next_action(gate_values)

    profiles, failures = analyze_candidate_failures(
        repeat=repeat,
        judge=judge,
        repository_root=repository_root,
    )
    synthesis_loss = _analyze_synthesis_loss(
        repeat=repeat,
        judge=judge,
        repository_root=repository_root,
        failures=failures,
    )
    all_passed = all(row.passed for row in gate_values)
    artifact = EvidenceBandwidthDecision(
        created_at=(
            existing.created_at
            if existing is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        decision="promote" if all_passed else "reject",
        model=repeat.model,
        baseline_retriever_id=repeat.baseline.retriever_id,
        candidate_retriever_id=repeat.candidate.retriever_id,
        baseline_result_count=repeat.baseline_max_search_results,
        candidate_result_count=repeat.candidate_max_search_results,
        trial_count=repeat.trial_count,
        queries_per_trial=repeat.queries_per_trial,
        strict_exact_delta_pp=strict_delta,
        evidence_recall_delta_pp=evidence_delta,
        gold_recall_delta_pp=gold_delta,
        judge_accuracy_delta_pp=judge_delta,
        candidate_to_baseline_search_call_ratio=search_ratio,
        candidate_to_baseline_total_token_ratio=token_ratio,
        candidate_to_baseline_provider_cost_ratio=cost_ratio,
        candidate_to_baseline_latency_ratio=latency_ratio,
        combined_provider_cost_usd=combined_cost,
        gates=gate_values,
        failure_aggregate=failures,
        failure_profiles=profiles,
        synthesis_loss=synthesis_loss,
        next_action=next_action,
        sources={
            "preregistration": _source(preregistration_path, repository_root),
            "candidate_manifest": _source(candidate_manifest_path, repository_root),
            "repeat_experiment": _source(repeat_experiment_path, repository_root),
            "repeat_comparison": _source(repeat_comparison_path, repository_root),
            "target_manifest": _source(target_manifest_path, repository_root),
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
            "The synthesis-loss diagnostic is an error localization signal, not "
            "causal proof that context length alone caused the Judge regressions.",
            "Judge labels come from a calibrated persistent service matching the "
            "frozen grader/model contract, not the official evaluator process.",
            "Observed differences are harness effects with frozen model parameters, "
            "not model-capability improvements.",
        ),
    )
    if existing is not None:
        if artifact != existing:
            raise ValueError("existing evidence bandwidth decision changed with sources")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return artifact


def choose_next_action(
    gates: list[EvidenceBandwidthGate],
) -> str:
    gate_by_id = {gate.gate_id: gate for gate in gates}
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
    if not all(gate_by_id[gate_id].passed for gate_id in structural_ids):
        return "audit_incomplete_or_structurally_invalid_grid"
    if not gate_by_id["evidence_recall_delta"].passed or not gate_by_id[
        "evidence_query_wins_minus_losses"
    ].passed:
        return "diagnose_retrieval_bandwidth_from_saved_replays"
    if not gate_by_id["judge_accuracy_delta"].passed:
        return "diagnose_evidence_selection_and_synthesis_from_saved_runs"
    resource_ids = {
        "candidate_to_baseline_search_call_ratio_min",
        "candidate_to_baseline_search_call_ratio_max",
        "candidate_to_baseline_total_token_ratio",
        "candidate_to_baseline_provider_cost_ratio",
    }
    if not all(gate_by_id[gate_id].passed for gate_id in resource_ids):
        return "calibrate_adaptive_evidence_bandwidth_budget"
    return "promote_evidence_bandwidth_and_cluster_remaining_failures"


def _analyze_synthesis_loss(
    *,
    repeat: RepeatComparisonSummary,
    judge: RepeatDevelopmentJudgeComparison,
    repository_root: Path,
    failures: FailureAggregate,
) -> SynthesisLossDiagnostic:
    diagnostic_rows: dict[tuple[str, str, str], object] = {}
    for trial in repeat.trials:
        for variant in ("baseline", "candidate"):
            source = getattr(trial, f"{variant}_diagnostic")
            path = _resolve_repository_file(source.path, repository_root)
            payload = path.read_bytes()
            if sha256(payload).hexdigest() != source.sha256:
                raise ValueError("diagnostic hash changed after repeat aggregation")
            diagnostic = DiagnosticSummary.model_validate_json(payload)
            for row in diagnostic.rows:
                key = (trial.trial_id, row.query_id, variant)
                if key in diagnostic_rows:
                    raise ValueError("paired diagnostic rows are duplicated")
                diagnostic_rows[key] = row

    judge_rows = {
        (row.trial_id, row.query_id, row.variant): row for row in judge.observations
    }
    if set(diagnostic_rows) != set(judge_rows):
        raise ValueError("paired diagnostics and Judge observations differ")

    improvements = 0
    regressions = 0
    ties = 0
    relevant_regressions = 0
    advantage_regressions = 0
    advantage_improvements = 0
    baseline_categories = {
        "format": 0,
        "no_relevant": 0,
        "relevant_but_incorrect": 0,
    }
    for trial in repeat.trials:
        query_ids = sorted(
            key[1]
            for key in diagnostic_rows
            if key[0] == trial.trial_id and key[2] == "candidate"
        )
        for query_id in query_ids:
            candidate_key = (trial.trial_id, query_id, "candidate")
            baseline_key = (trial.trial_id, query_id, "baseline")
            candidate_judge = judge_rows[candidate_key]
            baseline_judge = judge_rows[baseline_key]
            candidate_diagnostic = diagnostic_rows[candidate_key]
            baseline_diagnostic = diagnostic_rows[baseline_key]
            candidate_recall = float(candidate_diagnostic.evidence_recall or 0.0)
            baseline_recall = float(baseline_diagnostic.evidence_recall or 0.0)
            candidate_has_relevant = max(
                candidate_recall,
                float(candidate_diagnostic.gold_recall or 0.0),
            ) > 0
            candidate_has_advantage = candidate_recall > baseline_recall
            if candidate_judge.correct and not baseline_judge.correct:
                improvements += 1
                advantage_improvements += int(candidate_has_advantage)
            elif baseline_judge.correct and not candidate_judge.correct:
                regressions += 1
                relevant_regressions += int(candidate_has_relevant)
                advantage_regressions += int(
                    candidate_has_relevant and candidate_has_advantage
                )
            else:
                ties += 1
            if not baseline_judge.correct:
                if (
                    not baseline_diagnostic.answer_schema_complete
                    or not baseline_diagnostic.exact_answer_extracted
                ):
                    baseline_categories["format"] += 1
                elif max(
                    baseline_recall,
                    float(baseline_diagnostic.gold_recall or 0.0),
                ) == 0:
                    baseline_categories["no_relevant"] += 1
                else:
                    baseline_categories["relevant_but_incorrect"] += 1

    return SynthesisLossDiagnostic(
        paired_comparisons=repeat.paired_query_observations,
        candidate_improvements=improvements,
        candidate_regressions=regressions,
        paired_ties=ties,
        regressions_with_candidate_relevant_evidence=relevant_regressions,
        regressions_with_candidate_evidence_advantage=advantage_regressions,
        improvements_with_candidate_evidence_advantage=advantage_improvements,
        baseline_incorrect=sum(baseline_categories.values()),
        baseline_format_failures=baseline_categories["format"],
        baseline_no_relevant_doc_retrieved=baseline_categories["no_relevant"],
        baseline_relevant_evidence_but_incorrect=baseline_categories[
            "relevant_but_incorrect"
        ],
        candidate_incorrect=failures.candidate_incorrect,
        candidate_format_failures=failures.format_failures,
        candidate_no_relevant_doc_retrieved=failures.no_relevant_doc_retrieved,
        candidate_relevant_evidence_but_incorrect=(
            failures.relevant_doc_retrieved_but_incorrect
        ),
    )


def _gate(
    gate_id: str,
    observed: str | float | int,
    operator: Literal["eq", "ge", "le"],
    threshold: str | float | int,
) -> EvidenceBandwidthGate:
    passed = (
        observed == threshold
        if operator == "eq"
        else observed >= threshold
        if operator == "ge"
        else observed <= threshold
    )
    return EvidenceBandwidthGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _delta(candidate: float, baseline: float) -> float:
    return round(candidate - baseline, 6)


def _ratio(candidate: float, baseline: float, label: str) -> float:
    if baseline <= 0:
        raise ValueError(f"{label} must be positive")
    return round(candidate / baseline, 12)


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
        registration.get("status") != "registered_pre_inference"
        or registration.get("metric_status")
        != "calibrated_development_diagnostic_not_official"
        or execution.get("status") != "succeeded"
        or execution.get("error") is not None
        or execution.get("registration_sha256")
        != sha256(registration_path.read_bytes()).hexdigest()
    ):
        raise ValueError("repeat Judge execution wrapper did not succeed")
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
        raise ValueError(f"evidence bandwidth preregistration lacks {field_name}")
    return value


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence bandwidth source must be a JSON object")
    return value


def _resolve_repository_file(value: str, repository_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence bandwidth source path must be repository-relative")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("evidence bandwidth runtime source must stay under runs/")
    if not resolved.is_file():
        raise ValueError(f"evidence bandwidth source is missing: {value}")
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
    raise ValueError("could not locate repository root for evidence bandwidth")


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("evidence bandwidth artifacts must stay under runs/")
