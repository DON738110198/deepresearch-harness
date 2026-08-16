from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DiagnosticSummary
from .browsecomp_plus import PiBrowseCompRun, load_pi_browsecomp_run
from .development_judge import DevelopmentJudgeResult
from .pi_browsecomp import PiSmokeSummary


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: float | int
    operator: Literal["eq", "ge", "le"]
    threshold: float | int
    passed: bool

    @model_validator(mode="after")
    def passed_matches_observation(self) -> "ExperimentGate":
        expected = {
            "eq": self.observed == self.threshold,
            "ge": self.observed >= self.threshold,
            "le": self.observed <= self.threshold,
        }[self.operator]
        if self.passed != expected:
            raise ValueError("gate result differs from its observation")
        return self


class LiveCalibrationRow(StrictContract):
    query_id: str = Field(min_length=1)
    prior_outcome: Literal["candidate_regression", "candidate_improvement"]
    prior_judge_correct: bool
    audit_status: Literal[
        "resolved",
        "repair_requested",
        "parse_failure",
        "supported",
        "no_repair_trigger",
        "open",
        "unscorable",
    ]
    repair_call_count: int = Field(ge=0, le=2)
    candidate_judge_correct: bool | None
    transition: Literal[
        "regression_repaired",
        "regression_still_wrong",
        "improvement_preserved",
        "improvement_lost",
        "unscored",
    ]
    search_calls: int = Field(ge=0, le=8)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)


class EvidenceDebtLiveCalibrationDecision(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-evidence-debt-live-calibration-decision-v0"
    ] = "browsecomp-plus-evidence-debt-live-calibration-decision-v0"
    created_at: str
    status: Literal["outcome_selected_calibration_not_effectiveness_evidence"] = (
        "outcome_selected_calibration_not_effectiveness_evidence"
    )
    decision: Literal["pass", "reject"]
    query_count: int = Field(gt=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    budget_exhausted: int = Field(ge=0)
    schema_complete: int = Field(ge=0)
    audit_trace_count: int = Field(ge=0)
    audit_parse_failures: int = Field(ge=0)
    repair_triggered_regressions: int = Field(ge=0)
    repair_triggered_improvements: int = Field(ge=0)
    repaired_regression_corrections: int = Field(ge=0)
    preserved_improvements: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    candidate_judge_accuracy_percent: float | None = Field(default=None, ge=0, le=100)
    candidate_exact_match_percent: float = Field(ge=0, le=100)
    candidate_evidence_recall_percent: float = Field(ge=0, le=100)
    total_search_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    recovery_status: Literal["none", "validation_only_failed_resume"] = "none"
    resumed_query_count: int = Field(default=0, ge=0)
    additional_provider_attempts: int = Field(default=0, ge=0)
    provider_cost_observability: Literal[
        "complete",
        "lower_bound_missing_pre_validation_usage",
    ] = "complete"
    rows: list[LiveCalibrationRow] = Field(min_length=1)
    gates: list[ExperimentGate] = Field(min_length=1)
    sources: dict[str, FileReference]
    next_action: Literal[
        "preregister_fresh_development_comparison",
        "diagnose_outcome_selected_calibration",
    ]
    claim_boundary: Literal[
        "Outcome-selected development calibration only; no effectiveness, leaderboard, or model-capability claim."
    ] = "Outcome-selected development calibration only; no effectiveness, leaderboard, or model-capability claim."

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "EvidenceDebtLiveCalibrationDecision":
        expected = "pass" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected:
            raise ValueError("live calibration decision differs from its gates")
        expected_action = (
            "preregister_fresh_development_comparison"
            if expected == "pass"
            else "diagnose_outcome_selected_calibration"
        )
        if self.next_action != expected_action:
            raise ValueError("live calibration next action differs from its decision")
        if self.query_count != len(self.rows):
            raise ValueError("live calibration row count differs")
        if self.recovery_status == "none":
            if self.resumed_query_count or self.additional_provider_attempts:
                raise ValueError("non-recovered decision cannot record resume attempts")
            if self.provider_cost_observability != "complete":
                raise ValueError("ordinary calibration cost must be fully observable")
        elif (
            self.resumed_query_count == 0
            or self.additional_provider_attempts < self.resumed_query_count
            or self.provider_cost_observability
            != "lower_bound_missing_pre_validation_usage"
        ):
            raise ValueError("validation recovery must expose its missing usage boundary")
        return self


class FreshComparisonRow(StrictContract):
    query_id: str = Field(min_length=1)
    baseline_judge_correct: bool | None
    candidate_judge_correct: bool | None
    paired_outcome: Literal[
        "candidate_improvement",
        "candidate_regression",
        "both_correct",
        "both_incorrect",
        "unscored",
    ]
    baseline_evidence_recall: float | None = Field(default=None, ge=0, le=1)
    candidate_evidence_recall: float | None = Field(default=None, ge=0, le=1)
    baseline_search_calls: int = Field(ge=0, le=8)
    candidate_search_calls: int = Field(ge=0, le=8)
    candidate_audit_status: Literal[
        "resolved",
        "repair_requested",
        "parse_failure",
        "supported",
        "no_repair_trigger",
        "open",
        "unscorable",
    ]
    candidate_repair_call_count: int = Field(ge=0, le=2)


class EvidenceDebtFreshComparisonDecision(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-evidence-debt-fresh-comparison-decision-v0"
    ] = "browsecomp-plus-evidence-debt-fresh-comparison-decision-v0"
    created_at: str
    status: Literal["fresh_development_diagnostic_not_official"] = (
        "fresh_development_diagnostic_not_official"
    )
    decision: Literal["promote", "reject"]
    query_count: int = Field(gt=0)
    baseline_judge_correct: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    baseline_judge_accuracy_percent: float | None = Field(default=None, ge=0, le=100)
    candidate_judge_accuracy_percent: float | None = Field(default=None, ge=0, le=100)
    judge_accuracy_delta_pp: float | None = None
    paired_improvements: int = Field(ge=0)
    paired_regressions: int = Field(ge=0)
    baseline_exact_match_percent: float = Field(ge=0, le=100)
    candidate_exact_match_percent: float = Field(ge=0, le=100)
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_evidence_recall_percent: float = Field(ge=0, le=100)
    evidence_recall_delta_pp: float
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0)
    search_call_ratio: float = Field(ge=0)
    baseline_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(ge=0)
    total_token_ratio: float = Field(ge=0)
    baseline_cost_usd: float = Field(gt=0)
    candidate_cost_usd: float = Field(ge=0)
    provider_cost_ratio: float = Field(ge=0)
    combined_generation_cost_usd: float = Field(ge=0)
    candidate_audit_trace_count: int = Field(ge=0)
    candidate_audit_parse_failures: int = Field(ge=0)
    candidate_repair_call_count: int = Field(ge=0)
    rows: list[FreshComparisonRow] = Field(min_length=1)
    gates: list[ExperimentGate] = Field(min_length=1)
    sources: dict[str, FileReference]
    next_action: Literal[
        "preregister_fresh25_confirmation",
        "diagnose_fresh_saved_traces",
    ]
    claim_boundary: Literal[
        "Fresh development comparison only; no sealed-holdout, leaderboard, or model-capability claim."
    ] = "Fresh development comparison only; no sealed-holdout, leaderboard, or model-capability claim."

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "EvidenceDebtFreshComparisonDecision":
        expected = "promote" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected:
            raise ValueError("fresh comparison decision differs from its gates")
        expected_action = (
            "preregister_fresh25_confirmation"
            if expected == "promote"
            else "diagnose_fresh_saved_traces"
        )
        if self.next_action != expected_action:
            raise ValueError("fresh comparison next action differs from its decision")
        if self.query_count != len(self.rows):
            raise ValueError("fresh comparison row count differs")
        return self


def decide_evidence_debt_live_calibration(
    *,
    registration_path: Path,
    candidate_summary_path: Path,
    candidate_diagnostic_path: Path,
    candidate_judge_execution_path: Path,
    candidate_judge_result_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> EvidenceDebtLiveCalibrationDecision:
    registration, root = _load_registration(
        registration_path,
        "browsecomp-plus-evidence-debt-live-calibration-registration-v0",
    )
    _validate_frozen_artifacts(registration, root)
    selected = registration["selection"]
    query_ids = [str(value) for value in selected["query_ids"]]
    _require_hash(root / str(selected["queries_path"]), str(selected["queries_sha256"]))
    if "offline_parity_path" in selected:
        _require_hash(
            root / str(selected["offline_parity_path"]),
            str(selected["offline_parity_sha256"]),
        )
    prior_judge_path = root / str(selected["prior_judge_result_path"])
    _require_hash(prior_judge_path, str(selected["prior_judge_result_sha256"]))
    prior_judge = DevelopmentJudgeResult.model_validate_json(
        prior_judge_path.read_text(encoding="utf-8")
    )
    prior_by_id = {row.query_id: row for row in prior_judge.observations}
    fixed_contract = registration.get("fixed_contract")
    if not isinstance(fixed_contract, dict):
        raise ValueError("live calibration registration lacks a fixed contract")
    expected_adapter_version = str(fixed_contract.get("adapter_version"))
    if expected_adapter_version not in {
        "pi-browsecomp-v11",
        "pi-browsecomp-v12",
        "pi-browsecomp-v13",
    }:
        raise ValueError("live calibration adapter is not registered")

    summary, diagnostic, judge, runs = _load_variant(
        root=root,
        summary_path=candidate_summary_path,
        diagnostic_path=candidate_diagnostic_path,
        judge_execution_path=candidate_judge_execution_path,
        judge_result_path=candidate_judge_result_path,
        query_ids=query_ids,
        expected_adapter_version=expected_adapter_version,
    )
    recovery = _validate_operational_recovery(
        registration=registration,
        root=root,
        summary=summary,
    )
    case_by_id = {str(row["query_id"]): row for row in selected["cases"]}
    if set(case_by_id) != set(query_ids):
        raise ValueError("live calibration cases differ from registered queries")
    judge_by_id = {row.query_id: row for row in judge.observations}
    rows: list[LiveCalibrationRow] = []
    for query_id in query_ids:
        case = case_by_id[query_id]
        prior = prior_by_id.get(query_id)
        if prior is None or prior.correct is not bool(case["prior_judge_correct"]):
            raise ValueError("registered prior outcome differs from frozen Judge result")
        run = runs[query_id]
        audit_status, repair_call_count, _ = _audit_summary(run)
        candidate_correct = judge_by_id[query_id].correct
        prior_outcome = str(case["prior_outcome"])
        transition = _calibration_transition(prior_outcome, candidate_correct)
        item = next(item for item in summary.items if item.query_id == query_id)
        rows.append(
            LiveCalibrationRow(
                query_id=query_id,
                prior_outcome=prior_outcome,
                prior_judge_correct=bool(prior.correct),
                audit_status=audit_status,
                repair_call_count=repair_call_count,
                candidate_judge_correct=candidate_correct,
                transition=transition,
                search_calls=item.search_calls,
                total_tokens=item.total_tokens,
                cost_usd=item.cost_usd,
            )
        )

    regressions = [row for row in rows if row.prior_outcome == "candidate_regression"]
    improvements = [row for row in rows if row.prior_outcome == "candidate_improvement"]
    audit_summaries = [_audit_summary(run) for run in runs.values()]
    audit_trace_count = len(audit_summaries)
    audit_parse_failures = sum(parse_failure for _, _, parse_failure in audit_summaries)
    repair_triggered_regressions = sum(row.repair_call_count > 0 for row in regressions)
    repair_triggered_improvements = sum(row.repair_call_count > 0 for row in improvements)
    repaired_regression_corrections = sum(
        row.repair_call_count > 0 and row.candidate_judge_correct is True
        for row in regressions
    )
    preserved_improvements = sum(
        row.candidate_judge_correct is True for row in improvements
    )
    acceptance = registration["acceptance"]
    maximum_search_calls = max(row.search_calls for row in rows)
    gates = [
        _gate("query_count", summary.query_count, "eq", len(query_ids)),
        _gate("succeeded", summary.succeeded, "eq", len(query_ids)),
        _gate("failed", summary.failed, "eq", 0),
        _gate("budget_exhausted", summary.budget_exhausted, "eq", 0),
        _gate("schema_complete", summary.schema_complete or 0, "eq", len(query_ids)),
        _gate("output_budget_overshoot", summary.total_output_budget_overshoot_tokens, "eq", 0),
        _gate("audit_trace_count", audit_trace_count, "eq", len(query_ids)),
        _gate("audit_parse_failures", audit_parse_failures, "eq", 0),
        _gate("maximum_search_calls", maximum_search_calls, "le", int(acceptance["maximum_search_calls_per_query"])),
        _gate("judge_parse_failures", judge.parse_failures, "eq", 0),
        _gate("judge_request_failures", judge.request_failures, "eq", 0),
        _gate("repair_triggered_regressions", repair_triggered_regressions, "ge", int(acceptance["minimum_repair_triggered_regressions"])),
        _gate("repair_triggered_improvements", repair_triggered_improvements, "le", int(acceptance["maximum_repair_triggered_improvements"])),
        _gate("repaired_regression_corrections", repaired_regression_corrections, "ge", int(acceptance["minimum_repaired_regression_corrections"])),
        _gate(
            "preserved_improvements",
            preserved_improvements,
            "eq",
            int(acceptance["preserved_improvements_must_equal"]),
        ),
        _gate("total_cost_usd", summary.total_cost_usd, "le", float(acceptance["maximum_generation_cost_usd"])),
    ]
    if recovery is not None:
        gates.extend(
            (
                _gate("resume_count", summary.resume_count, "eq", 1),
                _gate(
                    "resumed_query_count",
                    recovery["resumed_query_count"],
                    "eq",
                    len(recovery["failed_query_ids"]),
                ),
                _gate(
                    "additional_provider_attempts",
                    recovery["additional_provider_attempts"],
                    "eq",
                    len(recovery["failed_query_ids"]),
                ),
            )
        )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = EvidenceDebtLiveCalibrationDecision(
        created_at=_created_at(output_path, validate_existing),
        decision=decision,
        query_count=summary.query_count,
        succeeded=summary.succeeded,
        failed=summary.failed,
        budget_exhausted=summary.budget_exhausted,
        schema_complete=summary.schema_complete or 0,
        audit_trace_count=audit_trace_count,
        audit_parse_failures=audit_parse_failures,
        repair_triggered_regressions=repair_triggered_regressions,
        repair_triggered_improvements=repair_triggered_improvements,
        repaired_regression_corrections=repaired_regression_corrections,
        preserved_improvements=preserved_improvements,
        candidate_judge_correct=judge.correct,
        candidate_judge_accuracy_percent=judge.accuracy_percent,
        candidate_exact_match_percent=diagnostic.normalized_exact_match_percent,
        candidate_evidence_recall_percent=_required_recall(diagnostic),
        total_search_calls=summary.total_search_calls,
        total_tokens=summary.total_tokens,
        total_cost_usd=summary.total_cost_usd,
        recovery_status=(
            "none" if recovery is None else "validation_only_failed_resume"
        ),
        resumed_query_count=(0 if recovery is None else recovery["resumed_query_count"]),
        additional_provider_attempts=(
            0 if recovery is None else recovery["additional_provider_attempts"]
        ),
        provider_cost_observability=(
            "complete"
            if recovery is None
            else "lower_bound_missing_pre_validation_usage"
        ),
        rows=rows,
        gates=gates,
        sources=_sources(
            root,
            registration=registration_path,
            candidate_summary=candidate_summary_path,
            candidate_diagnostic=candidate_diagnostic_path,
            candidate_judge_execution=candidate_judge_execution_path,
            candidate_judge_result=candidate_judge_result_path,
            prior_judge_result=prior_judge_path,
        ),
        next_action=(
            "preregister_fresh_development_comparison"
            if decision == "pass"
            else "diagnose_outcome_selected_calibration"
        ),
    )
    _write_or_validate(result, output_path, validate_existing)
    return result


def decide_evidence_debt_fresh_comparison(
    *,
    registration_path: Path,
    baseline_summary_path: Path,
    baseline_diagnostic_path: Path,
    baseline_judge_execution_path: Path,
    baseline_judge_result_path: Path,
    candidate_summary_path: Path,
    candidate_diagnostic_path: Path,
    candidate_judge_execution_path: Path,
    candidate_judge_result_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> EvidenceDebtFreshComparisonDecision:
    registration, root = _load_registration(
        registration_path,
        "browsecomp-plus-evidence-debt-fresh-comparison-registration-v0",
    )
    _validate_frozen_artifacts(registration, root)
    selection = registration["selection"]
    query_ids = [str(value) for value in selection["query_ids"]]
    _require_hash(root / str(selection["queries_path"]), str(selection["queries_sha256"]))
    fixed_contract = registration.get("fixed_contract")
    if not isinstance(fixed_contract, dict):
        raise ValueError("fresh comparison registration lacks a fixed contract")
    baseline_adapter_version = str(
        fixed_contract.get("baseline_adapter_version", "pi-browsecomp-v10")
    )
    candidate_adapter_version = str(
        fixed_contract.get(
            "candidate_adapter_version", fixed_contract.get("adapter_version")
        )
    )
    if candidate_adapter_version not in {
        "pi-browsecomp-v11",
        "pi-browsecomp-v12",
        "pi-browsecomp-v13",
    }:
        raise ValueError("fresh comparison candidate adapter is not registered")
    baseline, baseline_diagnostic, baseline_judge, baseline_runs = _load_variant(
        root=root,
        summary_path=baseline_summary_path,
        diagnostic_path=baseline_diagnostic_path,
        judge_execution_path=baseline_judge_execution_path,
        judge_result_path=baseline_judge_result_path,
        query_ids=query_ids,
        expected_adapter_version=baseline_adapter_version,
    )
    candidate, candidate_diagnostic, candidate_judge, candidate_runs = _load_variant(
        root=root,
        summary_path=candidate_summary_path,
        diagnostic_path=candidate_diagnostic_path,
        judge_execution_path=candidate_judge_execution_path,
        judge_result_path=candidate_judge_result_path,
        query_ids=query_ids,
        expected_adapter_version=candidate_adapter_version,
    )
    baseline_judge_by_id = {row.query_id: row for row in baseline_judge.observations}
    candidate_judge_by_id = {row.query_id: row for row in candidate_judge.observations}
    baseline_diag_by_id = {row.query_id: row for row in baseline_diagnostic.rows}
    candidate_diag_by_id = {row.query_id: row for row in candidate_diagnostic.rows}
    baseline_item_by_id = {row.query_id: row for row in baseline.items}
    candidate_item_by_id = {row.query_id: row for row in candidate.items}
    rows: list[FreshComparisonRow] = []
    for query_id in query_ids:
        baseline_correct = baseline_judge_by_id[query_id].correct
        candidate_correct = candidate_judge_by_id[query_id].correct
        audit_status, repair_call_count, _ = _audit_summary(candidate_runs[query_id])
        rows.append(
            FreshComparisonRow(
                query_id=query_id,
                baseline_judge_correct=baseline_correct,
                candidate_judge_correct=candidate_correct,
                paired_outcome=_paired_outcome(baseline_correct, candidate_correct),
                baseline_evidence_recall=baseline_diag_by_id[query_id].evidence_recall,
                candidate_evidence_recall=candidate_diag_by_id[query_id].evidence_recall,
                baseline_search_calls=baseline_item_by_id[query_id].search_calls,
                candidate_search_calls=candidate_item_by_id[query_id].search_calls,
                candidate_audit_status=audit_status,
                candidate_repair_call_count=repair_call_count,
            )
        )

    paired_improvements = sum(row.paired_outcome == "candidate_improvement" for row in rows)
    paired_regressions = sum(row.paired_outcome == "candidate_regression" for row in rows)
    candidate_audit_summaries = [
        _audit_summary(run) for run in candidate_runs.values()
    ]
    candidate_audit_trace_count = len(candidate_audit_summaries)
    candidate_audit_parse_failures = sum(
        parse_failure for _, _, parse_failure in candidate_audit_summaries
    )
    candidate_repair_call_count = sum(
        repair_count for _, repair_count, _ in candidate_audit_summaries
    )
    baseline_recall = _required_recall(baseline_diagnostic)
    candidate_recall = _required_recall(candidate_diagnostic)
    search_ratio = _ratio(candidate.total_search_calls, baseline.total_search_calls)
    token_ratio = _ratio(candidate.total_tokens, baseline.total_tokens)
    cost_ratio = _ratio(candidate.total_cost_usd, baseline.total_cost_usd)
    judge_delta = _optional_delta(
        candidate_judge.accuracy_percent,
        baseline_judge.accuracy_percent,
    )
    acceptance = registration["acceptance"]
    maximum_candidate_searches = max(row.candidate_search_calls for row in rows)
    gates = [
        _gate("baseline_query_count", baseline.query_count, "eq", len(query_ids)),
        _gate("candidate_query_count", candidate.query_count, "eq", len(query_ids)),
        _gate("baseline_succeeded", baseline.succeeded, "eq", len(query_ids)),
        _gate("candidate_succeeded", candidate.succeeded, "eq", len(query_ids)),
        _gate("baseline_failed", baseline.failed, "eq", 0),
        _gate("candidate_failed", candidate.failed, "eq", 0),
        _gate("baseline_budget_exhausted", baseline.budget_exhausted, "eq", 0),
        _gate("candidate_budget_exhausted", candidate.budget_exhausted, "eq", 0),
        _gate("baseline_schema_complete", baseline.schema_complete or 0, "eq", len(query_ids)),
        _gate("candidate_schema_complete", candidate.schema_complete or 0, "eq", len(query_ids)),
        _gate("baseline_output_budget_overshoot", baseline.total_output_budget_overshoot_tokens, "eq", 0),
        _gate("candidate_output_budget_overshoot", candidate.total_output_budget_overshoot_tokens, "eq", 0),
        _gate("candidate_audit_trace_count", candidate_audit_trace_count, "eq", len(query_ids)),
        _gate("candidate_audit_parse_failures", candidate_audit_parse_failures, "eq", 0),
        _gate("maximum_candidate_search_calls", maximum_candidate_searches, "le", int(acceptance["maximum_search_calls_per_query"])),
        _gate("baseline_judge_parse_failures", baseline_judge.parse_failures, "eq", 0),
        _gate("candidate_judge_parse_failures", candidate_judge.parse_failures, "eq", 0),
        _gate("baseline_judge_request_failures", baseline_judge.request_failures, "eq", 0),
        _gate("candidate_judge_request_failures", candidate_judge.request_failures, "eq", 0),
        _gate("judge_accuracy_delta_pp", judge_delta if judge_delta is not None else -100.0, "ge", float(acceptance["minimum_judge_accuracy_delta_pp"])),
        _gate("paired_improvements", paired_improvements, "ge", int(acceptance["minimum_paired_improvements"])),
        _gate("paired_regressions", paired_regressions, "le", int(acceptance["maximum_paired_regressions"])),
        _gate("evidence_recall_delta_pp", candidate_recall - baseline_recall, "ge", float(acceptance["minimum_evidence_recall_delta_pp"])),
        _gate("search_call_ratio", search_ratio, "le", float(acceptance["maximum_search_call_ratio"])),
        _gate("total_token_ratio", token_ratio, "le", float(acceptance["maximum_total_token_ratio"])),
        _gate("provider_cost_ratio", cost_ratio, "le", float(acceptance["maximum_provider_cost_ratio"])),
        _gate("combined_generation_cost", baseline.total_cost_usd + candidate.total_cost_usd, "le", float(acceptance["maximum_combined_generation_cost_usd"])),
    ]
    decision: Literal["promote", "reject"] = (
        "promote" if all(gate.passed for gate in gates) else "reject"
    )
    result = EvidenceDebtFreshComparisonDecision(
        created_at=_created_at(output_path, validate_existing),
        decision=decision,
        query_count=len(query_ids),
        baseline_judge_correct=baseline_judge.correct,
        candidate_judge_correct=candidate_judge.correct,
        baseline_judge_accuracy_percent=baseline_judge.accuracy_percent,
        candidate_judge_accuracy_percent=candidate_judge.accuracy_percent,
        judge_accuracy_delta_pp=judge_delta,
        paired_improvements=paired_improvements,
        paired_regressions=paired_regressions,
        baseline_exact_match_percent=baseline_diagnostic.normalized_exact_match_percent,
        candidate_exact_match_percent=candidate_diagnostic.normalized_exact_match_percent,
        baseline_evidence_recall_percent=baseline_recall,
        candidate_evidence_recall_percent=candidate_recall,
        evidence_recall_delta_pp=round(candidate_recall - baseline_recall, 6),
        baseline_search_calls=baseline.total_search_calls,
        candidate_search_calls=candidate.total_search_calls,
        search_call_ratio=search_ratio,
        baseline_total_tokens=baseline.total_tokens,
        candidate_total_tokens=candidate.total_tokens,
        total_token_ratio=token_ratio,
        baseline_cost_usd=baseline.total_cost_usd,
        candidate_cost_usd=candidate.total_cost_usd,
        provider_cost_ratio=cost_ratio,
        combined_generation_cost_usd=baseline.total_cost_usd + candidate.total_cost_usd,
        candidate_audit_trace_count=candidate_audit_trace_count,
        candidate_audit_parse_failures=candidate_audit_parse_failures,
        candidate_repair_call_count=candidate_repair_call_count,
        rows=rows,
        gates=gates,
        sources=_sources(
            root,
            registration=registration_path,
            baseline_summary=baseline_summary_path,
            baseline_diagnostic=baseline_diagnostic_path,
            baseline_judge_execution=baseline_judge_execution_path,
            baseline_judge_result=baseline_judge_result_path,
            candidate_summary=candidate_summary_path,
            candidate_diagnostic=candidate_diagnostic_path,
            candidate_judge_execution=candidate_judge_execution_path,
            candidate_judge_result=candidate_judge_result_path,
        ),
        next_action=(
            "preregister_fresh25_confirmation"
            if decision == "promote"
            else "diagnose_fresh_saved_traces"
        ),
    )
    _write_or_validate(result, output_path, validate_existing)
    return result


def _load_variant(
    *,
    root: Path,
    summary_path: Path,
    diagnostic_path: Path,
    judge_execution_path: Path,
    judge_result_path: Path,
    query_ids: list[str],
    expected_adapter_version: str,
) -> tuple[PiSmokeSummary, DiagnosticSummary, DevelopmentJudgeResult, dict[str, PiBrowseCompRun]]:
    summary_bytes = summary_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    diagnostic = DiagnosticSummary.model_validate_json(
        diagnostic_path.read_text(encoding="utf-8")
    )
    judge = DevelopmentJudgeResult.model_validate_json(
        judge_result_path.read_text(encoding="utf-8")
    )
    if not judge_execution_path.is_file():
        raise ValueError("Judge execution artifact is missing")
    if [item.query_id for item in summary.items] != query_ids:
        raise ValueError("variant summary query order differs from registration")
    if {row.query_id for row in diagnostic.rows} != set(query_ids):
        raise ValueError("variant diagnostic query IDs differ from registration")
    if {row.query_id for row in judge.observations} != set(query_ids):
        raise ValueError("variant Judge query IDs differ from registration")
    summary_hash = sha256(summary_bytes).hexdigest()
    if diagnostic.source_summary_sha256 != summary_hash:
        raise ValueError("variant diagnostic targets another summary")
    if judge.source_summary_sha256 != summary_hash:
        raise ValueError("variant Judge targets another summary")
    runs: dict[str, PiBrowseCompRun] = {}
    for item in summary.items:
        if item.run_path is None or item.run_sha256 is None:
            raise ValueError("variant item is missing a frozen run")
        run_path = root / item.run_path
        _require_hash(run_path, item.run_sha256)
        run = load_pi_browsecomp_run(run_path)
        if run.adapter_version != expected_adapter_version:
            raise ValueError("variant adapter differs from registration")
        runs[item.query_id] = run
    return summary, diagnostic, judge, runs


def _audit_summary(run: PiBrowseCompRun) -> tuple[str, int, bool]:
    if run.adapter_version == "pi-browsecomp-v11":
        audit = run.evidence_debt_audit
        if audit is None:
            raise ValueError("v11 run is missing its evidence-debt audit")
        return audit.status, len(audit.repair_calls), audit.status == "parse_failure"
    if run.adapter_version in {"pi-browsecomp-v12", "pi-browsecomp-v13"}:
        audit = run.answer_first_audit
        if audit is None:
            raise ValueError("answer-first run is missing its deterministic audit")
        repair_count = 1 if audit.repair_status == "executed" else 0
        return audit.audit_status, repair_count, audit.audit_status == "unscorable"
    raise ValueError("candidate run does not expose a registered audit trace")


def _validate_operational_recovery(
    *,
    registration: dict[str, object],
    root: Path,
    summary: PiSmokeSummary,
) -> dict[str, object] | None:
    registration_status = registration.get("status")
    if registration_status == "registered_before_live_generation":
        if registration.get("operational_recovery") is not None:
            raise ValueError("ordinary registration cannot contain recovery metadata")
        return None
    recovery = registration.get("operational_recovery")
    if not isinstance(recovery, dict) or recovery.get("status") != (
        "registered_after_validation_failure_before_failed_only_resume"
    ):
        raise ValueError("failed-only resume lacks a registered recovery contract")

    for key in ("original_registration", "partial_summary"):
        reference = recovery.get(key)
        if not isinstance(reference, dict):
            raise ValueError(f"recovery {key} reference is missing")
        _require_hash(root / str(reference["path"]), str(reference["sha256"]))
    partial_reference = recovery["partial_summary"]
    partial = PiSmokeSummary.model_validate_json(
        (root / str(partial_reference["path"])).read_text(encoding="utf-8")
    )
    failed_query_ids = [str(value) for value in recovery.get("failed_query_ids", [])]
    if not failed_query_ids or len(failed_query_ids) != len(set(failed_query_ids)):
        raise ValueError("recovery failed query IDs must be non-empty and unique")
    if {
        item.query_id for item in partial.items if item.status == "failed"
    } != set(failed_query_ids):
        raise ValueError("recovery query IDs differ from the frozen partial summary")

    error_hashes: dict[str, str] = {}
    error_artifacts = recovery.get("failed_error_artifacts")
    if not isinstance(error_artifacts, list) or len(error_artifacts) != len(
        failed_query_ids
    ):
        raise ValueError("recovery error artifact count differs from failed queries")
    for raw in error_artifacts:
        if not isinstance(raw, dict):
            raise ValueError("recovery error artifact must be an object")
        query_id = str(raw["query_id"])
        _require_hash(root / str(raw["path"]), str(raw["sha256"]))
        error_hashes[query_id] = str(raw["sha256"])
    if set(error_hashes) != set(failed_query_ids):
        raise ValueError("recovery error artifacts differ from failed queries")

    resumed_items = [item for item in summary.items if item.attempt_count > 1]
    if {item.query_id for item in resumed_items} != set(failed_query_ids):
        raise ValueError("failed-only resume touched an unregistered query")
    for item in resumed_items:
        if item.attempt_count != 2 or len(item.prior_attempts) != 1:
            raise ValueError("recovery permits exactly one retry per failed query")
        if item.prior_attempts[0].error_sha256 != error_hashes[item.query_id]:
            raise ValueError("archived validation failure differs from registration")
    additional_provider_attempts = sum(
        item.attempt_count - 1 for item in summary.items
    )
    return {
        "failed_query_ids": failed_query_ids,
        "resumed_query_count": len(resumed_items),
        "additional_provider_attempts": additional_provider_attempts,
    }


def _load_registration(path: Path, schema: str) -> tuple[dict[str, object], Path]:
    registration = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registration, dict):
        raise ValueError("registration must be a JSON object")
    if registration.get("schema_version") != schema:
        raise ValueError("registration schema is not recognized")
    if registration.get("status") not in {
        "registered_before_live_generation",
        "registered_before_failed_only_resume",
    }:
        raise ValueError("experiment was not registered before live generation")
    return registration, _repository_root(path)


def _validate_frozen_artifacts(registration: dict[str, object], root: Path) -> None:
    artifacts = registration.get("frozen_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("registration requires frozen artifacts")
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise ValueError("frozen artifact must be an object")
        _require_hash(root / str(raw["path"]), str(raw["sha256"]))


def _calibration_transition(prior_outcome: str, candidate_correct: bool | None) -> str:
    if candidate_correct is None:
        return "unscored"
    if prior_outcome == "candidate_regression":
        return "regression_repaired" if candidate_correct else "regression_still_wrong"
    return "improvement_preserved" if candidate_correct else "improvement_lost"


def _paired_outcome(baseline: bool | None, candidate: bool | None) -> str:
    if baseline is None or candidate is None:
        return "unscored"
    if candidate and not baseline:
        return "candidate_improvement"
    if baseline and not candidate:
        return "candidate_regression"
    return "both_correct" if baseline else "both_incorrect"


def _gate(
    gate_id: str,
    observed: float | int,
    operator: Literal["eq", "ge", "le"],
    threshold: float | int,
) -> ExperimentGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return ExperimentGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _ratio(candidate: float | int, baseline: float | int) -> float:
    if baseline <= 0:
        raise ValueError("comparison baseline must be positive")
    return round(candidate / baseline, 12)


def _optional_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, 6)


def _required_recall(diagnostic: DiagnosticSummary) -> float:
    if diagnostic.evidence_recall_percent is None:
        raise ValueError("evidence recall is unavailable")
    return diagnostic.evidence_recall_percent


def _sources(root: Path, **paths: Path) -> dict[str, FileReference]:
    return {
        name: FileReference(
            path=str(path.resolve().relative_to(root.resolve())).replace("\\", "/"),
            sha256=_file_sha256(path),
        )
        for name, path in paths.items()
    }


def _created_at(output_path: Path, validate_existing: bool) -> str:
    if not validate_existing:
        if output_path.exists():
            raise ValueError("decision output already exists")
        return datetime.now(timezone.utc).isoformat()
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    return str(existing["created_at"])


def _write_or_validate(
    result: BaseModel,
    output_path: Path,
    validate_existing: bool,
) -> None:
    serialized = result.model_dump_json(indent=2) + "\n"
    if validate_existing:
        if output_path.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing decision differs from recomputation")
        return
    _atomic_write(output_path, serialized)


def _repository_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError("could not locate repository root")


def _require_hash(path: Path, expected: str) -> None:
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(f"hash mismatch for {path}: expected {expected}, observed {observed}")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
