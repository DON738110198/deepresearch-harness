from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_decision import DecisionSource
from .browsecomp_evaluation import DiagnosticSummary
from .browsecomp_plus import load_pi_browsecomp_run
from .development_judge import DevelopmentJudgeResult
from .pi_browsecomp import PiSmokeSummary


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProgressiveDisclosureGate(StrictContract):
    gate_id: Literal[
        "succeeded",
        "failed",
        "budget_exhausted",
        "output_budget_overshoot",
        "judge_parse_failures",
        "judge_request_failures",
        "judge_correct",
        "total_token_ratio",
        "provider_cost_ratio",
        "generation_cost",
        "evidence_ingress_tokens",
        "successful_open_calls",
    ]
    observed: float | int
    operator: Literal["eq", "ge", "le"]
    threshold: float | int
    passed: bool


class ProgressiveDisclosureQueryTrace(StrictContract):
    query_id: str = Field(min_length=1)
    judge_correct: bool
    normalized_exact_match: bool
    evidence_recall: float | None = Field(default=None, ge=0, le=1)
    search_calls: int = Field(ge=0)
    open_attempts: int = Field(ge=0)
    successful_open_calls: int = Field(ge=0)
    evidence_ingress_tokens: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)


class ProgressiveDisclosureDecision(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-progressive-disclosure-fresh5-decision-v0"
    ] = "browsecomp-plus-progressive-disclosure-fresh5-decision-v0"
    created_at: str
    status: Literal[
        "development_decision_not_leaderboard"
    ] = "development_decision_not_leaderboard"
    judge_metric_status: Literal[
        "calibrated_development_diagnostic_not_official"
    ] = "calibrated_development_diagnostic_not_official"
    decision: Literal["promote", "reject"]
    candidate_layer: Literal[
        "progressive_disclosure_v0"
    ] = "progressive_disclosure_v0"
    query_count: int = Field(gt=0)
    baseline_judge_correct: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    judge_correct_delta: int
    normalized_exact_match: int = Field(ge=0)
    evidence_recall_percent: float | None = Field(default=None, ge=0, le=100)
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0)
    candidate_to_baseline_search_call_ratio: float = Field(ge=0)
    baseline_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(ge=0)
    candidate_to_baseline_total_token_ratio: float = Field(ge=0)
    baseline_cost_usd: float = Field(gt=0)
    candidate_cost_usd: float = Field(ge=0)
    candidate_to_baseline_provider_cost_ratio: float = Field(ge=0)
    candidate_evidence_ingress_tokens: int = Field(ge=0)
    candidate_open_attempts: int = Field(ge=0)
    candidate_successful_open_calls: int = Field(ge=0)
    gates: list[ProgressiveDisclosureGate] = Field(min_length=12, max_length=12)
    query_traces: list[ProgressiveDisclosureQueryTrace] = Field(min_length=1)
    failure_mode: Literal[
        "tool_loop_context_replay_without_answer_gain",
        "none",
    ]
    next_action: Literal[
        "preregister_paired_25_query_confirmation",
        "preregister_fresh_slice_tool_loop_governor",
    ]
    sources: dict[str, DecisionSource]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_gates_and_traces(self) -> "ProgressiveDisclosureDecision":
        expected_gate_ids = {
            "succeeded",
            "failed",
            "budget_exhausted",
            "output_budget_overshoot",
            "judge_parse_failures",
            "judge_request_failures",
            "judge_correct",
            "total_token_ratio",
            "provider_cost_ratio",
            "generation_cost",
            "evidence_ingress_tokens",
            "successful_open_calls",
        }
        if {gate.gate_id for gate in self.gates} != expected_gate_ids:
            raise ValueError("progressive disclosure gate set is incomplete")
        expected_decision = "promote" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected_decision:
            raise ValueError("progressive disclosure decision differs from its gates")
        if self.query_count != len(self.query_traces):
            raise ValueError("query trace count differs from query_count")
        if self.candidate_successful_open_calls != sum(
            row.successful_open_calls for row in self.query_traces
        ):
            raise ValueError("successful open total differs from query traces")
        if self.decision == "promote":
            if self.failure_mode != "none" or self.next_action != (
                "preregister_paired_25_query_confirmation"
            ):
                raise ValueError("promoted decision has an invalid route")
        elif self.failure_mode == "none" or self.next_action != (
            "preregister_fresh_slice_tool_loop_governor"
        ):
            raise ValueError("rejected decision must route to the diagnosed failure")
        expected_sources = {
            "preregistration",
            "candidate_summary",
            "candidate_diagnostic",
            "judge_execution_result",
            "judge_result",
            "baseline_summary",
        }
        if set(self.sources) != expected_sources:
            raise ValueError("progressive disclosure source set is incomplete")
        return self


def decide_progressive_disclosure_fresh5(
    *,
    preregistration_path: Path,
    candidate_summary_path: Path,
    candidate_diagnostic_path: Path,
    judge_execution_result_path: Path,
    judge_result_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> ProgressiveDisclosureDecision:
    if output_path.exists() and not validate_existing:
        raise ValueError("progressive disclosure decision already exists")

    preregistration = _load_object(preregistration_path)
    if preregistration.get("schema_version") != (
        "browsecomp-plus-progressive-disclosure-fresh5-v0"
    ):
        raise ValueError("unexpected progressive disclosure preregistration")
    if preregistration.get("status") != "registered_before_live_generation":
        raise ValueError("progressive disclosure experiment was not preregistered")

    candidate_summary = PiSmokeSummary.model_validate_json(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    diagnostic = DiagnosticSummary.model_validate_json(
        candidate_diagnostic_path.read_text(encoding="utf-8")
    )
    judge_result = DevelopmentJudgeResult.model_validate_json(
        judge_result_path.read_text(encoding="utf-8")
    )
    judge_execution = _load_object(judge_execution_result_path)
    if judge_execution.get("status") != "succeeded":
        raise ValueError("persistent Judge execution did not succeed")

    candidate_summary_sha = _file_sha256(candidate_summary_path)
    if diagnostic.source_summary_sha256 != candidate_summary_sha:
        raise ValueError("diagnostic targets a different candidate summary")
    if judge_result.source_summary_sha256 != candidate_summary_sha:
        raise ValueError("Judge result targets a different candidate summary")

    query_ids = [item.query_id for item in candidate_summary.items]
    if query_ids != preregistration["query_selection"]["query_ids"]:
        raise ValueError("candidate query order differs from preregistration")
    if {row.query_id for row in diagnostic.rows} != set(query_ids):
        raise ValueError("diagnostic query IDs differ from candidate")
    if {row.query_id for row in judge_result.observations} != set(query_ids):
        raise ValueError("Judge query IDs differ from candidate")

    baseline_registration = preregistration["stored_baseline"]
    baseline_summary_path = _resolve_repository_path(
        preregistration_path, baseline_registration["source_summary_path"]
    )
    if _file_sha256(baseline_summary_path) != (
        baseline_registration["source_summary_sha256"]
    ):
        raise ValueError("stored baseline summary hash changed")
    baseline_summary = PiSmokeSummary.model_validate_json(
        baseline_summary_path.read_text(encoding="utf-8")
    )
    baseline_items = [
        item for item in baseline_summary.items if item.query_id in set(query_ids)
    ]
    if {item.query_id for item in baseline_items} != set(query_ids):
        raise ValueError("stored baseline does not cover the candidate queries")
    baseline_search_calls = sum(item.search_calls for item in baseline_items)
    baseline_total_tokens = sum(item.total_tokens for item in baseline_items)
    baseline_cost_usd = sum(item.cost_usd for item in baseline_items)
    for field_name, actual in (
        ("selected_search_calls", baseline_search_calls),
        ("selected_total_tokens", baseline_total_tokens),
        ("selected_cost_usd", baseline_cost_usd),
    ):
        expected = baseline_registration[field_name]
        if abs(float(actual) - float(expected)) > 1e-10:
            raise ValueError(f"stored baseline {field_name} changed")

    run_by_query = {}
    for item in candidate_summary.items:
        run_path = _resolve_repository_path(candidate_summary_path, item.run_path)
        if _file_sha256(run_path) != item.run_sha256:
            raise ValueError(f"candidate run hash changed for query {item.query_id}")
        run_by_query[item.query_id] = load_pi_browsecomp_run(run_path)

    diagnostic_by_query = {row.query_id: row for row in diagnostic.rows}
    judge_by_query = {row.query_id: row for row in judge_result.observations}
    query_traces = []
    for item in candidate_summary.items:
        run = run_by_query[item.query_id]
        disclosure_state = run.disclosure_state
        query_traces.append(
            ProgressiveDisclosureQueryTrace(
                query_id=item.query_id,
                judge_correct=judge_by_query[item.query_id].correct is True,
                normalized_exact_match=(
                    diagnostic_by_query[item.query_id].normalized_exact_match
                ),
                evidence_recall=diagnostic_by_query[item.query_id].evidence_recall,
                search_calls=len(run.search_calls),
                open_attempts=len(run.evidence_open_calls),
                successful_open_calls=(
                    disclosure_state.successful_open_calls if disclosure_state else 0
                ),
                evidence_ingress_tokens=(
                    disclosure_state.cumulative_ingress_tokens if disclosure_state else 0
                ),
                model_requests=run.model_requests or 0,
                input_tokens=run.usage.input_tokens,
                cache_read_tokens=run.usage.cache_read_tokens,
                output_tokens=run.usage.output_tokens,
                total_tokens=run.usage.total_tokens,
                cost_usd=run.usage.cost_usd,
            )
        )

    acceptance = preregistration["acceptance"]
    token_ratio = candidate_summary.total_tokens / baseline_total_tokens
    cost_ratio = candidate_summary.total_cost_usd / baseline_cost_usd
    successful_open_calls = sum(row.successful_open_calls for row in query_traces)
    gates = [
        _gate("succeeded", candidate_summary.succeeded, "eq", acceptance["succeeded_must_equal"]),
        _gate("failed", candidate_summary.failed, "eq", acceptance["failed_must_equal"]),
        _gate("budget_exhausted", candidate_summary.budget_exhausted, "eq", acceptance["budget_exhausted_must_equal"]),
        _gate("output_budget_overshoot", candidate_summary.total_output_budget_overshoot_tokens, "eq", acceptance["output_budget_overshoot_tokens_must_equal"]),
        _gate("judge_parse_failures", judge_result.parse_failures, "eq", acceptance["judge_parse_failures_must_equal"]),
        _gate("judge_request_failures", judge_result.request_failures, "eq", acceptance["judge_request_failures_must_equal"]),
        _gate("judge_correct", judge_result.correct, "ge", acceptance["minimum_judge_correct"]),
        _gate("total_token_ratio", token_ratio, "le", acceptance["maximum_total_token_ratio_vs_stored_baseline"]),
        _gate("provider_cost_ratio", cost_ratio, "le", acceptance["maximum_cost_ratio_vs_stored_baseline"]),
        _gate("generation_cost", candidate_summary.total_cost_usd, "le", acceptance["maximum_generation_cost_usd"]),
        _gate("evidence_ingress_tokens", candidate_summary.total_evidence_ingress_tokens, "le", acceptance["maximum_evidence_ingress_tokens"]),
        _gate("successful_open_calls", successful_open_calls, "le", acceptance["maximum_successful_open_calls"]),
    ]
    decision_value = "promote" if all(gate.passed for gate in gates) else "reject"
    result = ProgressiveDisclosureDecision(
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=decision_value,
        query_count=candidate_summary.query_count,
        baseline_judge_correct=baseline_registration["selected_query_correct"],
        candidate_judge_correct=judge_result.correct,
        judge_correct_delta=(
            judge_result.correct - baseline_registration["selected_query_correct"]
        ),
        normalized_exact_match=diagnostic.normalized_exact_match,
        evidence_recall_percent=diagnostic.evidence_recall_percent,
        baseline_search_calls=baseline_search_calls,
        candidate_search_calls=candidate_summary.total_search_calls,
        candidate_to_baseline_search_call_ratio=round(
            candidate_summary.total_search_calls / baseline_search_calls, 12
        ),
        baseline_total_tokens=baseline_total_tokens,
        candidate_total_tokens=candidate_summary.total_tokens,
        candidate_to_baseline_total_token_ratio=round(token_ratio, 12),
        baseline_cost_usd=baseline_cost_usd,
        candidate_cost_usd=candidate_summary.total_cost_usd,
        candidate_to_baseline_provider_cost_ratio=round(cost_ratio, 12),
        candidate_evidence_ingress_tokens=(
            candidate_summary.total_evidence_ingress_tokens
        ),
        candidate_open_attempts=candidate_summary.total_evidence_open_calls,
        candidate_successful_open_calls=successful_open_calls,
        gates=gates,
        query_traces=query_traces,
        failure_mode=(
            "none"
            if decision_value == "promote"
            else "tool_loop_context_replay_without_answer_gain"
        ),
        next_action=(
            "preregister_paired_25_query_confirmation"
            if decision_value == "promote"
            else "preregister_fresh_slice_tool_loop_governor"
        ),
        sources={
            "preregistration": _source(preregistration_path),
            "candidate_summary": _source(candidate_summary_path),
            "candidate_diagnostic": _source(candidate_diagnostic_path),
            "judge_execution_result": _source(judge_execution_result_path),
            "judge_result": _source(judge_result_path),
            "baseline_summary": _source(baseline_summary_path),
        },
        claim_boundary=(
            "This five-query development decision is an engineering diagnostic. "
            "It is not official BrowseComp-Plus accuracy, a leaderboard result, or "
            "a model-capability improvement claim."
        ),
    )
    if output_path.exists():
        existing = ProgressiveDisclosureDecision.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if existing.model_dump(exclude={"created_at"}) != result.model_dump(
            exclude={"created_at"}
        ):
            raise ValueError("existing progressive disclosure decision changed")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def _gate(
    gate_id: str,
    observed: float | int,
    operator: Literal["eq", "ge", "le"],
    threshold: float | int,
) -> ProgressiveDisclosureGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return ProgressiveDisclosureGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _source(path: Path) -> DecisionSource:
    repository_root = _find_repository_root(path.resolve())
    return DecisionSource(
        path=path.resolve().relative_to(repository_root).as_posix(),
        sha256=_file_sha256(path),
    )


def _resolve_repository_path(anchor_path: Path, value: str) -> Path:
    repository_root = _find_repository_root(anchor_path.resolve())
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    resolved.relative_to(repository_root)
    return resolved


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _find_repository_root(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
            return candidate
    raise ValueError(f"could not locate repository root from {path}")
