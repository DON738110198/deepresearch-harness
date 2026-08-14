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


class ToolLoopGovernorGate(StrictContract):
    gate_id: Literal[
        "succeeded",
        "failed",
        "budget_exhausted",
        "output_budget_overshoot",
        "research_budget_trace_count",
        "maximum_search_calls_per_query",
        "total_search_calls",
        "judge_parse_failures",
        "judge_request_failures",
        "judge_correct",
        "evidence_recall_percent",
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


class ToolLoopGovernorQueryTrace(StrictContract):
    query_id: str = Field(min_length=1)
    judge_correct: bool
    evidence_recall: float | None = Field(default=None, ge=0, le=1)
    search_calls: int = Field(ge=0, le=8)
    search_budget_exhausted: bool
    blocked_search_calls: int = Field(ge=0)
    successful_open_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)


class ToolLoopGovernorDecision(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-tool-loop-governor-fresh5-decision-v0",
        "browsecomp-plus-query-aware-preview-fresh5-decision-v0",
    ] = "browsecomp-plus-tool-loop-governor-fresh5-decision-v0"
    created_at: str
    status: Literal[
        "development_decision_not_leaderboard"
    ] = "development_decision_not_leaderboard"
    judge_metric_status: Literal[
        "calibrated_development_diagnostic_not_official"
    ] = "calibrated_development_diagnostic_not_official"
    decision: Literal["promote", "reject"]
    candidate_layer: Literal[
        "progressive_disclosure_plus_tool_loop_governor_v0",
        "query_aware_preview_plus_tool_loop_governor_v0",
    ] = "progressive_disclosure_plus_tool_loop_governor_v0"
    query_count: int = Field(gt=0)
    baseline_judge_correct: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_evidence_recall_percent: float = Field(ge=0, le=100)
    evidence_recall_delta_pp: float
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0)
    baseline_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(ge=0)
    candidate_to_baseline_total_token_ratio: float = Field(ge=0)
    baseline_cost_usd: float = Field(gt=0)
    candidate_cost_usd: float = Field(ge=0)
    candidate_to_baseline_provider_cost_ratio: float = Field(ge=0)
    gates: list[ToolLoopGovernorGate] = Field(min_length=16, max_length=16)
    query_traces: list[ToolLoopGovernorQueryTrace] = Field(min_length=1)
    failure_mode: Literal[
        "answer_quality_no_gain_despite_retrieval_and_resource_gains",
        "none",
    ]
    next_action: Literal[
        "preregister_paired_25_query_confirmation",
        "calibrate_query_aware_dense_previews",
    ]
    sources: dict[str, DecisionSource]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "ToolLoopGovernorDecision":
        expected = {
            "succeeded",
            "failed",
            "budget_exhausted",
            "output_budget_overshoot",
            "research_budget_trace_count",
            "maximum_search_calls_per_query",
            "total_search_calls",
            "judge_parse_failures",
            "judge_request_failures",
            "judge_correct",
            "evidence_recall_percent",
            "total_token_ratio",
            "provider_cost_ratio",
            "generation_cost",
            "evidence_ingress_tokens",
            "successful_open_calls",
        }
        if {gate.gate_id for gate in self.gates} != expected:
            raise ValueError("tool-loop governor gate set is incomplete")
        expected_decision = "promote" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected_decision:
            raise ValueError("tool-loop governor decision differs from its gates")
        if self.query_count != len(self.query_traces):
            raise ValueError("tool-loop governor trace count differs")
        if self.decision == "promote":
            if self.failure_mode != "none" or self.next_action != (
                "preregister_paired_25_query_confirmation"
            ):
                raise ValueError("promoted tool-loop decision has an invalid route")
        elif self.failure_mode == "none" or self.next_action != (
            "calibrate_query_aware_dense_previews"
        ):
            raise ValueError("rejected tool-loop decision must route to preview work")
        return self


def decide_tool_loop_governor_fresh5(
    *,
    preregistration_path: Path,
    candidate_summary_path: Path,
    candidate_diagnostic_path: Path,
    judge_execution_result_path: Path,
    judge_result_path: Path,
    output_path: Path,
    validate_existing: bool = False,
    registration_schema_version: Literal[
        "browsecomp-plus-tool-loop-governor-fresh5-v0",
        "browsecomp-plus-query-aware-preview-fresh5-v0",
    ] = "browsecomp-plus-tool-loop-governor-fresh5-v0",
    decision_schema_version: Literal[
        "browsecomp-plus-tool-loop-governor-fresh5-decision-v0",
        "browsecomp-plus-query-aware-preview-fresh5-decision-v0",
    ] = "browsecomp-plus-tool-loop-governor-fresh5-decision-v0",
    candidate_layer: Literal[
        "progressive_disclosure_plus_tool_loop_governor_v0",
        "query_aware_preview_plus_tool_loop_governor_v0",
    ] = "progressive_disclosure_plus_tool_loop_governor_v0",
) -> ToolLoopGovernorDecision:
    if output_path.exists() and not validate_existing:
        raise ValueError("tool-loop governor decision already exists")
    registration = _load_object(preregistration_path)
    if registration.get("schema_version") != registration_schema_version or (
        registration.get("status") != "registered_before_live_generation"
    ):
        raise ValueError("budgeted retrieval candidate was not preregistered")

    summary = PiSmokeSummary.model_validate_json(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    diagnostic = DiagnosticSummary.model_validate_json(
        candidate_diagnostic_path.read_text(encoding="utf-8")
    )
    judge = DevelopmentJudgeResult.model_validate_json(
        judge_result_path.read_text(encoding="utf-8")
    )
    judge_execution = _load_object(judge_execution_result_path)
    if judge_execution.get("status") != "succeeded":
        raise ValueError("persistent Judge execution did not succeed")
    summary_sha = _file_sha256(candidate_summary_path)
    if diagnostic.source_summary_sha256 != summary_sha:
        raise ValueError("diagnostic targets a different candidate summary")
    if judge.source_summary_sha256 != summary_sha:
        raise ValueError("Judge targets a different candidate summary")

    registered_ids = registration["query_selection"]["query_ids"]
    if [item.query_id for item in summary.items] != registered_ids:
        raise ValueError("candidate query order differs from preregistration")
    if {row.query_id for row in diagnostic.rows} != set(registered_ids):
        raise ValueError("diagnostic query IDs differ")
    if {row.query_id for row in judge.observations} != set(registered_ids):
        raise ValueError("Judge query IDs differ")

    baseline = registration["stored_baseline"]
    repository_root = _repository_root(preregistration_path)
    baseline_summary_path = repository_root / baseline["source_summary_path"]
    baseline_diagnostic_path = repository_root / baseline["source_diagnostic_path"]
    baseline_judge_path = repository_root / baseline["judge_result_path"]
    for path, expected_hash in (
        (baseline_summary_path, baseline["source_summary_sha256"]),
        (baseline_diagnostic_path, baseline["source_diagnostic_sha256"]),
        (baseline_judge_path, baseline["judge_result_sha256"]),
    ):
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"stored baseline changed: {path}")
    baseline_summary = PiSmokeSummary.model_validate_json(
        baseline_summary_path.read_text(encoding="utf-8")
    )
    baseline_items = [
        item for item in baseline_summary.items if item.query_id in set(registered_ids)
    ]
    observed_baseline = {
        "selected_search_calls": sum(item.search_calls for item in baseline_items),
        "selected_total_tokens": sum(item.total_tokens for item in baseline_items),
        "selected_cost_usd": sum(item.cost_usd for item in baseline_items),
        "selected_latency_ms": sum(item.latency_ms for item in baseline_items),
    }
    for name, observed in observed_baseline.items():
        if abs(float(observed) - float(baseline[name])) > 1e-10:
            raise ValueError(f"stored baseline metric changed: {name}")

    diagnostic_by_query = {row.query_id: row for row in diagnostic.rows}
    judge_by_query = {row.query_id: row for row in judge.observations}
    traces = []
    research_trace_count = 0
    successful_open_calls = 0
    maximum_search_calls = 0
    for item in summary.items:
        run_path = (repository_root / item.run_path).resolve()
        if _file_sha256(run_path) != item.run_sha256:
            raise ValueError(f"candidate run changed: {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        if run.research_budget is None:
            raise ValueError(f"candidate lacks research budget: {item.query_id}")
        research_trace_count += 1
        maximum_search_calls = max(maximum_search_calls, len(run.search_calls))
        opened = run.disclosure_state.successful_open_calls if run.disclosure_state else 0
        successful_open_calls += opened
        traces.append(
            ToolLoopGovernorQueryTrace(
                query_id=item.query_id,
                judge_correct=judge_by_query[item.query_id].correct is True,
                evidence_recall=diagnostic_by_query[item.query_id].evidence_recall,
                search_calls=len(run.search_calls),
                search_budget_exhausted=run.research_budget.exhausted,
                blocked_search_calls=len(run.research_budget.blocked_search_calls),
                successful_open_calls=opened,
                total_tokens=run.usage.total_tokens,
                cost_usd=run.usage.cost_usd,
            )
        )

    acceptance = registration["acceptance"]
    token_ratio = summary.total_tokens / baseline["selected_total_tokens"]
    cost_ratio = summary.total_cost_usd / baseline["selected_cost_usd"]
    evidence_recall = diagnostic.evidence_recall_percent or 0.0
    gates = [
        _gate("succeeded", summary.succeeded, "eq", acceptance["succeeded_must_equal"]),
        _gate("failed", summary.failed, "eq", acceptance["failed_must_equal"]),
        _gate("budget_exhausted", summary.budget_exhausted, "eq", acceptance["budget_exhausted_must_equal"]),
        _gate("output_budget_overshoot", summary.total_output_budget_overshoot_tokens, "eq", acceptance["output_budget_overshoot_tokens_must_equal"]),
        _gate("research_budget_trace_count", research_trace_count, "eq", acceptance["research_budget_trace_count_must_equal"]),
        _gate("maximum_search_calls_per_query", maximum_search_calls, "le", acceptance["maximum_executed_search_calls_per_query"]),
        _gate("total_search_calls", summary.total_search_calls, "le", acceptance["maximum_total_search_calls"]),
        _gate("judge_parse_failures", judge.parse_failures, "eq", acceptance["judge_parse_failures_must_equal"]),
        _gate("judge_request_failures", judge.request_failures, "eq", acceptance["judge_request_failures_must_equal"]),
        _gate("judge_correct", judge.correct, "ge", acceptance["minimum_judge_correct"]),
        _gate("evidence_recall_percent", evidence_recall, "ge", acceptance["minimum_evidence_recall_percent"]),
        _gate("total_token_ratio", token_ratio, "le", acceptance["maximum_total_token_ratio_vs_stored_baseline"]),
        _gate("provider_cost_ratio", cost_ratio, "le", acceptance["maximum_cost_ratio_vs_stored_baseline"]),
        _gate("generation_cost", summary.total_cost_usd, "le", acceptance["maximum_generation_cost_usd"]),
        _gate("evidence_ingress_tokens", summary.total_evidence_ingress_tokens, "le", acceptance["maximum_evidence_ingress_tokens"]),
        _gate("successful_open_calls", successful_open_calls, "le", acceptance["maximum_successful_open_calls"]),
    ]
    decision_value = "promote" if all(gate.passed for gate in gates) else "reject"
    result = ToolLoopGovernorDecision(
        schema_version=decision_schema_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=decision_value,
        candidate_layer=candidate_layer,
        query_count=summary.query_count,
        baseline_judge_correct=baseline["selected_query_correct"],
        candidate_judge_correct=judge.correct,
        baseline_evidence_recall_percent=baseline["selected_evidence_recall_percent"],
        candidate_evidence_recall_percent=evidence_recall,
        evidence_recall_delta_pp=round(
            evidence_recall - baseline["selected_evidence_recall_percent"], 6
        ),
        baseline_search_calls=baseline["selected_search_calls"],
        candidate_search_calls=summary.total_search_calls,
        baseline_total_tokens=baseline["selected_total_tokens"],
        candidate_total_tokens=summary.total_tokens,
        candidate_to_baseline_total_token_ratio=round(token_ratio, 12),
        baseline_cost_usd=baseline["selected_cost_usd"],
        candidate_cost_usd=summary.total_cost_usd,
        candidate_to_baseline_provider_cost_ratio=round(cost_ratio, 12),
        gates=gates,
        query_traces=traces,
        failure_mode=(
            "none"
            if decision_value == "promote"
            else "answer_quality_no_gain_despite_retrieval_and_resource_gains"
        ),
        next_action=(
            "preregister_paired_25_query_confirmation"
            if decision_value == "promote"
            else "calibrate_query_aware_dense_previews"
        ),
        sources={
            "preregistration": _source(preregistration_path, repository_root),
            "candidate_summary": _source(candidate_summary_path, repository_root),
            "candidate_diagnostic": _source(candidate_diagnostic_path, repository_root),
            "judge_execution_result": _source(judge_execution_result_path, repository_root),
            "judge_result": _source(judge_result_path, repository_root),
            "baseline_summary": _source(baseline_summary_path, repository_root),
            "baseline_diagnostic": _source(baseline_diagnostic_path, repository_root),
            "baseline_judge_result": _source(baseline_judge_path, repository_root),
        },
        claim_boundary=(
            "This fresh five-query result is a calibrated development diagnostic. "
            "It is not official benchmark accuracy, leaderboard performance, or a "
            "model-capability improvement claim."
        ),
    )
    if output_path.exists():
        existing = ToolLoopGovernorDecision.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if existing.model_dump(exclude={"created_at"}) != result.model_dump(
            exclude={"created_at"}
        ):
            raise ValueError("existing tool-loop governor decision changed")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _gate(
    gate_id: str,
    observed: float | int,
    operator: Literal["eq", "ge", "le"],
    threshold: float | int,
) -> ToolLoopGovernorGate:
    passed = {"eq": observed == threshold, "ge": observed >= threshold, "le": observed <= threshold}[operator]
    return ToolLoopGovernorGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _repository_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise ValueError(f"repository root not found from {path}")


def _source(path: Path, root: Path) -> DecisionSource:
    resolved = path.resolve()
    return DecisionSource(
        path=resolved.relative_to(root).as_posix(),
        sha256=_file_sha256(resolved),
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
