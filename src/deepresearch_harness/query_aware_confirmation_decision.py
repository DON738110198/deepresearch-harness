from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_decision import DecisionSource
from .browsecomp_evaluation import DiagnosticSummary
from .browsecomp_plus import (
    PiBrowseCompRun,
    load_development_queries,
    load_pi_browsecomp_run,
    normalized_text_file_sha256,
)
from .development_judge import DevelopmentJudgeResult
from .pi_browsecomp import PiSmokeSummary


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


ConfirmationGateId = Literal[
    "baseline_query_count",
    "candidate_query_count",
    "baseline_succeeded",
    "candidate_succeeded",
    "baseline_failed",
    "candidate_failed",
    "baseline_budget_exhausted",
    "candidate_budget_exhausted",
    "baseline_output_budget_overshoot",
    "candidate_output_budget_overshoot",
    "baseline_schema_complete_percent",
    "candidate_schema_complete_percent",
    "candidate_research_budget_trace_count",
    "maximum_candidate_search_calls_per_query",
    "baseline_judge_parse_failures",
    "candidate_judge_parse_failures",
    "baseline_judge_request_failures",
    "candidate_judge_request_failures",
    "judge_accuracy_delta_pp",
    "evidence_recall_delta_pp",
    "search_call_ratio",
    "total_token_ratio",
    "provider_cost_ratio",
    "combined_generation_cost",
]


EXPECTED_GATE_IDS: frozenset[str] = frozenset(
    (
        "baseline_query_count",
        "candidate_query_count",
        "baseline_succeeded",
        "candidate_succeeded",
        "baseline_failed",
        "candidate_failed",
        "baseline_budget_exhausted",
        "candidate_budget_exhausted",
        "baseline_output_budget_overshoot",
        "candidate_output_budget_overshoot",
        "baseline_schema_complete_percent",
        "candidate_schema_complete_percent",
        "candidate_research_budget_trace_count",
        "maximum_candidate_search_calls_per_query",
        "baseline_judge_parse_failures",
        "candidate_judge_parse_failures",
        "baseline_judge_request_failures",
        "candidate_judge_request_failures",
        "judge_accuracy_delta_pp",
        "evidence_recall_delta_pp",
        "search_call_ratio",
        "total_token_ratio",
        "provider_cost_ratio",
        "combined_generation_cost",
    )
)


class QueryAwareConfirmationGate(StrictContract):
    gate_id: ConfirmationGateId
    observed: float | int
    operator: Literal["eq", "ge", "le"]
    threshold: float | int
    passed: bool


class QueryAwareConfirmationTrace(StrictContract):
    query_id: str = Field(min_length=1)
    paired_judge_outcome: Literal[
        "candidate_improvement",
        "candidate_regression",
        "both_correct",
        "both_incorrect",
        "unscored",
    ]
    baseline_judge_correct: bool | None
    candidate_judge_correct: bool | None
    baseline_exact_match: bool
    candidate_exact_match: bool
    baseline_evidence_recall: float | None = Field(default=None, ge=0, le=1)
    candidate_evidence_recall: float | None = Field(default=None, ge=0, le=1)
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0, le=8)
    candidate_search_limit_reached: bool
    candidate_blocked_search_calls: int = Field(ge=0)
    candidate_successful_open_calls: int = Field(ge=0)
    baseline_total_tokens: int = Field(ge=0)
    candidate_total_tokens: int = Field(ge=0)
    baseline_cost_usd: float = Field(ge=0)
    candidate_cost_usd: float = Field(ge=0)


class QueryAwareConfirmationDecision(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-query-aware-preview-confirmation-decision-v0"
    ] = "browsecomp-plus-query-aware-preview-confirmation-decision-v0"
    created_at: str
    status: Literal[
        "calibrated_development_decision_not_official"
    ] = "calibrated_development_decision_not_official"
    decision: Literal["promote", "reject"]
    candidate_layer: Literal[
        "query_aware_preview_plus_tool_loop_governor_v0"
    ] = "query_aware_preview_plus_tool_loop_governor_v0"
    query_count: int = Field(gt=0)
    baseline_judge_correct: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    baseline_judge_accuracy_percent: float = Field(ge=0, le=100)
    candidate_judge_accuracy_percent: float = Field(ge=0, le=100)
    judge_accuracy_delta_pp: float
    paired_judge_improvements: int = Field(ge=0)
    paired_judge_regressions: int = Field(ge=0)
    baseline_exact_match_percent: float = Field(ge=0, le=100)
    candidate_exact_match_percent: float = Field(ge=0, le=100)
    exact_match_delta_pp: float
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_evidence_recall_percent: float = Field(ge=0, le=100)
    evidence_recall_delta_pp: float
    baseline_gold_recall_percent: float = Field(ge=0, le=100)
    candidate_gold_recall_percent: float = Field(ge=0, le=100)
    gold_recall_delta_pp: float
    baseline_search_calls: int = Field(gt=0)
    candidate_search_calls: int = Field(ge=0)
    candidate_to_baseline_search_call_ratio: float = Field(ge=0)
    baseline_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(ge=0)
    candidate_to_baseline_total_token_ratio: float = Field(ge=0)
    baseline_cost_usd: float = Field(gt=0)
    candidate_cost_usd: float = Field(ge=0)
    candidate_to_baseline_provider_cost_ratio: float = Field(ge=0)
    combined_generation_cost_usd: float = Field(ge=0)
    gates: list[QueryAwareConfirmationGate] = Field(min_length=24, max_length=24)
    query_traces: list[QueryAwareConfirmationTrace] = Field(min_length=1)
    failure_mode: Literal[
        "none",
        "execution_contract_failure",
        "quality_gate_failure",
        "resource_gate_failure",
        "mixed_gate_failure",
    ]
    next_action: Literal[
        "preregister_three_trial_stability_confirmation",
        "diagnose_paired_saved_traces",
    ]
    sources: dict[str, DecisionSource]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_evidence(self) -> "QueryAwareConfirmationDecision":
        if {gate.gate_id for gate in self.gates} != EXPECTED_GATE_IDS:
            raise ValueError("query-aware confirmation gate set is incomplete")
        expected_decision = "promote" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected_decision:
            raise ValueError("query-aware confirmation decision differs from its gates")
        if self.query_count != len(self.query_traces):
            raise ValueError("query-aware confirmation trace count differs")
        improvements = sum(
            row.paired_judge_outcome == "candidate_improvement"
            for row in self.query_traces
        )
        regressions = sum(
            row.paired_judge_outcome == "candidate_regression"
            for row in self.query_traces
        )
        if (
            self.paired_judge_improvements != improvements
            or self.paired_judge_regressions != regressions
        ):
            raise ValueError("paired Judge outcome counts differ from traces")
        if self.decision == "promote":
            if self.failure_mode != "none" or self.next_action != (
                "preregister_three_trial_stability_confirmation"
            ):
                raise ValueError("promoted confirmation has an invalid route")
        elif self.failure_mode == "none" or self.next_action != (
            "diagnose_paired_saved_traces"
        ):
            raise ValueError("rejected confirmation must route to saved-trace diagnosis")
        return self


def decide_query_aware_confirmation(
    *,
    preregistration_path: Path,
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
) -> QueryAwareConfirmationDecision:
    if output_path.exists() and not validate_existing:
        raise ValueError("query-aware confirmation decision already exists")
    registration = _load_object(preregistration_path)
    if registration.get("schema_version") != (
        "browsecomp-plus-query-aware-preview-confirmation-v0"
    ) or registration.get("status") != "registered_before_live_generation":
        raise ValueError("query-aware confirmation was not preregistered")

    repository_root = _repository_root(preregistration_path)
    query_registration = registration["query_selection"]
    query_path = repository_root / query_registration["selected_path"]
    if _file_sha256(query_path) != query_registration["selected_sha256"]:
        raise ValueError("registered development query artifact changed")
    queries = load_development_queries(query_path)
    registered_ids = query_registration["query_ids"]
    if [row.query_id for row in queries.queries] != registered_ids:
        raise ValueError("development query order differs from preregistration")
    if queries.queries_sha256 != query_registration["queries_sha256"]:
        raise ValueError("development query content differs from preregistration")

    _validate_registered_code(registration=registration, repository_root=repository_root)

    baseline_summary = PiSmokeSummary.model_validate_json(
        baseline_summary_path.read_text(encoding="utf-8")
    )
    candidate_summary = PiSmokeSummary.model_validate_json(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    baseline_diagnostic = DiagnosticSummary.model_validate_json(
        baseline_diagnostic_path.read_text(encoding="utf-8")
    )
    candidate_diagnostic = DiagnosticSummary.model_validate_json(
        candidate_diagnostic_path.read_text(encoding="utf-8")
    )
    baseline_judge = DevelopmentJudgeResult.model_validate_json(
        baseline_judge_result_path.read_text(encoding="utf-8")
    )
    candidate_judge = DevelopmentJudgeResult.model_validate_json(
        candidate_judge_result_path.read_text(encoding="utf-8")
    )

    baseline_runs = _validate_variant(
        summary=baseline_summary,
        summary_path=baseline_summary_path,
        diagnostic=baseline_diagnostic,
        diagnostic_path=baseline_diagnostic_path,
        judge=baseline_judge,
        judge_result_path=baseline_judge_result_path,
        judge_execution_path=baseline_judge_execution_path,
        variant_registration=registration["baseline"],
        paired_registration=registration["paired_contract"],
        query_path=query_path,
        registered_ids=registered_ids,
        repository_root=repository_root,
        require_research_budget=False,
        judge_registration=registration["judge"],
    )
    candidate_runs = _validate_variant(
        summary=candidate_summary,
        summary_path=candidate_summary_path,
        diagnostic=candidate_diagnostic,
        diagnostic_path=candidate_diagnostic_path,
        judge=candidate_judge,
        judge_result_path=candidate_judge_result_path,
        judge_execution_path=candidate_judge_execution_path,
        variant_registration=registration["candidate"],
        paired_registration=registration["paired_contract"],
        query_path=query_path,
        registered_ids=registered_ids,
        repository_root=repository_root,
        require_research_budget=True,
        judge_registration=registration["judge"],
    )
    if baseline_summary.target_manifest_sha256 != candidate_summary.target_manifest_sha256:
        raise ValueError("paired variants target different benchmark manifests")

    baseline_schema_percent = _schema_percent(baseline_summary)
    candidate_schema_percent = _schema_percent(candidate_summary)
    baseline_judge_accuracy = _accuracy_or_zero(baseline_judge)
    candidate_judge_accuracy = _accuracy_or_zero(candidate_judge)
    baseline_evidence_recall = _required_metric(
        baseline_diagnostic.evidence_recall_percent, "baseline evidence recall"
    )
    candidate_evidence_recall = _required_metric(
        candidate_diagnostic.evidence_recall_percent, "candidate evidence recall"
    )
    baseline_gold_recall = _required_metric(
        baseline_diagnostic.gold_recall_percent, "baseline gold recall"
    )
    candidate_gold_recall = _required_metric(
        candidate_diagnostic.gold_recall_percent, "candidate gold recall"
    )
    search_ratio = _ratio(
        candidate_summary.total_search_calls,
        baseline_summary.total_search_calls,
        "baseline search calls",
    )
    token_ratio = _ratio(
        candidate_summary.total_tokens,
        baseline_summary.total_tokens,
        "baseline total tokens",
    )
    cost_ratio = _ratio(
        candidate_summary.total_cost_usd,
        baseline_summary.total_cost_usd,
        "baseline provider cost",
    )
    combined_cost = baseline_summary.total_cost_usd + candidate_summary.total_cost_usd
    candidate_research_trace_count = sum(
        run.research_budget is not None for run in candidate_runs.values()
    )
    maximum_candidate_search_calls = max(
        len(run.search_calls) for run in candidate_runs.values()
    )

    acceptance = registration["acceptance"]
    gates = [
        _gate("baseline_query_count", baseline_summary.query_count, "eq", acceptance["query_count_per_variant_must_equal"]),
        _gate("candidate_query_count", candidate_summary.query_count, "eq", acceptance["query_count_per_variant_must_equal"]),
        _gate("baseline_succeeded", baseline_summary.succeeded, "eq", acceptance["succeeded_per_variant_must_equal"]),
        _gate("candidate_succeeded", candidate_summary.succeeded, "eq", acceptance["succeeded_per_variant_must_equal"]),
        _gate("baseline_failed", baseline_summary.failed, "eq", acceptance["failed_per_variant_must_equal"]),
        _gate("candidate_failed", candidate_summary.failed, "eq", acceptance["failed_per_variant_must_equal"]),
        _gate("baseline_budget_exhausted", baseline_summary.budget_exhausted, "eq", acceptance["budget_exhausted_per_variant_must_equal"]),
        _gate("candidate_budget_exhausted", candidate_summary.budget_exhausted, "eq", acceptance["budget_exhausted_per_variant_must_equal"]),
        _gate("baseline_output_budget_overshoot", baseline_summary.total_output_budget_overshoot_tokens, "eq", acceptance["output_budget_overshoot_tokens_per_variant_must_equal"]),
        _gate("candidate_output_budget_overshoot", candidate_summary.total_output_budget_overshoot_tokens, "eq", acceptance["output_budget_overshoot_tokens_per_variant_must_equal"]),
        _gate("baseline_schema_complete_percent", baseline_schema_percent, "ge", acceptance["minimum_schema_complete_percent_per_variant"]),
        _gate("candidate_schema_complete_percent", candidate_schema_percent, "ge", acceptance["minimum_schema_complete_percent_per_variant"]),
        _gate("candidate_research_budget_trace_count", candidate_research_trace_count, "eq", acceptance["candidate_research_budget_trace_count_must_equal"]),
        _gate("maximum_candidate_search_calls_per_query", maximum_candidate_search_calls, "le", acceptance["maximum_candidate_search_calls_per_query"]),
        _gate("baseline_judge_parse_failures", baseline_judge.parse_failures, "eq", acceptance["judge_parse_failures_per_variant_must_equal"]),
        _gate("candidate_judge_parse_failures", candidate_judge.parse_failures, "eq", acceptance["judge_parse_failures_per_variant_must_equal"]),
        _gate("baseline_judge_request_failures", baseline_judge.request_failures, "eq", acceptance["judge_request_failures_per_variant_must_equal"]),
        _gate("candidate_judge_request_failures", candidate_judge.request_failures, "eq", acceptance["judge_request_failures_per_variant_must_equal"]),
        _gate("judge_accuracy_delta_pp", candidate_judge_accuracy - baseline_judge_accuracy, "ge", acceptance["minimum_candidate_minus_baseline_judge_accuracy_pp"]),
        _gate("evidence_recall_delta_pp", candidate_evidence_recall - baseline_evidence_recall, "ge", acceptance["minimum_candidate_minus_baseline_evidence_recall_pp"]),
        _gate("search_call_ratio", search_ratio, "le", acceptance["maximum_candidate_to_baseline_search_call_ratio"]),
        _gate("total_token_ratio", token_ratio, "le", acceptance["maximum_candidate_to_baseline_total_token_ratio"]),
        _gate("provider_cost_ratio", cost_ratio, "le", acceptance["maximum_candidate_to_baseline_provider_cost_ratio"]),
        _gate("combined_generation_cost", combined_cost, "le", acceptance["maximum_combined_generation_cost_usd"]),
    ]

    baseline_diagnostic_by_id = {row.query_id: row for row in baseline_diagnostic.rows}
    candidate_diagnostic_by_id = {row.query_id: row for row in candidate_diagnostic.rows}
    baseline_judge_by_id = {row.query_id: row for row in baseline_judge.observations}
    candidate_judge_by_id = {row.query_id: row for row in candidate_judge.observations}
    baseline_item_by_id = {row.query_id: row for row in baseline_summary.items}
    candidate_item_by_id = {row.query_id: row for row in candidate_summary.items}
    traces = []
    for query_id in registered_ids:
        baseline_label = baseline_judge_by_id[query_id].correct
        candidate_label = candidate_judge_by_id[query_id].correct
        candidate_run = candidate_runs[query_id]
        research_budget = candidate_run.research_budget
        if research_budget is None:
            raise ValueError(f"candidate lacks research budget: {query_id}")
        traces.append(
            QueryAwareConfirmationTrace(
                query_id=query_id,
                paired_judge_outcome=_paired_outcome(baseline_label, candidate_label),
                baseline_judge_correct=baseline_label,
                candidate_judge_correct=candidate_label,
                baseline_exact_match=baseline_diagnostic_by_id[query_id].normalized_exact_match,
                candidate_exact_match=candidate_diagnostic_by_id[query_id].normalized_exact_match,
                baseline_evidence_recall=baseline_diagnostic_by_id[query_id].evidence_recall,
                candidate_evidence_recall=candidate_diagnostic_by_id[query_id].evidence_recall,
                baseline_search_calls=len(baseline_runs[query_id].search_calls),
                candidate_search_calls=len(candidate_run.search_calls),
                candidate_search_limit_reached=research_budget.exhausted,
                candidate_blocked_search_calls=len(research_budget.blocked_search_calls),
                candidate_successful_open_calls=(
                    candidate_run.disclosure_state.successful_open_calls
                    if candidate_run.disclosure_state is not None
                    else 0
                ),
                baseline_total_tokens=baseline_item_by_id[query_id].total_tokens,
                candidate_total_tokens=candidate_item_by_id[query_id].total_tokens,
                baseline_cost_usd=baseline_item_by_id[query_id].cost_usd,
                candidate_cost_usd=candidate_item_by_id[query_id].cost_usd,
            )
        )

    decision_value = "promote" if all(gate.passed for gate in gates) else "reject"
    failed_gate_ids = {gate.gate_id for gate in gates if not gate.passed}
    result = QueryAwareConfirmationDecision(
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=decision_value,
        query_count=len(registered_ids),
        baseline_judge_correct=baseline_judge.correct,
        candidate_judge_correct=candidate_judge.correct,
        baseline_judge_accuracy_percent=baseline_judge_accuracy,
        candidate_judge_accuracy_percent=candidate_judge_accuracy,
        judge_accuracy_delta_pp=round(candidate_judge_accuracy - baseline_judge_accuracy, 6),
        paired_judge_improvements=sum(
            row.paired_judge_outcome == "candidate_improvement" for row in traces
        ),
        paired_judge_regressions=sum(
            row.paired_judge_outcome == "candidate_regression" for row in traces
        ),
        baseline_exact_match_percent=baseline_diagnostic.normalized_exact_match_percent,
        candidate_exact_match_percent=candidate_diagnostic.normalized_exact_match_percent,
        exact_match_delta_pp=round(
            candidate_diagnostic.normalized_exact_match_percent
            - baseline_diagnostic.normalized_exact_match_percent,
            6,
        ),
        baseline_evidence_recall_percent=baseline_evidence_recall,
        candidate_evidence_recall_percent=candidate_evidence_recall,
        evidence_recall_delta_pp=round(
            candidate_evidence_recall - baseline_evidence_recall, 6
        ),
        baseline_gold_recall_percent=baseline_gold_recall,
        candidate_gold_recall_percent=candidate_gold_recall,
        gold_recall_delta_pp=round(candidate_gold_recall - baseline_gold_recall, 6),
        baseline_search_calls=baseline_summary.total_search_calls,
        candidate_search_calls=candidate_summary.total_search_calls,
        candidate_to_baseline_search_call_ratio=round(search_ratio, 12),
        baseline_total_tokens=baseline_summary.total_tokens,
        candidate_total_tokens=candidate_summary.total_tokens,
        candidate_to_baseline_total_token_ratio=round(token_ratio, 12),
        baseline_cost_usd=baseline_summary.total_cost_usd,
        candidate_cost_usd=candidate_summary.total_cost_usd,
        candidate_to_baseline_provider_cost_ratio=round(cost_ratio, 12),
        combined_generation_cost_usd=round(combined_cost, 12),
        gates=gates,
        query_traces=traces,
        failure_mode=_failure_mode(failed_gate_ids),
        next_action=(
            "preregister_three_trial_stability_confirmation"
            if decision_value == "promote"
            else "diagnose_paired_saved_traces"
        ),
        sources={
            "preregistration": _source(preregistration_path, repository_root),
            "query_artifact": _source(query_path, repository_root),
            "baseline_summary": _source(baseline_summary_path, repository_root),
            "baseline_diagnostic": _source(baseline_diagnostic_path, repository_root),
            "baseline_judge_execution": _source(baseline_judge_execution_path, repository_root),
            "baseline_judge_result": _source(baseline_judge_result_path, repository_root),
            "candidate_summary": _source(candidate_summary_path, repository_root),
            "candidate_diagnostic": _source(candidate_diagnostic_path, repository_root),
            "candidate_judge_execution": _source(candidate_judge_execution_path, repository_root),
            "candidate_judge_result": _source(candidate_judge_result_path, repository_root),
        },
        claim_boundary=(
            "This is one fresh 25-query development confirmation with a calibrated "
            "service Judge. It is not official BrowseComp-Plus accuracy, a leaderboard "
            "result, or evidence that the frozen model itself improved."
        ),
    )
    if output_path.exists():
        existing = QueryAwareConfirmationDecision.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if existing.model_dump(exclude={"created_at"}) != result.model_dump(
            exclude={"created_at"}
        ):
            raise ValueError("existing query-aware confirmation decision changed")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _validate_registered_code(
    *, registration: dict[str, object], repository_root: Path
) -> None:
    baseline = registration["baseline"]
    candidate = registration["candidate"]
    files = (
        (
            repository_root / "integrations/pi-browsecomp/v8/contract.mjs",
            baseline["adapter_contract_sha256"],
        ),
        (
            repository_root / "integrations/pi-browsecomp/v8/runner.mjs",
            baseline["adapter_runner_sha256"],
        ),
        (
            repository_root / "src/deepresearch_harness/bm25_server.py",
            baseline["server_sha256"],
        ),
        (
            repository_root / "integrations/pi-browsecomp/v10/contract.mjs",
            candidate["adapter_contract_sha256"],
        ),
        (
            repository_root / "integrations/pi-browsecomp/v10/runner.mjs",
            candidate["adapter_runner_sha256"],
        ),
        (
            repository_root / "src/deepresearch_harness/evidence_preview.py",
            candidate["preview_module_sha256"],
        ),
        (
            repository_root
            / "src/deepresearch_harness/progressive_disclosure_server.py",
            candidate["server_sha256"],
        ),
        (
            repository_root / "benchmarks/browsecomp_plus_v0/retriever_candidates.json",
            candidate["retriever_manifest_sha256"],
        ),
        (
            repository_root / "benchmarks/browsecomp_plus_v0/persistent_bf16_judge_v0.json",
            registration["judge"]["manifest_sha256"],
        ),
    )
    for path, expected_hash in files:
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"registered code or contract changed: {path}")


def _validate_variant(
    *,
    summary: PiSmokeSummary,
    summary_path: Path,
    diagnostic: DiagnosticSummary,
    diagnostic_path: Path,
    judge: DevelopmentJudgeResult,
    judge_result_path: Path,
    judge_execution_path: Path,
    variant_registration: dict[str, object],
    paired_registration: dict[str, object],
    query_path: Path,
    registered_ids: list[str],
    repository_root: Path,
    require_research_budget: bool,
    judge_registration: dict[str, object],
) -> dict[str, PiBrowseCompRun]:
    summary_sha = _file_sha256(summary_path)
    if diagnostic.source_summary_sha256 != summary_sha:
        raise ValueError("diagnostic targets a different generation summary")
    if judge.source_summary_sha256 != summary_sha:
        raise ValueError("Judge targets a different generation summary")
    if [row.query_id for row in summary.items] != registered_ids:
        raise ValueError("generation query order differs from preregistration")
    expected_ids = set(registered_ids)
    if {row.query_id for row in diagnostic.rows} != expected_ids:
        raise ValueError("diagnostic query IDs differ from preregistration")
    if {row.query_id for row in judge.observations} != expected_ids:
        raise ValueError("Judge query IDs differ from preregistration")
    if summary.development_queries_sha256 != normalized_text_file_sha256(query_path):
        raise ValueError("generation used a different development query artifact")
    if (
        summary.model != paired_registration["model"]
        or summary.thinking_level != paired_registration["thinking_level"]
        or summary.system_prompt_policy != paired_registration["system_prompt_policy"]
        or summary.control_policy != paired_registration["control_policy"]
    ):
        raise ValueError("generation changed the paired model or control contract")
    if (
        summary.max_search_results != variant_registration["max_search_results"]
        or summary.retriever_id != variant_registration["retriever_id"]
    ):
        raise ValueError("generation changed the registered retrieval contract")
    if require_research_budget and summary.retriever_manifest_sha256 != (
        variant_registration["retriever_manifest_sha256"]
    ):
        raise ValueError("candidate retriever manifest differs from registration")
    if judge.judge_manifest_sha256 != judge_registration["manifest_sha256"]:
        raise ValueError("Judge result changes the registered Judge manifest")
    if judge.calibration_sha256 != judge_registration["calibration_sha256"]:
        raise ValueError("Judge result changes the accepted calibration")
    if (
        judge.judge_model != judge_registration["model"]
        or judge.served_model_name != judge_registration["served_model_name"]
        or judge.metric_status != judge_registration["metric_status"]
    ):
        raise ValueError("Judge result changes the registered evaluator contract")
    _validate_judge_execution(
        execution_path=judge_execution_path,
        judge_result_path=judge_result_path,
        summary_sha=summary_sha,
        judge_registration=judge_registration,
    )

    runs: dict[str, PiBrowseCompRun] = {}
    for item in summary.items:
        if item.run_sha256 is None:
            raise ValueError(f"generation run is not frozen: {item.query_id}")
        run_path = repository_root / item.run_path
        if _file_sha256(run_path) != item.run_sha256:
            raise ValueError(f"generation run changed: {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        if (
            run.query_id != item.query_id
            or run.adapter_version != variant_registration["adapter_version"]
            or run.model != paired_registration["model"]
            or run.thinking_level != paired_registration["thinking_level"]
            or run.control_policy != paired_registration["control_policy"]
            or run.system_prompt != ""
            or run.max_search_results != variant_registration["max_search_results"]
        ):
            raise ValueError(f"run changes the paired contract: {item.query_id}")
        if require_research_budget != (run.research_budget is not None):
            raise ValueError(f"run has the wrong research-budget contract: {item.query_id}")
        runs[item.query_id] = run
    return runs


def _validate_judge_execution(
    *,
    execution_path: Path,
    judge_result_path: Path,
    summary_sha: str,
    judge_registration: dict[str, object],
) -> None:
    execution = _load_object(execution_path)
    if execution.get("status") != "succeeded":
        raise ValueError("persistent Judge execution did not succeed")
    result = execution.get("judge_result")
    if not isinstance(result, dict) or result.get("sha256") != _file_sha256(
        judge_result_path
    ):
        raise ValueError("persistent Judge execution does not bind its result")
    registration_path = execution_path.parent / "execution_registration.json"
    judge_execution_registration = _load_object(registration_path)
    expected = {
        "judge_manifest_sha256": judge_registration["manifest_sha256"],
        "calibration_result_sha256": judge_registration["calibration_sha256"],
        "service_registration_sha256": judge_registration[
            "service_registration_sha256"
        ],
        "asset_verification_sha256": judge_registration[
            "asset_verification_sha256"
        ],
        "source_summary_sha256": summary_sha,
        "served_model_name": judge_registration["served_model_name"],
        "metric_status": judge_registration["metric_status"],
    }
    for field_name, expected_value in expected.items():
        if judge_execution_registration.get(field_name) != expected_value:
            raise ValueError(f"persistent Judge execution changed {field_name}")


def _schema_percent(summary: PiSmokeSummary) -> float:
    if summary.schema_complete is None:
        raise ValueError("generation summary lacks schema-complete accounting")
    return round(summary.schema_complete / summary.query_count * 100, 6)


def _accuracy_or_zero(result: DevelopmentJudgeResult) -> float:
    return result.accuracy_percent if result.accuracy_percent is not None else 0.0


def _required_metric(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"diagnostic lacks {name}")
    return value


def _ratio(numerator: float | int, denominator: float | int, name: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{name} must be positive")
    return float(numerator) / float(denominator)


def _paired_outcome(
    baseline: bool | None, candidate: bool | None
) -> Literal[
    "candidate_improvement",
    "candidate_regression",
    "both_correct",
    "both_incorrect",
    "unscored",
]:
    if baseline is None or candidate is None:
        return "unscored"
    if candidate and not baseline:
        return "candidate_improvement"
    if baseline and not candidate:
        return "candidate_regression"
    return "both_correct" if baseline else "both_incorrect"


def _failure_mode(
    failed_gate_ids: set[str],
) -> Literal[
    "none",
    "execution_contract_failure",
    "quality_gate_failure",
    "resource_gate_failure",
    "mixed_gate_failure",
]:
    if not failed_gate_ids:
        return "none"
    structural = EXPECTED_GATE_IDS - {
        "judge_accuracy_delta_pp",
        "evidence_recall_delta_pp",
        "search_call_ratio",
        "total_token_ratio",
        "provider_cost_ratio",
        "combined_generation_cost",
    }
    quality = {"judge_accuracy_delta_pp", "evidence_recall_delta_pp"}
    resource = {
        "search_call_ratio",
        "total_token_ratio",
        "provider_cost_ratio",
        "combined_generation_cost",
    }
    categories = sum(
        bool(failed_gate_ids & category)
        for category in (structural, quality, resource)
    )
    if categories > 1:
        return "mixed_gate_failure"
    if failed_gate_ids & structural:
        return "execution_contract_failure"
    if failed_gate_ids & quality:
        return "quality_gate_failure"
    return "resource_gate_failure"


def _gate(
    gate_id: ConfirmationGateId,
    observed: float | int,
    operator: Literal["eq", "ge", "le"],
    threshold: float | int,
) -> QueryAwareConfirmationGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return QueryAwareConfirmationGate(
        gate_id=gate_id,
        observed=round(observed, 12) if isinstance(observed, float) else observed,
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
