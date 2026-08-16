from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DiagnosticSummary
from .browsecomp_plus import (
    PiBrowseCompRun,
    load_development_queries,
    load_pi_browsecomp_run,
)
from .development_judge import DevelopmentJudgeResult
from .evidence_span_oracle import ArtifactReference
from .pi_browsecomp import PiSmokeSummary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FreshVariant(StrictContract):
    name: Literal["baseline", "candidate"]
    adapter_version: Literal["pi-browsecomp-v10", "pi-browsecomp-v14"]
    adapter_runner: ArtifactReference
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    retriever_id: str = Field(min_length=1)


class FreshFixedContract(StrictContract):
    target_manifest: ArtifactReference
    query_partitions: ArtifactReference
    query_partitions_normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_artifact_normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Literal["deepseek-v4-flash"] = "deepseek-v4-flash"
    model_parameters: Literal["frozen API model; no training"] = (
        "frozen API model; no training"
    )
    thinking_level: Literal["high"] = "high"
    compilation_thinking_level: Literal["off"] = "off"
    control_policy: Literal["answer_reserve_nonthinking_v0"] = (
        "answer_reserve_nonthinking_v0"
    )
    system_prompt_policy: Literal["empty"] = "empty"
    retriever_manifest: ArtifactReference
    anchor_count: int = Field(gt=0)
    dense_lead_count: int = Field(gt=0)
    anchor_token_cap: int = Field(gt=0)
    dense_lead_token_cap: int = Field(gt=0)
    maximum_search_calls: int = Field(gt=0)
    maximum_open_calls: int = Field(gt=0)
    open_token_cap: int = Field(gt=0)
    total_evidence_ingress_token_budget: int = Field(gt=0)
    open_evidence_ingress_token_budget: int = Field(gt=0)
    max_search_results: int = Field(gt=0, le=20)
    maximum_output_tokens: int = Field(gt=0, le=10_000)
    maximum_iterations: int = Field(gt=0)
    provider_attempts_per_case: int = Field(ge=1, le=2)
    judge_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/v1$")
    judge_served_model: str = Field(min_length=1)
    judge_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    sealed_holdout_access: Literal["forbidden"] = "forbidden"


class ObligationSpanFreshAcceptance(StrictContract):
    query_count_must_equal: int = Field(gt=0)
    schema_complete_must_equal: int = Field(gt=0)
    generation_failures_must_equal: Literal[0] = 0
    output_budget_overshoot_must_equal: Literal[0] = 0
    judge_parse_failures_must_equal: Literal[0] = 0
    judge_request_failures_must_equal: Literal[0] = 0
    minimum_judge_correct_delta: int = Field(ge=1)
    minimum_normalized_exact_delta: int = Field(ge=0)
    minimum_evidence_recall_delta_pp: float
    minimum_candidate_successful_open_cases: int = Field(ge=1)
    maximum_total_token_ratio: float = Field(ge=1)
    maximum_provider_cost_ratio: float = Field(ge=1)
    maximum_combined_recorded_provider_cost_usd: float = Field(gt=0)


class FreshRecovery(StrictContract):
    reason: Literal["post_provider_trace_validation_failure"]
    original_registration: ArtifactReference
    failed_summary: ArtifactReference
    failed_query_ids: tuple[str, ...] = Field(min_length=1)
    known_unobservable_provider_attempts: int = Field(gt=0)
    maximum_additional_attempts_per_failed_query: Literal[1] = 1
    effectiveness_gate_already_failed: Literal[True] = True


class ObligationSpanFreshRegistration(StrictContract):
    schema_version: Literal["obligation-span-fresh-registration-v0"] = (
        "obligation-span-fresh-registration-v0"
    )
    status: Literal["registered_pre_generation"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    query_artifact: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    excluded_query_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    selection_prefix: str = Field(min_length=1)
    variants: tuple[FreshVariant, FreshVariant]
    fixed_contract: FreshFixedContract
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: ObligationSpanFreshAcceptance
    recovery: FreshRecovery | None = None
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def variants_and_cases_match(self) -> "ObligationSpanFreshRegistration":
        if [variant.name for variant in self.variants] != ["baseline", "candidate"]:
            raise ValueError("fresh variants must be ordered baseline then candidate")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("fresh query IDs must be unique")
        if self.acceptance.query_count_must_equal != len(self.query_ids):
            raise ValueError("fresh query-count gate differs from selected cases")
        if self.acceptance.schema_complete_must_equal != len(self.query_ids):
            raise ValueError("fresh schema gate differs from selected cases")
        expected_versions = {
            "baseline": "pi-browsecomp-v10",
            "candidate": "pi-browsecomp-v14",
        }
        if any(
            variant.adapter_version != expected_versions[variant.name]
            for variant in self.variants
        ):
            raise ValueError("fresh variants do not match the frozen adapter roles")
        if self.variants[0].search_url == self.variants[1].search_url:
            raise ValueError("fresh variants must use separately identified services")
        if self.recovery is None:
            if self.fixed_contract.provider_attempts_per_case != 1:
                raise ValueError("initial fresh registration permits one provider attempt")
        else:
            if self.fixed_contract.provider_attempts_per_case != 2:
                raise ValueError("fresh recovery must cap provider attempts at two")
            if not set(self.recovery.failed_query_ids).issubset(self.query_ids):
                raise ValueError("fresh recovery query IDs escape the registered slice")
            if self.recovery.known_unobservable_provider_attempts < len(
                self.recovery.failed_query_ids
            ):
                raise ValueError("fresh recovery undercounts unobservable attempts")
        return self


class FreshCaseResult(StrictContract):
    query_id: str = Field(min_length=1)
    baseline_judge_correct: bool
    candidate_judge_correct: bool
    paired_outcome: Literal["both_correct", "both_wrong", "improvement", "regression"]
    baseline_normalized_exact: bool
    candidate_normalized_exact: bool
    baseline_evidence_recall: float = Field(ge=0, le=1)
    candidate_evidence_recall: float = Field(ge=0, le=1)
    candidate_open_attempts: int = Field(ge=0)
    candidate_successful_opens: int = Field(ge=0)


class FreshGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class ObligationSpanFreshDecision(StrictContract):
    schema_version: Literal["obligation-span-fresh-decision-v0"] = (
        "obligation-span-fresh-decision-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["development_decision_not_official"] = (
        "development_decision_not_official"
    )
    decision: Literal["accept", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(gt=0)
    baseline_judge_correct: int = Field(ge=0)
    candidate_judge_correct: int = Field(ge=0)
    judge_correct_delta: int
    judge_accuracy_delta_pp: float
    paired_improvements: int = Field(ge=0)
    paired_regressions: int = Field(ge=0)
    baseline_normalized_exact: int = Field(ge=0)
    candidate_normalized_exact: int = Field(ge=0)
    normalized_exact_delta: int
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_evidence_recall_percent: float = Field(ge=0, le=100)
    evidence_recall_delta_pp: float
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0)
    candidate_open_attempts: int = Field(ge=0)
    candidate_successful_open_calls: int = Field(ge=0)
    candidate_successful_open_cases: int = Field(ge=0)
    baseline_total_tokens: int = Field(gt=0)
    candidate_total_tokens: int = Field(gt=0)
    total_token_ratio: float = Field(ge=0)
    baseline_cost_usd: float = Field(gt=0)
    candidate_cost_usd: float = Field(gt=0)
    provider_cost_ratio: float = Field(ge=0)
    combined_recorded_provider_cost_usd: float = Field(gt=0)
    known_unobservable_provider_attempts: int = Field(ge=0)
    cost_observability: Literal[
        "complete",
        "recorded_lower_bound_with_unobservable_failed_attempts",
    ]
    cases: tuple[FreshCaseResult, ...] = Field(min_length=1)
    gates: tuple[FreshGate, ...] = Field(min_length=1)
    sources: dict[str, ArtifactReference]
    next_action: Literal[
        "repeat_on_second_fresh_slice",
        "freeze_span_layer_and_rediagnose",
    ]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "ObligationSpanFreshDecision":
        expected = "accept" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected:
            raise ValueError("fresh decision differs from gates")
        if self.query_count != len(self.cases):
            raise ValueError("fresh case count differs from query_count")
        expected_next = (
            "repeat_on_second_fresh_slice"
            if self.decision == "accept"
            else "freeze_span_layer_and_rediagnose"
        )
        if self.next_action != expected_next:
            raise ValueError("fresh next action differs from decision")
        return self


def load_obligation_span_fresh_registration(
    path: Path,
) -> ObligationSpanFreshRegistration:
    registration = ObligationSpanFreshRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    artifacts = (
        registration.query_artifact,
        *registration.excluded_query_artifacts,
        *(variant.adapter_runner for variant in registration.variants),
        registration.fixed_contract.target_manifest,
        registration.fixed_contract.query_partitions,
        registration.fixed_contract.retriever_manifest,
        *registration.fixed_contract.judge_artifacts,
        *(
            (
                registration.recovery.original_registration,
                registration.recovery.failed_summary,
            )
            if registration.recovery is not None
            else ()
        ),
        *registration.frozen_artifacts,
    )
    for artifact in artifacts:
        artifact_path = (root / artifact.path).resolve()
        if not artifact_path.is_relative_to(root) or not artifact_path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")
    queries = load_development_queries(root / registration.query_artifact.path)
    if tuple(row.query_id for row in queries.queries) != registration.query_ids:
        raise ValueError("fresh query artifact differs from registered IDs")
    if (
        queries.target_manifest_sha256
        != registration.fixed_contract.target_manifest.sha256
    ):
        raise ValueError("fresh queries target a different corpus manifest")
    if (
        queries.query_partitions_sha256
        != registration.fixed_contract.query_partitions_normalized_sha256
    ):
        raise ValueError("fresh queries target different query partitions")
    excluded = {
        row.query_id
        for artifact in registration.excluded_query_artifacts
        for row in load_development_queries(root / artifact.path).queries
    }
    overlap = excluded.intersection(registration.query_ids)
    if overlap:
        raise ValueError(f"fresh query IDs were previously used: {sorted(overlap)}")
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
            raise ValueError("fresh recovery IDs differ from the frozen failed summary")
    return registration


def decide_obligation_span_fresh(
    *,
    registration_path: Path,
    baseline_summary_path: Path,
    baseline_diagnostic_path: Path,
    baseline_judge_path: Path,
    candidate_summary_path: Path,
    candidate_diagnostic_path: Path,
    candidate_judge_path: Path,
    output_path: Path,
) -> ObligationSpanFreshDecision:
    registration = load_obligation_span_fresh_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("fresh decision output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("fresh decision output already exists")
    baseline = _load_variant(
        registration,
        root=root,
        variant=registration.variants[0],
        summary_path=baseline_summary_path,
        diagnostic_path=baseline_diagnostic_path,
        judge_path=baseline_judge_path,
    )
    candidate = _load_variant(
        registration,
        root=root,
        variant=registration.variants[1],
        summary_path=candidate_summary_path,
        diagnostic_path=candidate_diagnostic_path,
        judge_path=candidate_judge_path,
    )
    baseline_summary, baseline_diagnostic, baseline_judge, baseline_runs = baseline
    candidate_summary, candidate_diagnostic, candidate_judge, candidate_runs = candidate
    baseline_diag = {row.query_id: row for row in baseline_diagnostic.rows}
    candidate_diag = {row.query_id: row for row in candidate_diagnostic.rows}
    baseline_labels = {row.query_id: bool(row.correct) for row in baseline_judge.observations}
    candidate_labels = {row.query_id: bool(row.correct) for row in candidate_judge.observations}
    cases: list[FreshCaseResult] = []
    for query_id in registration.query_ids:
        base_correct = baseline_labels[query_id]
        cand_correct = candidate_labels[query_id]
        outcome: Literal["both_correct", "both_wrong", "improvement", "regression"]
        if base_correct and cand_correct:
            outcome = "both_correct"
        elif not base_correct and not cand_correct:
            outcome = "both_wrong"
        elif cand_correct:
            outcome = "improvement"
        else:
            outcome = "regression"
        run = candidate_runs[query_id]
        successful_opens = sum(
            call.outcome == "ok"
            and call.result is not None
            and call.result.outcome == "opened"
            for call in run.evidence_open_calls
        )
        cases.append(
            FreshCaseResult(
                query_id=query_id,
                baseline_judge_correct=base_correct,
                candidate_judge_correct=cand_correct,
                paired_outcome=outcome,
                baseline_normalized_exact=baseline_diag[query_id].normalized_exact_match,
                candidate_normalized_exact=candidate_diag[query_id].normalized_exact_match,
                baseline_evidence_recall=baseline_diag[query_id].evidence_recall or 0.0,
                candidate_evidence_recall=candidate_diag[query_id].evidence_recall or 0.0,
                candidate_open_attempts=len(run.evidence_open_calls),
                candidate_successful_opens=successful_opens,
            )
        )

    query_count = len(cases)
    baseline_correct = baseline_judge.correct
    candidate_correct = candidate_judge.correct
    judge_delta = candidate_correct - baseline_correct
    exact_delta = (
        candidate_diagnostic.normalized_exact_match
        - baseline_diagnostic.normalized_exact_match
    )
    recall_delta = round(
        candidate_diagnostic.evidence_recall_percent
        - baseline_diagnostic.evidence_recall_percent,
        8,
    )
    successful_open_calls = sum(row.candidate_successful_opens for row in cases)
    successful_open_cases = sum(row.candidate_successful_opens > 0 for row in cases)
    token_ratio = candidate_summary.total_tokens / baseline_summary.total_tokens
    cost_ratio = candidate_summary.total_cost_usd / baseline_summary.total_cost_usd
    combined_recorded_cost = (
        baseline_summary.total_cost_usd + candidate_summary.total_cost_usd
    )
    unobservable_attempts = (
        registration.recovery.known_unobservable_provider_attempts
        if registration.recovery is not None
        else 0
    )
    acceptance = registration.acceptance
    generation_failures = (
        baseline_summary.failed
        + baseline_summary.budget_exhausted
        + candidate_summary.failed
        + candidate_summary.budget_exhausted
    )
    gates = (
        _gate("baseline_query_count", baseline_summary.query_count, "eq", acceptance.query_count_must_equal),
        _gate("candidate_query_count", candidate_summary.query_count, "eq", acceptance.query_count_must_equal),
        _gate("baseline_schema_complete", baseline_summary.schema_complete or 0, "eq", acceptance.schema_complete_must_equal),
        _gate("candidate_schema_complete", candidate_summary.schema_complete or 0, "eq", acceptance.schema_complete_must_equal),
        _gate("generation_failures", generation_failures, "eq", acceptance.generation_failures_must_equal),
        _gate("output_budget_overshoot", baseline_summary.total_output_budget_overshoot_tokens + candidate_summary.total_output_budget_overshoot_tokens, "eq", acceptance.output_budget_overshoot_must_equal),
        _gate("judge_parse_failures", baseline_judge.parse_failures + candidate_judge.parse_failures, "eq", acceptance.judge_parse_failures_must_equal),
        _gate("judge_request_failures", baseline_judge.request_failures + candidate_judge.request_failures, "eq", acceptance.judge_request_failures_must_equal),
        _gate("judge_correct_delta", judge_delta, "ge", acceptance.minimum_judge_correct_delta),
        _gate("normalized_exact_delta", exact_delta, "ge", acceptance.minimum_normalized_exact_delta),
        _gate("evidence_recall_delta_pp", recall_delta, "ge", acceptance.minimum_evidence_recall_delta_pp),
        _gate("candidate_successful_open_cases", successful_open_cases, "ge", acceptance.minimum_candidate_successful_open_cases),
        _gate("total_token_ratio", token_ratio, "le", acceptance.maximum_total_token_ratio),
        _gate("provider_cost_ratio", cost_ratio, "le", acceptance.maximum_provider_cost_ratio),
        _gate("combined_recorded_provider_cost_usd", combined_recorded_cost, "le", acceptance.maximum_combined_recorded_provider_cost_usd),
    )
    decision: Literal["accept", "reject"] = (
        "accept" if all(gate.passed for gate in gates) else "reject"
    )
    result = ObligationSpanFreshDecision(
        created_at=_utc_now(),
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=query_count,
        baseline_judge_correct=baseline_correct,
        candidate_judge_correct=candidate_correct,
        judge_correct_delta=judge_delta,
        judge_accuracy_delta_pp=round(judge_delta * 100 / query_count, 8),
        paired_improvements=sum(row.paired_outcome == "improvement" for row in cases),
        paired_regressions=sum(row.paired_outcome == "regression" for row in cases),
        baseline_normalized_exact=baseline_diagnostic.normalized_exact_match,
        candidate_normalized_exact=candidate_diagnostic.normalized_exact_match,
        normalized_exact_delta=exact_delta,
        baseline_evidence_recall_percent=baseline_diagnostic.evidence_recall_percent,
        candidate_evidence_recall_percent=candidate_diagnostic.evidence_recall_percent,
        evidence_recall_delta_pp=recall_delta,
        baseline_search_calls=baseline_summary.total_search_calls,
        candidate_search_calls=candidate_summary.total_search_calls,
        candidate_open_attempts=sum(row.candidate_open_attempts for row in cases),
        candidate_successful_open_calls=successful_open_calls,
        candidate_successful_open_cases=successful_open_cases,
        baseline_total_tokens=baseline_summary.total_tokens,
        candidate_total_tokens=candidate_summary.total_tokens,
        total_token_ratio=round(token_ratio, 12),
        baseline_cost_usd=baseline_summary.total_cost_usd,
        candidate_cost_usd=candidate_summary.total_cost_usd,
        provider_cost_ratio=round(cost_ratio, 12),
        combined_recorded_provider_cost_usd=combined_recorded_cost,
        known_unobservable_provider_attempts=unobservable_attempts,
        cost_observability=(
            "recorded_lower_bound_with_unobservable_failed_attempts"
            if unobservable_attempts
            else "complete"
        ),
        cases=tuple(cases),
        gates=gates,
        sources={
            "registration": _source(registration_path),
            "baseline_summary": _source(baseline_summary_path),
            "baseline_diagnostic": _source(baseline_diagnostic_path),
            "baseline_judge": _source(baseline_judge_path),
            "candidate_summary": _source(candidate_summary_path),
            "candidate_diagnostic": _source(candidate_diagnostic_path),
            "candidate_judge": _source(candidate_judge_path),
        },
        next_action=(
            "repeat_on_second_fresh_slice"
            if decision == "accept"
            else "freeze_span_layer_and_rediagnose"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_variant(
    registration: ObligationSpanFreshRegistration,
    *,
    root: Path,
    variant: FreshVariant,
    summary_path: Path,
    diagnostic_path: Path,
    judge_path: Path,
) -> tuple[
    PiSmokeSummary,
    DiagnosticSummary,
    DevelopmentJudgeResult,
    dict[str, PiBrowseCompRun],
]:
    summary = PiSmokeSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    diagnostic = DiagnosticSummary.model_validate_json(diagnostic_path.read_text(encoding="utf-8"))
    judge = DevelopmentJudgeResult.model_validate_json(judge_path.read_text(encoding="utf-8"))
    summary_sha = _sha256_file(summary_path)
    if diagnostic.source_summary_sha256 != summary_sha:
        raise ValueError("diagnostic targets a different summary")
    if judge.source_summary_sha256 != summary_sha:
        raise ValueError("Judge result targets a different summary")
    queries = load_development_queries(root / registration.query_artifact.path)
    fixed = registration.fixed_contract
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
            raise ValueError(
                f"{variant.name} summary changed frozen {field_name}: "
                f"{getattr(summary, field_name)!r} != {expected!r}"
            )
    item_ids = tuple(item.query_id for item in summary.items)
    if item_ids != registration.query_ids:
        raise ValueError("variant query order differs from registration")
    if {row.query_id for row in diagnostic.rows} != set(registration.query_ids):
        raise ValueError("diagnostic query IDs differ from registration")
    if {row.query_id for row in judge.observations} != set(registration.query_ids):
        raise ValueError("Judge query IDs differ from registration")
    runs_root = (root / "runs").resolve()
    runs: dict[str, PiBrowseCompRun] = {}
    for item in summary.items:
        if item.attempt_count > fixed.provider_attempts_per_case:
            raise ValueError(f"{variant.name} exceeded the registered attempt cap")
        if item.run_path is None or item.run_sha256 is None:
            raise ValueError(f"successful variant item lacks run artifact: {item.query_id}")
        recorded = Path(item.run_path)
        if recorded.is_absolute():
            run_path = recorded
        elif recorded.parts and recorded.parts[0].casefold() == "runs":
            run_path = root / recorded
        else:
            run_path = summary_path.parent / recorded
        run_path = run_path.resolve()
        if not run_path.is_relative_to(runs_root):
            raise ValueError(f"variant run escapes ignored runs/: {item.query_id}")
        if _sha256_file(run_path) != item.run_sha256:
            raise ValueError(f"variant run hash changed: {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        if run.adapter_version != variant.adapter_version:
            raise ValueError(f"{variant.name} run used a different adapter")
        if run.query_id != item.query_id:
            raise ValueError(f"{variant.name} run query ID changed")
        expected_run = {
            "model": fixed.model,
            "thinking_level": fixed.thinking_level,
            "control_policy": fixed.control_policy,
            "max_search_results": fixed.max_search_results,
        }
        for field_name, expected in expected_run.items():
            if getattr(run, field_name) != expected:
                raise ValueError(
                    f"{variant.name} run changed frozen {field_name}: {item.query_id}"
                )
        expected_compilation_level = (
            fixed.compilation_thinking_level
            if run.answer_compiler_invoked
            else fixed.thinking_level
        )
        if run.compilation_thinking_level != expected_compilation_level:
            raise ValueError(
                f"{variant.name} run changed compilation policy: {item.query_id}"
            )
        runs[item.query_id] = run
    return summary, diagnostic, judge, runs


def _gate(
    gate_id: str,
    observed: int | float,
    operator: Literal["eq", "ge", "le"],
    threshold: int | float,
) -> FreshGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return FreshGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _source(path: Path) -> ArtifactReference:
    return ArtifactReference(path=path.as_posix(), sha256=_sha256_file(path))
