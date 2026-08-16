from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import load_pi_browsecomp_run
from .evidence_span_oracle import ArtifactReference
from .pi_browsecomp import PiSmokeSummary
from .tool_health import SearchServiceUnavailable, require_search_service_health


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepeatExecutionAuditRegistration(StrictContract):
    schema_version: Literal["obligation-span-repeat-execution-audit-registration-v0"] = (
        "obligation-span-repeat-execution-audit-registration-v0"
    )
    status: Literal["registered_post_failure_for_integrity_audit"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    original_repeat_registration: ArtifactReference
    recovery_registration: ArtifactReference
    invalid_arm_summary: ArtifactReference
    invalid_arm_diagnostic: ArtifactReference
    failed_baseline_attempt_summary: ArtifactReference
    invalid_arm: Literal["trial-3-candidate"] = "trial-3-candidate"
    invalid_arm_run_root: str = Field(min_length=1)
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    expected_retriever_id: str = Field(min_length=1)
    expected_query_count: int = Field(gt=0)
    expected_total_search_calls: int = Field(gt=0)
    expected_transport_failure_calls: int = Field(gt=0)
    expected_invalid_arm_recorded_cost_usd: float = Field(gt=0)
    known_unobservable_provider_attempts: int = Field(ge=1)
    provider_attempt_policy_violations: int = Field(ge=1)
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def expected_failure_signature_is_possible(
        self,
    ) -> "RepeatExecutionAuditRegistration":
        if self.expected_transport_failure_calls > self.expected_total_search_calls:
            raise ValueError("transport failures cannot exceed total search calls")
        return self


class ExecutionIntegrityGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int | bool
    operator: Literal["eq"] = "eq"
    threshold: int | bool
    passed: bool


class RepeatExecutionAuditResult(StrictContract):
    schema_version: Literal["obligation-span-repeat-execution-audit-v0"] = (
        "obligation-span-repeat-execution-audit-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["execution_integrity_failure_not_effectiveness_result"] = (
        "execution_integrity_failure_not_effectiveness_result"
    )
    decision: Literal["reject_execution"] = "reject_execution"
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invalid_arm: Literal["trial-3-candidate"] = "trial-3-candidate"
    query_count: int = Field(gt=0)
    summary_reported_succeeded: int = Field(ge=0)
    raw_run_reported_succeeded: int = Field(ge=0)
    total_search_calls: int = Field(ge=0)
    transport_failure_calls: int = Field(ge=0)
    successful_search_calls: int = Field(ge=0)
    transport_failure_details: dict[str, int]
    recorded_invalid_arm_provider_cost_usd: float = Field(gt=0)
    known_unobservable_provider_attempts: int = Field(ge=1)
    provider_attempt_policy_violations: int = Field(ge=1)
    cost_observability: Literal[
        "recorded_lower_bound_with_unobservable_failed_attempts"
    ] = "recorded_lower_bound_with_unobservable_failed_attempts"
    search_service_health_at_audit: Literal["unavailable"] = "unavailable"
    search_service_health_error: str = Field(min_length=1)
    new_provider_calls_for_audit: Literal[0] = 0
    judge_calls_for_invalid_arm: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    gates: tuple[ExecutionIntegrityGate, ...] = Field(min_length=1)
    effectiveness_interpretation: Literal["invalid_arm_not_scored"] = (
        "invalid_arm_not_scored"
    )
    next_action: Literal[
        "fix_fail_closed_tool_health_then_reregister_repeats"
    ] = "fix_fail_closed_tool_health_then_reregister_repeats"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def rejection_matches_integrity_failure(self) -> "RepeatExecutionAuditResult":
        if all(gate.passed for gate in self.gates):
            raise ValueError("execution rejection requires at least one failed gate")
        if self.transport_failure_calls == 0:
            raise ValueError("execution rejection requires observed tool failures")
        if self.successful_search_calls != 0:
            raise ValueError("this audit is scoped to an all-search-failed arm")
        return self


def load_repeat_execution_audit_registration(
    path: Path,
) -> RepeatExecutionAuditRegistration:
    registration = RepeatExecutionAuditRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    artifacts = (
        registration.original_repeat_registration,
        registration.recovery_registration,
        registration.invalid_arm_summary,
        registration.invalid_arm_diagnostic,
        registration.failed_baseline_attempt_summary,
        *registration.frozen_artifacts,
    )
    for artifact in artifacts:
        artifact_path = (root / artifact.path).resolve()
        if not artifact_path.is_relative_to(root) or not artifact_path.is_file():
            raise ValueError(
                f"execution-audit artifact is missing or escapes root: {artifact.path}"
            )
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"execution-audit artifact hash changed: {artifact.path}")
    run_root = (root / registration.invalid_arm_run_root).resolve()
    if not run_root.is_relative_to((root / "runs").resolve()) or not run_root.is_dir():
        raise ValueError("invalid arm run root is missing or escapes ignored runs/")
    return registration


def run_repeat_execution_audit(
    *, registration_path: Path, output_path: Path
) -> RepeatExecutionAuditResult:
    registration = load_repeat_execution_audit_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("execution audit output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("execution audit output already exists")

    summary = PiSmokeSummary.model_validate_json(
        (root / registration.invalid_arm_summary.path).read_text(encoding="utf-8")
    )
    if summary.query_count != registration.expected_query_count:
        raise ValueError("invalid arm query count differs from audit registration")
    if abs(
        summary.total_cost_usd
        - registration.expected_invalid_arm_recorded_cost_usd
    ) > 1e-12:
        raise ValueError("invalid arm recorded cost differs from audit registration")

    runs = []
    for item in summary.items:
        if item.run_path is None or item.run_sha256 is None:
            raise ValueError(f"invalid arm item lacks a run artifact: {item.query_id}")
        run_path = (root / item.run_path).resolve()
        if not run_path.is_relative_to((root / "runs").resolve()):
            raise ValueError("invalid arm run path escapes ignored runs/")
        if _sha256_file(run_path) != item.run_sha256:
            raise ValueError(f"invalid arm run hash changed: {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        if run.query_id != item.query_id:
            raise ValueError("invalid arm run identity differs from summary")
        runs.append(run)

    calls = [call for run in runs for call in run.search_calls]
    failures = [call for call in calls if call.outcome == "error"]
    successes = [call for call in calls if call.outcome == "ok"]
    failure_details = Counter((call.detail or "unspecified") for call in failures)
    if len(calls) != registration.expected_total_search_calls:
        raise ValueError("search-call count differs from audit registration")
    if len(failures) != registration.expected_transport_failure_calls:
        raise ValueError("tool-failure count differs from audit registration")

    try:
        require_search_service_health(
            registration.search_url,
            expected_retriever_id=registration.expected_retriever_id,
        )
    except SearchServiceUnavailable as error:
        health_error = str(error)
    else:
        raise ValueError("candidate search service unexpectedly healthy at audit time")

    gates = (
        _gate("transport_failure_calls", len(failures), 0),
        _gate("successful_search_calls", len(successes), len(calls)),
        _gate("search_service_available_at_audit", False, True),
        _gate(
            "summary_succeeded_matches_usable_runs",
            summary.succeeded,
            0,
        ),
    )
    result = RepeatExecutionAuditResult(
        created_at=_utc_now(),
        registration_sha256=_sha256_file(registration_path),
        query_count=summary.query_count,
        summary_reported_succeeded=summary.succeeded,
        raw_run_reported_succeeded=sum(run.status == "succeeded" for run in runs),
        total_search_calls=len(calls),
        transport_failure_calls=len(failures),
        successful_search_calls=len(successes),
        transport_failure_details=dict(sorted(failure_details.items())),
        recorded_invalid_arm_provider_cost_usd=summary.total_cost_usd,
        known_unobservable_provider_attempts=(
            registration.known_unobservable_provider_attempts
        ),
        provider_attempt_policy_violations=(
            registration.provider_attempt_policy_violations
        ),
        search_service_health_error=health_error,
        gates=gates,
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _gate(
    gate_id: str, observed: int | bool, threshold: int | bool
) -> ExecutionIntegrityGate:
    return ExecutionIntegrityGate(
        gate_id=gate_id,
        observed=observed,
        threshold=threshold,
        passed=observed == threshold,
    )
