from __future__ import annotations

from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrameworkComparison(StrictContract):
    framework: str = Field(min_length=1)
    official_source_url: str = Field(min_length=1)
    inspected_on: date
    relevant_mechanisms: tuple[str, ...] = Field(min_length=1)
    current_gap: str = Field(min_length=1)
    disposition: Literal["borrow_boundary", "benchmark_later", "defer", "reject"]

    @model_validator(mode="after")
    def source_must_be_public_http(self) -> "FrameworkComparison":
        parsed = urlparse(self.official_source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("framework comparison requires an official HTTPS source")
        return self


class LoopBudget(StrictContract):
    maximum_development_queries: int = Field(gt=0)
    maximum_provider_cost_usd: float = Field(ge=0)
    maximum_search_calls_per_query: int = Field(ge=0)
    maximum_output_tokens_per_query: int = Field(ge=0)
    sealed_holdout_access: Literal["forbidden"] = "forbidden"


class ResearchLoopCheckpoint(StrictContract):
    schema_version: Literal["deepresearch-research-loop-checkpoint-v0"] = (
        "deepresearch-research-loop-checkpoint-v0"
    )
    loop_id: str = Field(min_length=1)
    status: Literal[
        "diagnosing",
        "comparing",
        "preregistering",
        "implementing",
        "offline_calibrating",
        "live_evaluating",
        "judging",
        "deciding",
        "closed",
    ]
    problem: str = Field(min_length=1)
    observed_evidence: list[ArtifactReference] = Field(min_length=1)
    simplest_baseline: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    changed_mechanisms: tuple[str, ...] = Field(min_length=1, max_length=1)
    anti_claim: str = Field(min_length=1)
    framework_comparisons: list[FrameworkComparison] = Field(min_length=3)
    frozen_controls: tuple[str, ...] = Field(min_length=1)
    budget: LoopBudget
    multi_agent_status: Literal["deferred", "eligible"] = "deferred"
    multi_agent_failure_triggers: tuple[
        Literal[
            "independent_branch_omission",
            "single_context_interference",
            "contradictory_evidence_unchecked",
            "serial_research_latency",
        ],
        ...,
    ] = ()
    multi_agent_comparison_contract: str | None = None
    measured_result: ArtifactReference | None = None
    decision: Literal["accept", "reject", "blocked"] | None = None
    next_action: str = Field(min_length=1)
    active_paid_calls: int = Field(default=0, ge=0)
    retained_services: tuple[str, ...] = ()
    task_owned_processes: Literal["named_and_retained", "stopped"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def loop_boundaries_are_consistent(self) -> "ResearchLoopCheckpoint":
        names = [item.framework.casefold() for item in self.framework_comparisons]
        if len(names) != len(set(names)):
            raise ValueError("framework comparisons must name distinct frameworks")
        sources = [item.official_source_url for item in self.framework_comparisons]
        if len(sources) != len(set(sources)):
            raise ValueError("framework comparisons must use distinct official sources")
        if self.multi_agent_status == "eligible":
            if not self.multi_agent_failure_triggers:
                raise ValueError("multi-agent eligibility requires an observed trigger")
            if not self.multi_agent_comparison_contract:
                raise ValueError("multi-agent eligibility requires a frozen comparison")
        elif self.multi_agent_failure_triggers or self.multi_agent_comparison_contract:
            raise ValueError("deferred multi-agent work cannot carry an active contract")
        if self.status == "closed":
            if self.measured_result is None or self.decision is None:
                raise ValueError("closed loops require a measured result and decision")
        elif self.decision is not None:
            raise ValueError("only a closed loop can record a final decision")
        return self

    def pause_audit(self) -> "PauseAudit":
        blockers: list[str] = []
        if self.status != "closed":
            blockers.append("loop_not_closed")
        if self.measured_result is None:
            blockers.append("measured_result_missing")
        if self.decision is None:
            blockers.append("decision_missing")
        if self.active_paid_calls:
            blockers.append("paid_calls_still_active")
        return PauseAudit(
            loop_id=self.loop_id,
            ready_to_pause=not blockers,
            blockers=tuple(blockers),
            next_action=self.next_action,
        )


class PauseAudit(StrictContract):
    loop_id: str = Field(min_length=1)
    ready_to_pause: bool
    blockers: tuple[str, ...]
    next_action: str = Field(min_length=1)


class RejectedLoopReference(ArtifactReference):
    loop_id: str = Field(min_length=1)
    role: Literal["same_cluster_selection", "prior_analogue"]


class FailureClusterRoute(StrictContract):
    schema_version: Literal["deepresearch-failure-cluster-route-v0"] = (
        "deepresearch-failure-cluster-route-v0"
    )
    cluster_id: str = Field(min_length=1)
    status: Literal["regression_only"]
    query_ids: tuple[str, ...] = Field(min_length=1)
    selection_attempt_limit: int = Field(ge=1)
    same_cluster_rejections: tuple[RejectedLoopReference, ...] = Field(min_length=1)
    prior_analogue_rejections: tuple[RejectedLoopReference, ...] = ()
    prohibited_same_cluster_actions: tuple[str, ...] = Field(min_length=1)
    permitted_next_action: Literal["broader_development_profile"]
    framework_comparisons: list[FrameworkComparison] = Field(min_length=3)
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def route_is_consistent(self) -> "FailureClusterRoute":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("failure-cluster query IDs must be unique")
        references = (*self.same_cluster_rejections, *self.prior_analogue_rejections)
        loop_ids = [reference.loop_id for reference in references]
        paths = [reference.path for reference in references]
        if len(loop_ids) != len(set(loop_ids)):
            raise ValueError("failure-cluster loop references must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("failure-cluster checkpoint paths must be unique")
        if any(
            reference.role != "same_cluster_selection"
            for reference in self.same_cluster_rejections
        ):
            raise ValueError("same-cluster references must use the matching role")
        if any(
            reference.role != "prior_analogue"
            for reference in self.prior_analogue_rejections
        ):
            raise ValueError("prior analogues must use the matching role")
        if len(self.same_cluster_rejections) < self.selection_attempt_limit:
            raise ValueError(
                "regression-only routing requires the registered rejection cap"
            )
        return self

    def audit(self) -> "FailureClusterAudit":
        return FailureClusterAudit(
            cluster_id=self.cluster_id,
            status=self.status,
            selection_allowed=False,
            same_cluster_rejections=len(self.same_cluster_rejections),
            prior_analogue_rejections=len(self.prior_analogue_rejections),
            next_action=self.permitted_next_action,
        )


class FailureClusterAudit(StrictContract):
    cluster_id: str = Field(min_length=1)
    status: Literal["regression_only"]
    selection_allowed: Literal[False]
    same_cluster_rejections: int = Field(ge=1)
    prior_analogue_rejections: int = Field(ge=0)
    next_action: Literal["broader_development_profile"]


def load_research_loop_checkpoint(path: Path) -> ResearchLoopCheckpoint:
    return ResearchLoopCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


def load_failure_cluster_route(path: Path) -> FailureClusterRoute:
    route = FailureClusterRoute.model_validate_json(path.read_text(encoding="utf-8"))
    root = _repository_root(path)
    for reference in (
        *route.same_cluster_rejections,
        *route.prior_analogue_rejections,
    ):
        checkpoint_path = (root / reference.path).resolve()
        if not checkpoint_path.is_relative_to(root) or not checkpoint_path.is_file():
            raise ValueError(
                f"failure-cluster checkpoint is missing or escapes root: {reference.path}"
            )
        if _sha256_file(checkpoint_path) != reference.sha256:
            raise ValueError(f"failure-cluster checkpoint hash changed: {reference.path}")
        checkpoint = load_research_loop_checkpoint(checkpoint_path)
        if checkpoint.loop_id != reference.loop_id:
            raise ValueError(f"failure-cluster loop ID changed: {reference.path}")
        if checkpoint.status != "closed" or checkpoint.decision != "reject":
            raise ValueError(f"failure-cluster loop is not a closed rejection: {reference.path}")
    return route


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("could not locate research-loop repository root")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
