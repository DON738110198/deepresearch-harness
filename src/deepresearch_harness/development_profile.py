from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .research_loop import ArtifactReference, load_failure_cluster_route
from .tool_health import SearchServiceHealth, require_search_service_health


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PartitionArtifact(ArtifactReference):
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProfileArtifacts(StrictContract):
    failure_cluster_route: ArtifactReference
    target_manifest: ArtifactReference
    query_partitions: PartitionArtifact
    development_queries: ArtifactReference
    retriever_manifest: ArtifactReference


class FrozenProfilePolicy(StrictContract):
    policy_id: Literal["pi-v10-query-aware-progressive-disclosure-failclosed"]
    adapter_version: Literal["pi-browsecomp-v10"]
    adapter_contract: ArtifactReference
    adapter_runner: ArtifactReference
    preview_module: ArtifactReference
    retrieval_server: ArtifactReference
    operational_patch_only: Literal[True]


class ProfileExecution(StrictContract):
    model: Literal["deepseek-v4-flash"]
    thinking_level: Literal["high"]
    system_prompt_policy: Literal["empty"]
    control_policy: Literal["answer_reserve_nonthinking_v0"]
    query_partition: Literal["development"]
    query_count: Literal[175]
    concurrency: Literal[1]
    max_output_tokens_per_query: Literal[10000]
    max_iterations_per_query: Literal[100]
    maximum_search_calls_per_query: Literal[8]
    maximum_open_calls_per_query: Literal[8]
    max_search_results: Literal[20]
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    retriever_id: str = Field(min_length=1)
    provider_key_env: Literal["DEEPSEEK_API_KEY"]
    failed_only_resume_limit: int = Field(ge=0, le=2)
    budget_exhausted_retry: Literal["forbidden"]
    output_directory: str = Field(min_length=1)
    maximum_provider_cost_usd: float = Field(gt=0)


class ProfileEvaluation(StrictContract):
    judge_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/v1$")
    served_model_name: Literal["qwen3-32b-bf16-judge"]
    metric_status: Literal["calibrated_development_diagnostic_not_official"]
    sealed_holdout_access: Literal["forbidden"]


class ProfileAcceptance(StrictContract):
    succeeded_must_equal: Literal[175]
    failed_must_equal: Literal[0]
    budget_exhausted_must_equal: Literal[0]
    output_budget_overshoot_tokens_must_equal: Literal[0]
    search_and_open_transport_failures_must_equal: Literal[0]
    minimum_schema_complete: int = Field(ge=1, le=175)
    judge_parse_failures_must_equal: Literal[0]
    judge_request_failures_must_equal: Literal[0]
    minimum_scored_wrong_cases_for_taxonomy: int = Field(ge=1, le=175)
    accuracy_promotion_threshold: Literal["none_profile_only"]


class DevelopmentProfileRegistration(StrictContract):
    schema_version: Literal["browsecomp-plus-development-profile-registration-v0"]
    status: Literal["registered_before_provider_call"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    artifacts: ProfileArtifacts
    policy: FrozenProfilePolicy
    execution: ProfileExecution
    evaluation: ProfileEvaluation
    acceptance: ProfileAcceptance
    planned_metrics: tuple[str, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def registration_is_consistent(self) -> "DevelopmentProfileRegistration":
        if self.acceptance.minimum_schema_complete < 168:
            raise ValueError("development profile requires at least 96% schema completeness")
        if len(self.planned_metrics) != len(set(self.planned_metrics)):
            raise ValueError("development profile metrics must be unique")
        return self


class DevelopmentProfilePreflight(StrictContract):
    schema_version: Literal["browsecomp-plus-development-profile-preflight-v0"] = (
        "browsecomp-plus-development-profile-preflight-v0"
    )
    created_at: str = Field(min_length=1)
    registration: ArtifactReference
    route_status: Literal["regression_only"]
    query_count: Literal[175]
    query_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Literal["deepseek-v4-flash"]
    retriever_id: str = Field(min_length=1)
    health_url: str = Field(min_length=1)
    provider_key_env: Literal["DEEPSEEK_API_KEY"]
    provider_key_present: Literal[True]
    output_mode: Literal["new", "failed_only_resume"]
    provider_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False


HealthCheck = Callable[..., SearchServiceHealth]


def load_development_profile_registration(
    path: Path,
) -> DevelopmentProfileRegistration:
    registration = DevelopmentProfileRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = _repository_root(path)
    references = (
        registration.artifacts.failure_cluster_route,
        registration.artifacts.target_manifest,
        registration.artifacts.query_partitions,
        registration.artifacts.development_queries,
        registration.artifacts.retriever_manifest,
        registration.policy.adapter_contract,
        registration.policy.adapter_runner,
        registration.policy.preview_module,
        registration.policy.retrieval_server,
    )
    for reference in references:
        _validate_artifact(root, reference)

    route_path = root / registration.artifacts.failure_cluster_route.path
    route = load_failure_cluster_route(route_path)
    if route.permitted_next_action != "broader_development_profile":
        raise ValueError("failure-cluster route does not permit development profiling")

    query_path = root / registration.artifacts.development_queries.path
    query_artifact = _read_object(query_path)
    if query_artifact.get("partition") != registration.execution.query_partition:
        raise ValueError("development profile query partition changed")
    if query_artifact.get("query_count") != registration.execution.query_count:
        raise ValueError("development profile query count changed")
    if query_artifact.get("target_manifest_sha256") != (
        registration.artifacts.target_manifest.sha256
    ):
        raise ValueError("development profile target manifest binding changed")
    if query_artifact.get("query_partitions_sha256") != (
        registration.artifacts.query_partitions.normalized_sha256
    ):
        raise ValueError("development profile partition binding changed")
    queries = query_artifact.get("queries")
    if not isinstance(queries, list) or len(queries) != registration.execution.query_count:
        raise ValueError("development profile queries are invalid")
    query_ids = [str(item.get("query_id")) for item in queries if isinstance(item, dict)]
    if len(query_ids) != registration.execution.query_count:
        raise ValueError("development profile query rows are invalid")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("development profile query IDs must be unique")
    return registration


def preflight_development_profile(
    *,
    registration_path: Path,
    output_path: Path,
    environment: Mapping[str, str],
    allow_existing_for_resume: bool = False,
    health_check: HealthCheck = require_search_service_health,
) -> DevelopmentProfilePreflight:
    registration = load_development_profile_registration(registration_path)
    root = _repository_root(registration_path)
    output_path = output_path.resolve()
    if not output_path.is_relative_to((root / "runs").resolve()):
        raise ValueError("development profile preflight output must stay under runs/")
    if output_path.exists():
        raise ValueError("development profile preflight output already exists")

    output_directory = (root / registration.execution.output_directory).resolve()
    if not output_directory.is_relative_to((root / "runs").resolve()):
        raise ValueError("development profile output directory must stay under runs/")
    output_mode: Literal["new", "failed_only_resume"]
    if output_directory.exists():
        if not allow_existing_for_resume:
            raise ValueError("development profile output already exists; resume is explicit")
        output_mode = "failed_only_resume"
    else:
        output_mode = "new"

    key = environment.get(registration.execution.provider_key_env, "")
    if not key.strip():
        raise ValueError("development profile provider key environment variable is missing")
    health = health_check(
        registration.execution.search_url,
        expected_retriever_id=registration.execution.retriever_id,
        timeout_seconds=5.0,
    )
    query_artifact = _read_object(root / registration.artifacts.development_queries.path)
    preflight = DevelopmentProfilePreflight(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=_sha256_file(registration_path),
        ),
        route_status="regression_only",
        query_count=175,
        query_ids_sha256=str(query_artifact["queries_sha256"]),
        model=registration.execution.model,
        retriever_id=health.retriever_id,
        health_url=health.health_url,
        provider_key_env=registration.execution.provider_key_env,
        provider_key_present=True,
        output_mode=output_mode,
    )
    _atomic_write(output_path, preflight.model_dump(mode="json"))
    return preflight


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("could not locate development-profile repository root")


def _validate_artifact(root: Path, reference: ArtifactReference) -> None:
    path = (root / reference.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(
            f"development profile artifact is missing or escapes root: {reference.path}"
        )
    if _sha256_file(path) != reference.sha256:
        raise ValueError(f"development profile artifact hash changed: {reference.path}")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"development profile artifact is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
