from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .contracts import HarnessConfig, SearchConfig
from .livedrbench_fresh_public import (
    FreshLiveDRBenchRegistration,
    validate_fresh_public_registration,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreparedArm(StrictContract):
    id: Literal["baseline", "candidate"]
    variant_id: str = Field(min_length=1)
    search_kind: Literal["duckduckgo", "tavily"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_path: str = Field(min_length=1)


class FreshPublicPairManifest(StrictContract):
    schema_version: Literal["livedrbench-fresh-public-pair-v0"]
    status: Literal["registered_before_generation"]
    run_label: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    benchmark_id: Literal["livedrbench-fresh-public-v0"]
    fresh_registration_path: str = Field(min_length=1)
    fresh_registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_task_keys: tuple[int, ...]
    selected_task_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_kind: Literal["openai_compatible"]
    provider_model: Literal["deepseek-v4-flash"]
    provider_base_url: HttpUrl
    thinking_mode: Literal["disabled"]
    max_model_calls_per_task: Literal[3]
    max_total_tokens_per_task: Literal[8000]
    max_estimated_cost_usd_per_task: float = Field(gt=0, le=0.01)
    max_output_tokens_per_call: Literal[2048]
    max_evidence_items_per_task: Literal[6]
    max_search_calls_per_task: Literal[5]
    execution_order: tuple[Literal["baseline", "candidate"], Literal["baseline", "candidate"]]
    arms: tuple[PreparedArm, PreparedArm]
    executor_status: Literal["not_implemented"]
    retry_policy: Literal["failed_only_explicit_resume_v0"]
    sealed_holdout_access: Literal["forbidden"]
    provider_calls_before_generation: Literal[0] = 0
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def pair_is_fixed_and_complete(self) -> "FreshPublicPairManifest":
        if self.execution_order != ("baseline", "candidate"):
            raise ValueError("first fresh-public pair must use baseline then candidate order")
        if tuple(arm.id for arm in self.arms) != self.execution_order:
            raise ValueError("arm records must follow the registered execution order")
        if self.arms[0].search_kind != "duckduckgo" or self.arms[1].search_kind != "tavily":
            raise ValueError("fresh-public pair must contain the registered search backends")
        if self.arms[0].config_sha256 == self.arms[1].config_sha256:
            raise ValueError("baseline and candidate config digests must differ by search backend")
        return self


def prepare_fresh_public_pair(
    *,
    registration_path: Path,
    base_config_path: Path,
    output_dir: Path,
    run_label: str,
) -> FreshPublicPairManifest:
    """Freeze both non-secret configs before any model, search, or dataset call."""
    registration = validate_fresh_public_registration(registration_path)
    base_config = HarnessConfig.model_validate_json(base_config_path.read_text(encoding="utf-8"))
    baseline_config, candidate_config = derive_pair_configs(base_config, registration)
    _validate_registered_controls(registration, baseline_config, candidate_config)

    run_dir = output_dir / run_label
    manifest_path = run_dir / "pair_manifest.json"
    expected = _build_manifest(
        registration=registration,
        registration_path=registration_path,
        run_label=run_label,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
    )
    if manifest_path.exists():
        actual = FreshPublicPairManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        _validate_existing_manifest(actual, expected, run_dir)
        return actual
    if run_dir.exists():
        raise ValueError(f"run directory exists without pair manifest: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(run_dir / "baseline.config.snapshot.json", baseline_config.model_dump(mode="json"))
    _atomic_write_json(run_dir / "candidate.config.snapshot.json", candidate_config.model_dump(mode="json"))
    _atomic_write_json(run_dir / "fresh_registration.snapshot.json", registration.model_dump(mode="json"))
    _atomic_write_json(manifest_path, expected.model_dump(mode="json"))
    return expected


def derive_pair_configs(
    base_config: HarnessConfig,
    registration: FreshLiveDRBenchRegistration,
) -> tuple[HarnessConfig, HarnessConfig]:
    """Derive the two arms from one common config so only search changes."""
    shared_search = {
        "timeout_seconds": base_config.search.timeout_seconds,
        "max_results_per_query": registration.budget.max_search_results_per_query,
        "max_download_bytes": base_config.search.max_download_bytes,
        "max_excerpt_characters": base_config.search.max_excerpt_characters,
        "user_agent": base_config.search.user_agent,
        "max_search_calls": registration.budget.max_search_calls_per_task,
    }
    baseline_search = SearchConfig(kind="duckduckgo", **shared_search)
    candidate_search = SearchConfig(
        kind="tavily",
        api_key_env=registration.candidate.api_key_env,
        tavily_search_depth=registration.candidate.tavily_search_depth,
        tavily_basic_credit_price_usd=registration.budget.candidate_search_credit_price_usd,
        **shared_search,
    )
    return (
        base_config.model_copy(update={"search": baseline_search}),
        base_config.model_copy(update={"search": candidate_search}),
    )


def load_and_validate_fresh_public_pair(path: Path) -> FreshPublicPairManifest:
    manifest = FreshPublicPairManifest.model_validate_json(path.read_text(encoding="utf-8"))
    run_dir = path.parent
    expected_registration = validate_fresh_public_registration(
        Path(manifest.fresh_registration_path)
    )
    if _sha256_file(Path(manifest.fresh_registration_path)) != manifest.fresh_registration_sha256:
        raise ValueError("fresh registration hash changed")
    if manifest.selected_task_keys != expected_registration.selected_task_keys:
        raise ValueError("fresh pair selected task keys changed")
    if manifest.selected_task_keys_sha256 != expected_registration.selected_task_keys_sha256:
        raise ValueError("fresh pair task-key hash changed")
    if manifest.dataset_response_sha256 != expected_registration.dataset_response_sha256:
        raise ValueError("fresh pair dataset hash changed")
    configs: dict[str, HarnessConfig] = {}
    for arm in manifest.arms:
        snapshot_path = run_dir / arm.config_path
        if not snapshot_path.is_file():
            raise ValueError(f"missing {arm.id} config snapshot")
        config = HarnessConfig.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
        if _config_sha256(config) != arm.config_sha256:
            raise ValueError(f"{arm.id} config snapshot hash changed")
        configs[arm.id] = config
    _validate_registered_controls(
        expected_registration, configs["baseline"], configs["candidate"]
    )
    return manifest


def _build_manifest(
    *,
    registration: FreshLiveDRBenchRegistration,
    registration_path: Path,
    run_label: str,
    baseline_config: HarnessConfig,
    candidate_config: HarnessConfig,
) -> FreshPublicPairManifest:
    budget = registration.budget
    return FreshPublicPairManifest(
        schema_version="livedrbench-fresh-public-pair-v0",
        status="registered_before_generation",
        run_label=run_label,
        benchmark_id=registration.benchmark_id,
        fresh_registration_path=str(registration_path.resolve()),
        fresh_registration_sha256=_sha256_file(registration_path),
        selected_task_keys=registration.selected_task_keys,
        selected_task_keys_sha256=registration.selected_task_keys_sha256,
        dataset_response_sha256=registration.dataset_response_sha256,
        provider_kind=baseline_config.provider.kind,
        provider_model=baseline_config.provider.model,
        provider_base_url=baseline_config.provider.base_url,
        thinking_mode=baseline_config.provider.thinking_mode,
        max_model_calls_per_task=budget.max_model_calls_per_task,
        max_total_tokens_per_task=budget.max_total_tokens_per_task,
        max_estimated_cost_usd_per_task=budget.max_estimated_cost_usd_per_task,
        max_output_tokens_per_call=budget.max_output_tokens_per_call,
        max_evidence_items_per_task=baseline_config.run.max_evidence,
        max_search_calls_per_task=budget.max_search_calls_per_task,
        execution_order=("baseline", "candidate"),
        arms=(
            PreparedArm(
                id="baseline",
                variant_id=registration.baseline.id,
                search_kind="duckduckgo",
                config_sha256=_config_sha256(baseline_config),
                config_path="baseline.config.snapshot.json",
            ),
            PreparedArm(
                id="candidate",
                variant_id=registration.candidate.id,
                search_kind="tavily",
                config_sha256=_config_sha256(candidate_config),
                config_path="candidate.config.snapshot.json",
            ),
        ),
        executor_status="not_implemented",
        retry_policy="failed_only_explicit_resume_v0",
        sealed_holdout_access="forbidden",
        claim_boundary=(
            "Pre-generation paired execution registration only. It freezes two search-backend "
            "arms from one common OpenAI-compatible model configuration and makes no provider, "
            "search, dataset, Judge, GPU, effectiveness, or model-capability claim."
        ),
    )


def _validate_registered_controls(
    registration: FreshLiveDRBenchRegistration,
    baseline_config: HarnessConfig,
    candidate_config: HarnessConfig,
) -> None:
    budget = registration.budget
    if baseline_config.provider.model != budget.model or candidate_config.provider.model != budget.model:
        raise ValueError("pair provider model differs from fresh registration")
    if (
        baseline_config.provider.kind != "openai_compatible"
        or candidate_config.provider.kind != "openai_compatible"
    ):
        raise ValueError("fresh pair requires openai_compatible provider")
    if (
        baseline_config.provider.base_url != candidate_config.provider.base_url
        or baseline_config.provider.thinking_mode != budget.thinking_mode
        or candidate_config.provider.thinking_mode != budget.thinking_mode
    ):
        raise ValueError("pair provider endpoint or thinking mode differs")
    if baseline_config.provider.pricing != candidate_config.provider.pricing:
        raise ValueError("pair provider pricing differs")
    for config in (baseline_config, candidate_config):
        if config.run.max_evidence != budget.max_evidence_items_per_task:
            raise ValueError("pair evidence cap differs from fresh registration")
        if config.run.budget.max_llm_calls != budget.max_model_calls_per_task:
            raise ValueError("pair LLM call cap differs from fresh registration")
        if config.run.budget.max_total_tokens != budget.max_total_tokens_per_task:
            raise ValueError("pair token cap differs from fresh registration")
        if config.run.budget.max_estimated_cost_usd != budget.max_estimated_cost_usd_per_task:
            raise ValueError("pair LLM cost cap differs from fresh registration")
        if config.run.budget.max_output_tokens_per_call != budget.max_output_tokens_per_call:
            raise ValueError("pair output-token cap differs from fresh registration")
        if config.search.max_search_calls != budget.max_search_calls_per_task:
            raise ValueError("pair search-call cap differs from fresh registration")
        if config.search.max_results_per_query != budget.max_search_results_per_query:
            raise ValueError("pair search result cap differs from fresh registration")
    if baseline_config.search.kind != "duckduckgo" or baseline_config.search.api_key_env is not None:
        raise ValueError("baseline search configuration changed")
    if (
        candidate_config.search.kind != "tavily"
        or candidate_config.search.api_key_env != registration.candidate.api_key_env
        or candidate_config.search.tavily_search_depth != registration.candidate.tavily_search_depth
        or candidate_config.search.tavily_basic_credit_price_usd
        != budget.candidate_search_credit_price_usd
    ):
        raise ValueError("candidate search configuration changed")


def _validate_existing_manifest(
    actual: FreshPublicPairManifest,
    expected: FreshPublicPairManifest,
    run_dir: Path,
) -> None:
    fields = (
        "run_label",
        "benchmark_id",
        "fresh_registration_sha256",
        "selected_task_keys",
        "selected_task_keys_sha256",
        "dataset_response_sha256",
        "provider_kind",
        "provider_model",
        "provider_base_url",
        "thinking_mode",
        "max_model_calls_per_task",
        "max_total_tokens_per_task",
        "max_estimated_cost_usd_per_task",
        "max_output_tokens_per_call",
        "max_evidence_items_per_task",
        "max_search_calls_per_task",
        "execution_order",
        "arms",
        "executor_status",
        "retry_policy",
        "sealed_holdout_access",
    )
    if any(getattr(actual, field) != getattr(expected, field) for field in fields):
        raise ValueError("existing fresh-public pair manifest differs from current controls")
    load_and_validate_fresh_public_pair(run_dir / "pair_manifest.json")


def _config_sha256(config: HarnessConfig) -> str:
    canonical = json.dumps(
        config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
