from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .contracts import HarnessConfig, RunState, RunStatus, SearchConfig
from .livedrbench_fresh_public import (
    FreshLiveDRBenchRegistration,
    validate_fresh_public_dataset,
    validate_fresh_public_registration,
)
from .pipeline import BenchmarkResearchPipeline
from .providers import provider_from_config
from .public_benchmark import (
    ExactClaimMetrics,
    LiveDRBenchTask,
    _coerce_prediction_shape,
    _benchmark_context,
    exact_main_claim_metrics,
    official_shape_compatible,
)
from .web_research import live_collector_from_config


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
    executor_status: Literal["implemented_unexecuted"]
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


class PairTaskAttempt(StrictContract):
    key: int
    category: str = Field(min_length=1)
    attempt_index: int = Field(ge=1)
    status: Literal["succeeded", "failed"]
    run_id: str | None = None
    state_path: str | None = None
    state_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_path: str | None = None
    report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_type: str | None = None
    total_tokens: int = Field(ge=0, default=0)
    estimated_llm_cost_usd: float = Field(ge=0, default=0)
    estimated_search_cost_usd: float = Field(ge=0, default=0)
    traced_latency_ms: int = Field(ge=0, default=0)
    http_search_attempts: int = Field(ge=0, default=0)
    logical_search_queries: int = Field(ge=0, default=0)
    fetched_sources: int = Field(ge=0, default=0)
    fetch_errors: int = Field(ge=0, default=0)
    exact_main_claim: ExactClaimMetrics | None = None


class PairArmExecution(StrictContract):
    id: Literal["baseline", "candidate"]
    variant_id: str = Field(min_length=1)
    status: Literal["not_started", "succeeded", "completed_with_failures"]
    attempts: list[PairTaskAttempt] = Field(default_factory=list)


class FreshPublicPairExecution(StrictContract):
    schema_version: Literal["livedrbench-fresh-public-execution-v0"]
    status: Literal["running", "succeeded", "completed_with_failures"]
    pair_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resumed_failed_only: bool
    provider_calls_before_execution: Literal[0] = 0
    arms: tuple[PairArmExecution, PairArmExecution]
    evaluator: Literal["compatibility_exact_main_claim_v1"]
    official_evaluator_status: Literal["planned_not_run"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def arms_follow_pair_order(self) -> "FreshPublicPairExecution":
        if tuple(arm.id for arm in self.arms) != ("baseline", "candidate"):
            raise ValueError("pair execution arms must remain baseline then candidate")
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


def execute_fresh_public_pair(
    *,
    pair_manifest_path: Path,
    resume_failed: bool = False,
) -> FreshPublicPairExecution:
    """Execute only the unresolved arm/task records from an immutable pair manifest.

    A resume is opt-in. Successful task records are hash-verified and never sent to a
    provider again; failed records are preserved and retried only with ``resume_failed``.
    """
    manifest = load_and_validate_fresh_public_pair(pair_manifest_path)
    run_dir = pair_manifest_path.parent
    registration = validate_fresh_public_registration(Path(manifest.fresh_registration_path))
    baseline_config, candidate_config = _load_arm_configs(manifest, run_dir)
    _preflight_execution_keys(baseline_config, candidate_config)
    tasks = validate_fresh_public_dataset(registration)
    if tuple(task.key for task in tasks) != manifest.selected_task_keys:
        raise ValueError("pinned dataset task order differs from pair registration")
    execution_path = run_dir / "pair_execution.json"
    if execution_path.exists():
        execution = FreshPublicPairExecution.model_validate_json(
            execution_path.read_text(encoding="utf-8")
        )
        _validate_execution(execution, manifest, tasks, run_dir)
        if execution.status == "succeeded":
            return execution
        if not resume_failed:
            raise ValueError(
                "pair has preserved failed tasks; rerun only with resume_failed=True"
            )
    else:
        execution = FreshPublicPairExecution(
            schema_version="livedrbench-fresh-public-execution-v0",
            status="running",
            pair_manifest_sha256=_sha256_file(pair_manifest_path),
            dataset_response_sha256=registration.dataset_response_sha256,
            resumed_failed_only=resume_failed,
            arms=(
                PairArmExecution(
                    id="baseline",
                    variant_id=manifest.arms[0].variant_id,
                    status="not_started",
                ),
                PairArmExecution(
                    id="candidate",
                    variant_id=manifest.arms[1].variant_id,
                    status="not_started",
                ),
            ),
            evaluator="compatibility_exact_main_claim_v1",
            official_evaluator_status="planned_not_run",
            claim_boundary=(
                "Execution artifact for a frozen-model token-matched search-backend ablation. "
                "It is not an official LiveDRBench result, leaderboard result, total-cost-matched "
                "comparison, or model-capability improvement."
            ),
        )
        _write_execution(execution_path, execution)

    config_by_arm = {"baseline": baseline_config, "candidate": candidate_config}
    arm_by_id = {arm.id: arm for arm in execution.arms}
    for arm_spec in manifest.arms:
        arm = arm_by_id[arm_spec.id]
        task_attempts = list(arm.attempts)
        for task in tasks:
            latest = _latest_attempt(task_attempts, task.key)
            if latest is not None and latest.status == "succeeded":
                _validate_success_attempt(latest, run_dir)
                continue
            if latest is not None and not resume_failed:
                continue
            attempt = _run_task_attempt(
                task=task,
                config=config_by_arm[arm_spec.id],
                variant_id=arm_spec.variant_id,
                task_root=run_dir / "arms" / arm_spec.id / f"key-{task.key}",
                attempt_index=(latest.attempt_index + 1) if latest else 1,
                decision_context=_benchmark_context_for_pair(registration),
            )
            task_attempts.append(attempt)
            updated_arm = arm.model_copy(
                update={"attempts": task_attempts, "status": _arm_status(task_attempts, tasks)}
            )
            arm_by_id[arm_spec.id] = updated_arm
            execution = execution.model_copy(
                update={
                    "arms": (arm_by_id["baseline"], arm_by_id["candidate"]),
                    "updated_at": datetime.now(timezone.utc),
                    "status": "running",
                    "resumed_failed_only": resume_failed,
                }
            )
            _write_execution(execution_path, execution)
        arm_by_id[arm_spec.id] = arm_by_id[arm_spec.id].model_copy(
            update={"status": _arm_status(task_attempts, tasks)}
        )
        execution = execution.model_copy(
            update={
                "arms": (arm_by_id["baseline"], arm_by_id["candidate"]),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        _write_execution(execution_path, execution)

    succeeded = all(arm.status == "succeeded" for arm in execution.arms)
    execution = execution.model_copy(
        update={
            "status": "succeeded" if succeeded else "completed_with_failures",
            "updated_at": datetime.now(timezone.utc),
        }
    )
    _write_execution(execution_path, execution)
    return execution


def _load_arm_configs(
    manifest: FreshPublicPairManifest, run_dir: Path
) -> tuple[HarnessConfig, HarnessConfig]:
    configs = {
        arm.id: HarnessConfig.model_validate_json(
            (run_dir / arm.config_path).read_text(encoding="utf-8")
        )
        for arm in manifest.arms
    }
    return configs["baseline"], configs["candidate"]


def _preflight_execution_keys(
    baseline_config: HarnessConfig, candidate_config: HarnessConfig
) -> None:
    """Fail before either arm can incur model/search cost."""
    provider_from_config(baseline_config)
    provider_from_config(candidate_config)
    if not candidate_config.search.api_key_env or not os.environ.get(
        candidate_config.search.api_key_env
    ):
        raise RuntimeError(
            f"missing search API key in environment variable {candidate_config.search.api_key_env}"
        )


def _run_task_attempt(
    *,
    task: LiveDRBenchTask,
    config: HarnessConfig,
    variant_id: str,
    task_root: Path,
    attempt_index: int,
    decision_context: str,
) -> PairTaskAttempt:
    provider = provider_from_config(config)
    pipeline = BenchmarkResearchPipeline(
        provider=provider,
        collector=live_collector_from_config(config.search),
        output_dir=task_root / f"attempt-{attempt_index}",
        max_evidence=config.run.max_evidence,
        budget_limits=config.run.budget,
        report_language="auto",
    )
    try:
        state = pipeline.run(task.question, decision_context=decision_context)
        return _attempt_from_state(task, state, attempt_index=attempt_index, status="succeeded")
    except Exception as error:
        state_path = _latest_state_path(task_root / f"attempt-{attempt_index}")
        state = _load_state(state_path)
        if state is None:
            return PairTaskAttempt(
                key=task.key,
                category=task.category,
                attempt_index=attempt_index,
                status="failed",
                error_type=type(error).__name__,
            )
        return _attempt_from_state(
            task,
            state,
            attempt_index=attempt_index,
            status="failed",
            error_type=type(error).__name__,
            state_path=state_path,
        )


def _attempt_from_state(
    task: LiveDRBenchTask,
    state: RunState,
    *,
    attempt_index: int,
    status: Literal["succeeded", "failed"],
    error_type: str | None = None,
    state_path: Path | None = None,
) -> PairTaskAttempt:
    state_path = state_path or (Path(state.report_path).parent / "state.json" if state.report_path else None)
    if state_path is None or not state_path.is_file():
        raise ValueError("pipeline state did not retain a state.json artifact")
    report_path = Path(state.report_path) if state.report_path else None
    prediction = _coerce_prediction_shape(state.structured_answer, task.ground_truths)
    metrics = exact_main_claim_metrics(task, prediction)
    search_events = [event for event in state.trace if event.stage == "search"]
    search_details = [_trace_detail(event.detail) for event in search_events]
    logical_queries = {
        detail.get("logical_query_index")
        for detail in search_details
        if isinstance(detail.get("logical_query_index"), int)
    }
    return PairTaskAttempt(
        key=task.key,
        category=task.category,
        attempt_index=attempt_index,
        status=status,
        run_id=state.run_id,
        state_path=str(state_path),
        state_sha256=_sha256_file(state_path),
        report_path=str(report_path) if report_path and report_path.is_file() else None,
        report_sha256=_sha256_file(report_path) if report_path and report_path.is_file() else None,
        error_type=error_type,
        total_tokens=state.total_usage.input_tokens + state.total_usage.output_tokens,
        estimated_llm_cost_usd=state.total_usage.estimated_cost_usd,
        estimated_search_cost_usd=sum(
            float(detail.get("estimated_search_cost_usd", 0.0)) for detail in search_details
        ),
        traced_latency_ms=sum(event.latency_ms for event in state.trace),
        http_search_attempts=len(search_events),
        logical_search_queries=len(logical_queries),
        fetched_sources=sum(event.stage == "fetch" and event.outcome == "ok" for event in state.trace),
        fetch_errors=sum(event.stage == "fetch" and event.outcome == "error" for event in state.trace),
        exact_main_claim=metrics,
    )


def _latest_attempt(attempts: list[PairTaskAttempt], key: int) -> PairTaskAttempt | None:
    matching = [attempt for attempt in attempts if attempt.key == key]
    return max(matching, key=lambda attempt: attempt.attempt_index) if matching else None


def _arm_status(
    attempts: list[PairTaskAttempt], tasks: tuple[LiveDRBenchTask, ...]
) -> Literal["not_started", "succeeded", "completed_with_failures"]:
    if not attempts:
        return "not_started"
    latest = [_latest_attempt(attempts, task.key) for task in tasks]
    if any(item is None for item in latest):
        return "completed_with_failures"
    return "succeeded" if all(item.status == "succeeded" for item in latest if item) else "completed_with_failures"


def _latest_state_path(root: Path) -> Path | None:
    candidates = sorted(root.glob("*/state.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_state(path: Path | None) -> RunState | None:
    if path is None:
        return None
    try:
        return RunState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _trace_detail(detail: str) -> dict[str, Any]:
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _validate_success_attempt(attempt: PairTaskAttempt, run_dir: Path) -> None:
    if not attempt.state_path or not attempt.state_sha256:
        raise ValueError("successful task attempt is missing its state artifact hash")
    state_path = Path(attempt.state_path)
    if not state_path.is_file() or _sha256_file(state_path) != attempt.state_sha256:
        raise ValueError("successful task state artifact changed")
    state = _load_state(state_path)
    if state is None or state.status != RunStatus.SUCCEEDED:
        raise ValueError("successful task state is not a succeeded run")
    if attempt.report_path:
        report_path = Path(attempt.report_path)
        if not attempt.report_sha256 or not report_path.is_file() or _sha256_file(report_path) != attempt.report_sha256:
            raise ValueError("successful task report artifact changed")
    if not state_path.is_relative_to(run_dir):
        raise ValueError("successful task state escaped the pair run directory")


def _validate_execution(
    execution: FreshPublicPairExecution,
    manifest: FreshPublicPairManifest,
    tasks: tuple[LiveDRBenchTask, ...],
    run_dir: Path,
) -> None:
    if execution.pair_manifest_sha256 != _sha256_file(run_dir / "pair_manifest.json"):
        raise ValueError("pair execution does not bind the current pair manifest")
    if execution.dataset_response_sha256 != manifest.dataset_response_sha256:
        raise ValueError("pair execution dataset hash differs from registration")
    expected_variant_ids = {arm.id: arm.variant_id for arm in manifest.arms}
    expected_keys = {task.key for task in tasks}
    for arm in execution.arms:
        if arm.variant_id != expected_variant_ids[arm.id]:
            raise ValueError("pair execution variant changed")
        for attempt in arm.attempts:
            if attempt.key not in expected_keys:
                raise ValueError("pair execution contains an unregistered task")
            if attempt.status == "succeeded":
                _validate_success_attempt(attempt, run_dir)


def _benchmark_context_for_pair(registration: FreshLiveDRBenchRegistration) -> str:
    return (
        f"Compatibility pilot for {registration.dataset_id} at revision {registration.dataset_revision}. "
        "The model receives only the public question and fetched evidence, never benchmark ground truth. "
        "Use one-pass B1 and follow the question's requested JSON output exactly."
    )


def _write_execution(path: Path, execution: FreshPublicPairExecution) -> None:
    _atomic_write_json(path, execution.model_dump(mode="json"))


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
        executor_status="implemented_unexecuted",
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
