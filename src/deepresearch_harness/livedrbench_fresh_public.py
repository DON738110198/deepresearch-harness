from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .public_benchmark import (
    LiveDRBenchManifest,
    LiveDRBenchTask,
    fetch_livedrbench_tasks,
    load_livedrbench_manifest,
)


SHA256_PATTERN = r"^[0-9a-f]{64}$"
KEYS = (10, 23, 38, 86, 99)
EXCLUDED_KEYS = (4, 31, 40, 55, 76)
ALL_PREVIEW_KEYS = (4, 10, 23, 31, 38, 40, 55, 76, 86, 99)
DEFAULT_PARENT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "livedrbench_preview_v0"
    / "manifest.json"
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchVariant(StrictContract):
    id: str = Field(min_length=1)
    status: Literal["existing", "planned"]
    search_provider: str = Field(min_length=1)
    adapter: str = Field(min_length=1)


class TaskBudget(StrictContract):
    model: Literal["deepseek-v4-flash"]
    thinking_mode: Literal["disabled"]
    max_model_calls_per_task: Literal[3]
    max_total_tokens_per_task: Literal[8000]
    max_estimated_cost_usd_per_task: float = Field(gt=0, le=0.01)
    max_evidence_items_per_task: Literal[6]
    max_search_calls_per_task: Literal[5]


class FreshLiveDRBenchRegistration(StrictContract):
    schema_version: Literal["livedrbench-fresh-public-registration-v0"]
    status: Literal["registered_before_generation"]
    benchmark_id: Literal["livedrbench-fresh-public-v0"]
    dataset_id: Literal["microsoft/LiveDRBench"]
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_config: Literal["preview"]
    dataset_split: Literal["test"]
    source_repository: HttpUrl
    source_repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    parent_preview_manifest_path: str = Field(min_length=1)
    parent_preview_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_response_sha256: str = Field(pattern=SHA256_PATTERN)
    available_task_keys: tuple[int, ...]
    excluded_prior_task_keys: tuple[int, ...]
    selected_task_keys: tuple[int, ...]
    selected_task_keys_sha256: str = Field(pattern=SHA256_PATTERN)
    selection_algorithm: Literal["sorted_key_filter_excluded_take_first_v0"]
    category_by_key: dict[str, str]
    selected_category_coverage: dict[str, int]
    baseline: SearchVariant
    candidate: SearchVariant
    budget: TaskBudget
    evaluator: Literal["compatibility_exact_main_claim_v1"]
    official_evaluator_status: Literal["planned_not_run"]
    sealed_holdout_access: Literal["forbidden"]
    provider_calls_before_generation: Literal[0] = 0
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def contract_is_fresh_and_deterministic(self) -> "FreshLiveDRBenchRegistration":
        if self.dataset_id != "microsoft/LiveDRBench":
            raise ValueError("fresh registration must target LiveDRBench")
        if self.available_task_keys != ALL_PREVIEW_KEYS:
            raise ValueError("available preview task keys changed")
        if self.excluded_prior_task_keys != EXCLUDED_KEYS:
            raise ValueError("prior preview task exclusion changed")
        available = set(self.available_task_keys)
        excluded = set(self.excluded_prior_task_keys)
        if not excluded.issubset(available):
            raise ValueError("excluded task keys are not in the pinned preview")
        expected = tuple(sorted(available - excluded))[: len(KEYS)]
        if self.selected_task_keys != expected or self.selected_task_keys != KEYS:
            raise ValueError("selected task keys do not follow the registered rule")
        if set(self.selected_task_keys) & excluded:
            raise ValueError("fresh task selection overlaps the prior preview")
        if set(self.category_by_key) != {str(key) for key in available}:
            raise ValueError("category map must cover every pinned preview key")
        expected_coverage: dict[str, int] = {}
        for key in self.selected_task_keys:
            category = self.category_by_key[str(key)]
            expected_coverage[category] = expected_coverage.get(category, 0) + 1
        if self.selected_category_coverage != expected_coverage:
            raise ValueError("selected category coverage differs from category map")
        if self.selected_task_keys_sha256 != selected_keys_sha256(self.selected_task_keys):
            raise ValueError("selected task-key hash changed")
        if self.baseline.status != "existing":
            raise ValueError("baseline must bind the existing public runner")
        if self.candidate.status != "planned":
            raise ValueError("candidate must remain planned")
        if self.candidate.id != "stable-search-provider-adapter-v0":
            raise ValueError("candidate mechanism changed")
        return self


def selected_keys_sha256(keys: tuple[int, ...] | list[int]) -> str:
    canonical = "".join(f"{key}\n" for key in keys)
    return sha256(canonical.encode("utf-8")).hexdigest()


def load_fresh_public_registration(path: Path) -> FreshLiveDRBenchRegistration:
    return FreshLiveDRBenchRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def validate_fresh_public_registration(
    path: Path, *, parent_manifest_path: Path | None = None
) -> FreshLiveDRBenchRegistration:
    registration = load_fresh_public_registration(path)
    parent_path = (
        parent_manifest_path
        or path.parents[1] / "livedrbench_preview_v0" / "manifest.json"
    )
    if not parent_path.is_file():
        raise ValueError(f"parent preview manifest is missing: {parent_path}")
    parent_bytes = parent_path.read_bytes()
    if sha256(parent_bytes).hexdigest() != registration.parent_preview_manifest_sha256:
        raise ValueError("parent preview manifest hash changed")
    parent = load_livedrbench_manifest(parent_path)
    if tuple(parent.task_keys) != EXCLUDED_KEYS:
        raise ValueError("parent preview task keys changed")
    if parent.dataset_revision != registration.dataset_revision:
        raise ValueError("dataset revision differs from parent preview")
    return registration


def validate_fresh_public_dataset(
    registration: FreshLiveDRBenchRegistration | Path,
    *,
    parent_manifest_path: Path | None = None,
) -> tuple[LiveDRBenchTask, ...]:
    """Fetch and validate the pinned public rows only when explicitly called.

    This helper is intentionally the sole network boundary for this registration.
    Loading or validating the static registration never fetches dataset rows.
    """
    if isinstance(registration, Path):
        registration = validate_fresh_public_registration(
            registration, parent_manifest_path=parent_manifest_path
        )
    parent_path = parent_manifest_path or DEFAULT_PARENT_MANIFEST_PATH
    parent = load_livedrbench_manifest(parent_path)
    fetch_manifest = parent.model_copy(
        update={"task_keys": list(registration.available_task_keys)}
    )
    tasks, response_sha256 = fetch_livedrbench_tasks(fetch_manifest)
    if response_sha256 != registration.dataset_response_sha256:
        raise ValueError("pinned dataset response hash changed")
    by_key = {task.key: task for task in tasks}
    if set(by_key) != set(registration.available_task_keys):
        raise ValueError("pinned dataset rows do not cover the registered keys")
    for key, category in registration.category_by_key.items():
        if by_key[int(key)].category != category:
            raise ValueError(f"category changed for LiveDRBench task {key}")
    return tuple(by_key[key] for key in registration.selected_task_keys)
