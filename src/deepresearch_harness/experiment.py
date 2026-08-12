from __future__ import annotations

import hashlib
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

from .benchmark import validate_suite_assets
from .contracts import BudgetLimits, Pricing


class BudgetMode(str, Enum):
    TOKEN_MATCHED = "token_matched"
    COST_MATCHED = "cost_matched"


class FrozenProviderSpec(BaseModel):
    kind: Literal["openai_compatible"]
    model: str = Field(min_length=1)
    base_url: HttpUrl
    thinking_mode: Literal["enabled", "disabled"]
    pricing: Pricing
    pricing_source: HttpUrl
    pricing_checked_at: date


class ExperimentManifest(BaseModel):
    experiment_id: str = Field(min_length=1)
    status: Literal["planned", "ready"]
    implementation_revision: str = Field(pattern="^[0-9a-f]{40}$")
    suite_path: str = Field(min_length=1)
    suite_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    variants: list[Literal["b0_search_write", "b1_plan_search_ledger_write"]]
    budget_mode: BudgetMode
    budget: BudgetLimits
    provider: FrozenProviderSpec
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def comparison_is_matched_and_priced(self) -> "ExperimentManifest":
        if set(self.variants) != {"b0_search_write", "b1_plan_search_ledger_write"} or len(self.variants) != 2:
            raise ValueError("manifest must compare B0 and B1 exactly once")
        if self.budget_mode is BudgetMode.TOKEN_MATCHED and self.budget.max_total_tokens is None:
            raise ValueError("token-matched manifest requires max_total_tokens")
        if self.budget_mode is BudgetMode.COST_MATCHED and self.budget.max_estimated_cost_usd is None:
            raise ValueError("cost-matched manifest requires max_estimated_cost_usd")
        if self.budget_mode is BudgetMode.COST_MATCHED:
            pricing = self.provider.pricing
            if pricing.input_cache_miss_per_million_usd <= 0 or pricing.output_per_million_usd <= 0:
                raise ValueError("cost-matched manifest requires frozen non-zero pricing")
        return self


def validate_experiment_manifest(path: Path) -> ExperimentManifest:
    manifest = ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))
    suite_path = (path.parent / manifest.suite_path).resolve()
    suite, _ = validate_suite_assets(suite_path)
    corpus_path = suite_path.parent / suite.corpus_path
    if _sha256(suite_path) != manifest.suite_sha256:
        raise ValueError("suite SHA-256 does not match the experiment manifest")
    if _sha256(corpus_path) != manifest.corpus_sha256:
        raise ValueError("corpus SHA-256 does not match the experiment manifest")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
