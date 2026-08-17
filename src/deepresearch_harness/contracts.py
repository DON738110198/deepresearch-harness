from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, JsonValue, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Task(BaseModel):
    id: str
    question: str = Field(min_length=1)
    decision_context: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class PlanStep(BaseModel):
    id: str
    objective: str = Field(min_length=1)


class PlannedObligation(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    search_query: str = Field(min_length=1)


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1)
    obligations: list[PlannedObligation] = Field(default_factory=list)

    @model_validator(mode="after")
    def plan_ids_are_unique(self) -> "Plan":
        obligation_ids = [item.id for item in self.obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("plan obligation ids must be unique")
        return self


class Evidence(BaseModel):
    id: str
    title: str = Field(min_length=1)
    url: HttpUrl
    excerpt: str = Field(min_length=1)
    query: str
    collected_at: datetime = Field(default_factory=utc_now)


class Claim(BaseModel):
    id: str
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    support: Literal["direct", "partial"] = "direct"


class Citation(BaseModel):
    id: str
    claim_id: str
    evidence_id: str
    marker: str


class EvidenceDebt(BaseModel):
    obligation_id: str = Field(min_length=1)
    status: Literal["resolved", "open"]
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def status_matches_links(self) -> "EvidenceDebt":
        if self.status == "resolved" and (not self.evidence_ids or not self.claim_ids):
            raise ValueError("resolved evidence debt requires evidence_ids and claim_ids")
        if self.status == "open" and self.claim_ids:
            raise ValueError("open evidence debt cannot reference resolved claims")
        return self


class Usage(BaseModel):
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    input_cache_hit_tokens: int = Field(ge=0, default=0)
    input_cache_miss_tokens: int = Field(ge=0, default=0)
    estimated_cost_usd: float = Field(ge=0, default=0.0)


class Pricing(BaseModel):
    input_per_million_usd: float = Field(ge=0, default=0.0)
    input_cache_hit_per_million_usd: float = Field(ge=0, default=0.0)
    input_cache_miss_per_million_usd: float = Field(ge=0, default=0.0)
    output_per_million_usd: float = Field(ge=0, default=0.0)


class ProviderConfig(BaseModel):
    kind: Literal["fake", "openai_compatible"] = "fake"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = Field(ge=1, default=60)
    thinking_mode: Literal["enabled", "disabled"] | None = None
    pricing: Pricing = Field(default_factory=Pricing)


class SearchConfig(BaseModel):
    kind: Literal["local", "duckduckgo", "tavily"] = "local"
    api_key_env: str | None = None
    tavily_search_depth: Literal["basic", "advanced"] = "basic"
    max_search_calls: int = Field(ge=1, le=20, default=5)
    tavily_basic_credit_price_usd: float = Field(ge=0, default=0.008)
    timeout_seconds: int = Field(ge=1, le=120, default=20)
    max_results_per_query: int = Field(ge=1, le=10, default=5)
    max_download_bytes: int = Field(ge=16_384, le=5_000_000, default=1_000_000)
    max_excerpt_characters: int = Field(ge=200, le=5_000, default=1_200)
    user_agent: str = Field(
        min_length=1,
        default="training-free-deepresearch-harness/0.1 (+https://github.com/DON738110198/deepresearch-harness)",
    )

    @model_validator(mode="after")
    def key_contract_matches_search_backend(self) -> "SearchConfig":
        if self.kind == "tavily" and self.api_key_env != "TAVILY_API_KEY":
            raise ValueError("tavily search requires api_key_env=TAVILY_API_KEY")
        if self.kind != "tavily" and self.api_key_env is not None:
            raise ValueError("api_key_env is only supported for tavily search")
        return self


class BudgetLimits(BaseModel):
    max_total_tokens: int | None = Field(default=None, gt=0)
    max_estimated_cost_usd: float | None = Field(default=None, gt=0)
    max_llm_calls: int = Field(default=3, gt=0)
    max_output_tokens_per_call: int = Field(default=2048, gt=0)


class RunConfig(BaseModel):
    max_evidence: int = Field(ge=1, default=6)
    budget: BudgetLimits = Field(default_factory=BudgetLimits)


class HarnessConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def cost_budget_requires_pricing(self) -> "HarnessConfig":
        if self.run.budget.max_estimated_cost_usd is None:
            return self
        pricing = self.provider.pricing
        has_input_price = any(
            (
                pricing.input_per_million_usd,
                pricing.input_cache_hit_per_million_usd,
                pricing.input_cache_miss_per_million_usd,
            )
        )
        if not has_input_price or pricing.output_per_million_usd <= 0:
            raise ValueError("a cost budget requires non-zero input and output pricing")
        return self


class TraceEvent(BaseModel):
    stage: str
    provider: str
    model: str
    started_at: datetime = Field(default_factory=utc_now)
    latency_ms: int = Field(ge=0)
    usage: Usage = Field(default_factory=Usage)
    outcome: Literal["ok", "error"]
    detail: str = ""


class StatusEvent(BaseModel):
    status: RunStatus
    at: datetime = Field(default_factory=utc_now)
    detail: str = ""


class RunState(BaseModel):
    run_id: str
    task: Task
    variant: Literal[
        "b0_search_write",
        "b1_plan_search_ledger_write",
        "b1_benchmark_structured",
        "b1_live_primary_sources",
        "b2_obligation_evidence_debt",
    ] = "b1_plan_search_ledger_write"
    status: RunStatus = RunStatus.PENDING
    status_history: list[StatusEvent] = Field(default_factory=list)
    plan: Plan | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_debts: list[EvidenceDebt] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    total_usage: Usage = Field(default_factory=Usage)
    budget_limits: BudgetLimits = Field(default_factory=BudgetLimits)
    stop_reason: Literal["budget_exhausted"] | None = None
    structured_answer: JsonValue | None = None
    report_path: str | None = None
    error: str | None = None

    def transition(self, status: RunStatus, detail: str = "") -> None:
        self.status = status
        self.status_history.append(StatusEvent(status=status, detail=detail))

    def add_trace(self, event: TraceEvent) -> None:
        self.trace.append(event)
        self.total_usage.input_tokens += event.usage.input_tokens
        self.total_usage.output_tokens += event.usage.output_tokens
        self.total_usage.input_cache_hit_tokens += event.usage.input_cache_hit_tokens
        self.total_usage.input_cache_miss_tokens += event.usage.input_cache_miss_tokens
        self.total_usage.estimated_cost_usd += event.usage.estimated_cost_usd

    @field_validator("citations")
    @classmethod
    def citations_reference_known_ids(cls, citations: list[Citation]) -> list[Citation]:
        markers = [citation.marker for citation in citations]
        if len(markers) != len(set(markers)):
            raise ValueError("citation markers must be unique")
        return citations

    @field_validator("evidence_debts")
    @classmethod
    def evidence_debt_obligations_are_unique(cls, debts: list[EvidenceDebt]) -> list[EvidenceDebt]:
        obligation_ids = [debt.obligation_id for debt in debts]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("evidence debt obligation ids must be unique")
        return debts


class DirectWriteDraft(BaseModel):
    claims: list[Claim] = Field(min_length=1)
    report: str = Field(min_length=1)

    @model_validator(mode="after")
    def claim_ids_are_unique(self) -> "DirectWriteDraft":
        claim_ids = [claim.id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("direct-write claim ids must be unique")
        return self
