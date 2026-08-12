from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


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
    created_at: datetime = Field(default_factory=utc_now)


class PlanStep(BaseModel):
    id: str
    objective: str = Field(min_length=1)


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1)


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


class Usage(BaseModel):
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    estimated_cost_usd: float = Field(ge=0, default=0.0)


class Pricing(BaseModel):
    input_per_million_usd: float = Field(ge=0, default=0.0)
    output_per_million_usd: float = Field(ge=0, default=0.0)


class ProviderConfig(BaseModel):
    kind: Literal["fake", "openai_compatible"] = "fake"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = Field(ge=1, default=60)
    pricing: Pricing = Field(default_factory=Pricing)


class RunConfig(BaseModel):
    max_evidence: int = Field(ge=1, default=6)


class HarnessConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    run: RunConfig = Field(default_factory=RunConfig)


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
    status: RunStatus = RunStatus.PENDING
    status_history: list[StatusEvent] = Field(default_factory=list)
    plan: Plan | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    total_usage: Usage = Field(default_factory=Usage)
    report_path: str | None = None
    error: str | None = None

    def transition(self, status: RunStatus, detail: str = "") -> None:
        self.status = status
        self.status_history.append(StatusEvent(status=status, detail=detail))

    def add_trace(self, event: TraceEvent) -> None:
        self.trace.append(event)
        self.total_usage.input_tokens += event.usage.input_tokens
        self.total_usage.output_tokens += event.usage.output_tokens
        self.total_usage.estimated_cost_usd += event.usage.estimated_cost_usd

    @field_validator("citations")
    @classmethod
    def citations_reference_known_ids(cls, citations: list[Citation]) -> list[Citation]:
        markers = [citation.marker for citation in citations]
        if len(markers) != len(set(markers)):
            raise ValueError("citation markers must be unique")
        return citations
