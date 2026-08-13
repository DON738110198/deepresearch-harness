from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue, model_validator

from .contracts import (
    BudgetLimits,
    Citation,
    Claim,
    DirectWriteDraft,
    Evidence,
    EvidenceDebt,
    Plan,
    PlannedObligation,
    PlanStep,
    RunState,
    RunStatus,
    Task,
    TraceEvent,
)
from .providers import LLMProvider
from .storage import RunStore


class EvidenceCollector(Protocol):
    def collect(self, queries: list[str], max_evidence: int) -> list[Evidence]: ...


class LocalCorpusCollector:
    """Deterministic lexical collector for an explicitly supplied corpus snapshot."""

    def __init__(self, corpus_path: Path) -> None:
        self._records = json.loads(corpus_path.read_text(encoding="utf-8"))

    def collect(self, queries: list[str], max_evidence: int) -> list[Evidence]:
        ranked_by_query = [self._rank_query(query) for query in queries]
        selected: list[tuple[dict[str, str], str]] = []
        selected_ids: set[str] = set()
        next_rank = [0] * len(ranked_by_query)
        while len(selected) < max_evidence:
            added = False
            for query_index, rows in enumerate(ranked_by_query):
                while next_rank[query_index] < len(rows) and rows[next_rank[query_index]][1]["id"] in selected_ids:
                    next_rank[query_index] += 1
                if next_rank[query_index] >= len(rows):
                    continue
                _, item, query = rows[next_rank[query_index]]
                next_rank[query_index] += 1
                selected.append((item, query))
                selected_ids.add(item["id"])
                added = True
                if len(selected) == max_evidence:
                    break
            if not added:
                break
        return [
            Evidence(id=item["id"], title=item["title"], url=item["url"], excerpt=item["snippet"], query=query)
            for item, query in selected
        ]

    def _rank_query(self, query: str) -> list[tuple[int, dict[str, str], str]]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[int, dict[str, str], str]] = []
        for item in self._records:
            haystack = f"{item['title']} {item['snippet']}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, item, query))
        scored.sort(key=lambda row: (-row[0], row[1]["id"]))
        return scored


class BudgetExceeded(RuntimeError):
    pass


class ObligationPlanDraft(BaseModel):
    steps: list[PlanStep] = Field(
        default_factory=lambda: [
            PlanStep(id="answer-contract", objective="Resolve each planned answer obligation with direct evidence.")
        ],
        min_length=1,
    )
    obligations: list[PlannedObligation] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def obligation_ids_and_queries_are_unique(self) -> "ObligationPlanDraft":
        ids = [item.id for item in self.obligations]
        queries = [item.search_query.casefold() for item in self.obligations]
        if len(ids) != len(set(ids)):
            raise ValueError("obligation plan ids must be unique")
        if len(queries) != len(set(queries)):
            raise ValueError("each obligation requires a distinct search query")
        return self


class BenchmarkAnswerDraft(BaseModel):
    answer: JsonValue


class BenchmarkPlanDraft(BaseModel):
    steps: list[PlanStep] = Field(
        default_factory=lambda: [
            PlanStep(id="identify", objective="Find sources satisfying every stated benchmark constraint.")
        ],
        min_length=1,
    )
    search_queries: list[str] = Field(min_length=1, max_length=5)


class BaselineResearchPipeline:
    variant = "b1_plan_search_ledger_write"

    def __init__(
        self,
        *,
        provider: LLMProvider,
        collector: EvidenceCollector,
        output_dir: Path,
        max_evidence: int = 6,
        budget_limits: BudgetLimits | None = None,
        report_language: str = "auto",
    ) -> None:
        self._provider = provider
        self._collector = collector
        self._output_dir = output_dir
        self._max_evidence = max_evidence
        self._budget_limits = budget_limits or BudgetLimits()
        self._report_language = report_language

    def run(self, question: str, *, decision_context: str | None = None) -> RunState:
        run_id = uuid.uuid4().hex
        state = RunState(
            run_id=run_id,
            task=Task(id=f"task-{run_id[:8]}", question=question, decision_context=decision_context),
            variant=self.variant,
            budget_limits=self._budget_limits.model_copy(deep=True),
        )
        store = RunStore(self._output_dir, run_id)
        state.transition(RunStatus.RUNNING, "baseline started")
        store.save(state)
        try:
            state.plan = self._plan(question, state)
            store.save(state)
            state.evidence = self._collector.collect(state.plan.search_queries, self._max_evidence)
            self._add_collector_trace(state, len(state.evidence))
            if not state.evidence:
                raise RuntimeError("collector returned no evidence for the planned query")
            store.save(state)
            state.claims = self._build_ledger(question, state.evidence, state)
            state.citations = self._citations_for(state.claims)
            store.save(state)
            report = self._write_report(question, state, state)
            state.report_path = str(store.write_report(report))
            state.transition(RunStatus.SUCCEEDED, "baseline completed")
            store.save(state)
            return state
        except Exception as error:
            state.error = str(error)
            state.transition(RunStatus.FAILED, "baseline failed")
            store.save(state)
            raise

    def _plan(self, question: str, state: RunState) -> Plan:
        prompt = json.dumps(
            {
                "instruction": "Create a concise research plan. Return json only.",
                "question": question,
                "decision_context": state.task.decision_context,
                "json_example": {
                    "steps": [{"id": "scope", "objective": "Identify required evidence."}],
                    "search_queries": ["one focused evidence query"],
                },
            }
        )
        completion = self._complete("plan", prompt, state, json_output=True)
        return Plan.model_validate_json(completion)

    def _build_ledger(self, question: str, evidence: list[Evidence], state: RunState) -> list[Claim]:
        prompt = json.dumps(
            {
                "instruction": (
                    "Select only evidence that directly helps answer the question, then convert it into atomic claims. "
                    "Omit topical distractors. Honor definitions and existing capabilities in the decision context. "
                    "Do not recommend a capability that the context says is already implemented. Return json only and do not add facts."
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "json_example": {
                    "claims": [
                        {
                            "id": "claim-evidence-id",
                            "text": "Exact evidence-backed claim.",
                            "evidence_ids": ["evidence-id"],
                            "support": "direct",
                        }
                    ]
                },
                "evidence": [item.model_dump(mode="json") for item in evidence],
            }
        )
        completion = self._complete("ledger", prompt, state, json_output=True)
        payload = json.loads(completion)
        claim_rows = payload.get("claims", [])
        if not isinstance(claim_rows, list):
            raise ValueError("claim ledger must contain a claims array")
        claims = [Claim.model_validate(item) for item in claim_rows]
        evidence_ids = {item.id for item in evidence}
        for claim in claims:
            if not set(claim.evidence_ids).issubset(evidence_ids):
                raise ValueError(f"claim {claim.id} cites evidence outside this run")
        return claims

    def _write_report(self, question: str, run: RunState, state: RunState) -> str:
        markers = {citation.claim_id: citation.marker for citation in run.citations}
        prompt = json.dumps(
            {
                "instruction": (
                    "Write a concise markdown decision report using only the supplied claims. "
                    "Preserve each citation marker after its claim. Do not write a references section because "
                    "the harness appends it deterministically. If the question asks for an axis that the supplied claims "
                    "do not support, state that this run did not find evidence; never relabel a nearby concept as that axis. "
                    "Separate source-backed findings from recommendations inferred from the decision context. Label any "
                    "unmeasured expected effect as planned rather than claiming improvement. "
                    f"{self._language_instruction()}"
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "claims": [claim.model_dump() for claim in run.claims],
                "citations": markers,
                "evidence": [item.model_dump(mode="json") for item in run.evidence],
            }
        )
        report = self._complete("write", prompt, state)
        return self._finalize_report(report, run)

    def _complete(self, stage: str, prompt: str, state: RunState, *, json_output: bool = False) -> str:
        try:
            max_output_tokens = self._next_call_output_limit(state)
        except BudgetExceeded as error:
            state.stop_reason = "budget_exhausted"
            state.add_trace(
                TraceEvent(
                    stage=stage,
                    provider=self._provider.name,
                    model=self._provider.model,
                    latency_ms=0,
                    outcome="error",
                    detail=str(error),
                )
            )
            raise
        try:
            completion = self._provider.complete(
                stage=stage,
                prompt=prompt,
                json_output=json_output,
                max_output_tokens=max_output_tokens,
            )
            state.add_trace(
                TraceEvent(stage=stage, provider=self._provider.name, model=self._provider.model, latency_ms=completion.latency_ms, usage=completion.usage, outcome="ok")
            )
            self._enforce_observed_budget(state)
            return completion.text
        except BudgetExceeded:
            raise
        except Exception as error:
            state.add_trace(TraceEvent(stage=stage, provider=self._provider.name, model=self._provider.model, latency_ms=0, outcome="error", detail=str(error)))
            raise

    def _next_call_output_limit(self, state: RunState) -> int:
        limits = state.budget_limits
        llm_calls = sum(event.provider == self._provider.name and event.outcome == "ok" for event in state.trace)
        if llm_calls >= limits.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget exhausted before stage {llm_calls + 1}")
        if limits.max_estimated_cost_usd is not None and state.total_usage.estimated_cost_usd >= limits.max_estimated_cost_usd:
            raise BudgetExceeded("estimated cost budget exhausted before the next LLM call")
        output_limit = limits.max_output_tokens_per_call
        if limits.max_total_tokens is not None:
            used_tokens = state.total_usage.input_tokens + state.total_usage.output_tokens
            remaining_tokens = limits.max_total_tokens - used_tokens
            if remaining_tokens <= 0:
                raise BudgetExceeded("observed token budget exhausted before the next LLM call")
            output_limit = min(output_limit, remaining_tokens)
        return output_limit

    @staticmethod
    def _enforce_observed_budget(state: RunState) -> None:
        limits = state.budget_limits
        total_tokens = state.total_usage.input_tokens + state.total_usage.output_tokens
        if limits.max_total_tokens is not None and total_tokens > limits.max_total_tokens:
            state.stop_reason = "budget_exhausted"
            raise BudgetExceeded(f"observed token budget exceeded: {total_tokens} > {limits.max_total_tokens}")
        if limits.max_estimated_cost_usd is not None and state.total_usage.estimated_cost_usd > limits.max_estimated_cost_usd:
            state.stop_reason = "budget_exhausted"
            raise BudgetExceeded(
                "estimated cost budget exceeded: "
                f"{state.total_usage.estimated_cost_usd:.8f} > {limits.max_estimated_cost_usd:.8f}"
            )

    def _add_collector_trace(self, state: RunState, count: int) -> None:
        for event in getattr(self._collector, "trace_events", []):
            state.add_trace(event.model_copy(deep=True))
        state.add_trace(
            TraceEvent(
                stage="collect",
                provider=type(self._collector).__name__,
                model="collector-v1",
                latency_ms=0,
                outcome="ok",
                detail=f"collected={count}",
            )
        )

    def _language_instruction(self) -> str:
        if self._report_language == "auto":
            return "Write in the same language as the user question."
        return f"Write the report in {self._report_language}."

    def _finalize_report(self, report: str, run: RunState) -> str:
        for citation in run.citations:
            if citation.marker not in report:
                raise ValueError(f"report omitted citation marker {citation.marker}")
        evidence_by_id = {item.id: item for item in run.evidence}
        heading = "## 来源" if self._report_language == "Simplified Chinese" else "## Sources"
        lines = [report.rstrip(), "", heading, ""]
        citations_by_evidence: dict[str, list[Citation]] = {}
        for citation in run.citations:
            citations_by_evidence.setdefault(citation.evidence_id, []).append(citation)
        for evidence_id, citations in citations_by_evidence.items():
            evidence = evidence_by_id[evidence_id]
            title = evidence.title.replace("[", "(").replace("]", ")")
            markers = " ".join(citation.marker for citation in citations)
            lines.append(f"- {markers} [{title}]({evidence.url})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _citations_for(claims: list[Claim]) -> list[Citation]:
        citations: list[Citation] = []
        for index, claim in enumerate(claims, start=1):
            citations.append(Citation(id=f"citation-{index}", claim_id=claim.id, evidence_id=claim.evidence_ids[0], marker=f"[{index}]"))
        return citations


class PrimarySourceResearchPipeline(BaselineResearchPipeline):
    """B1 live-search policy that favors one primary-source query per named entity."""

    variant = "b1_live_primary_sources"

    def _plan(self, question: str, state: RunState) -> Plan:
        prompt = json.dumps(
            {
                "instruction": (
                    "Create a concise public-web research plan. Return json only with 2-5 distinct search queries. "
                    "When the question names projects, products, or organizations, create one focused query for each "
                    "named entity and prefer its official repository or documentation. Keep the exact English entity "
                    "name in the query and include all evidence axes explicitly requested by the question, such as workflow, "
                    "benchmark, metrics, or evaluation. Do not search for conclusions or design implications; derive those "
                    "later from the collected primary sources."
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "json_example": {
                    "steps": [{"id": "primary-sources", "objective": "Collect official sources for each named entity."}],
                    "search_queries": ["Exact Project Name official repository workflow"],
                },
            }
        )
        completion = self._complete("plan", prompt, state, json_output=True)
        plan = Plan.model_validate_json(completion)
        if not 2 <= len(plan.search_queries) <= 5:
            raise ValueError("live primary-source plan requires 2-5 search queries")
        return plan


class BenchmarkResearchPipeline(BaselineResearchPipeline):
    """One-pass B1 with task-defined structured output for public benchmark adapters."""

    variant = "b1_benchmark_structured"

    def _plan(self, question: str, state: RunState) -> Plan:
        prompt = json.dumps(
            {
                "instruction": (
                    "Create a concise public-web research plan for this benchmark question. Return JSON only with 1-5 "
                    "distinct, focused search queries. Cover the question's identifying constraints and requested fields. "
                    "Do not search for the benchmark answer or benchmark ground truth."
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "json_example": {
                    "steps": [{"id": "identify", "objective": "Find sources satisfying every stated constraint."}],
                    "search_queries": ["focused public evidence query"],
                },
            }
        )
        completion = self._complete("plan", prompt, state, json_output=True)
        draft = BenchmarkPlanDraft.model_validate_json(completion)
        return Plan(steps=draft.steps, search_queries=draft.search_queries)

    def _write_report(self, question: str, run: RunState, state: RunState) -> str:
        prompt = json.dumps(
            {
                "instruction": (
                    "Return JSON only with one top-level field named answer. The answer value must follow the exact JSON "
                    "shape and field names requested by the benchmark question. Use only the supplied evidence-backed claims. "
                    "When evidence is incomplete, return the supported subset rather than inventing entries. Do not include "
                    "markdown, citation markers, commentary, or a references section inside answer."
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "claims": [claim.model_dump() for claim in run.claims],
                "evidence": [item.model_dump(mode="json") for item in run.evidence],
                "json_example": {"answer": []},
            }
        )
        completion = self._complete("benchmark_write", prompt, state, json_output=True)
        draft = BenchmarkAnswerDraft.model_validate_json(completion)
        run.structured_answer = draft.answer
        answer_json = json.dumps(draft.answer, ensure_ascii=False, indent=2)
        markers = {citation.claim_id: citation.marker for citation in run.citations}
        lines = ["# Benchmark answer", "", "```json", answer_json, "```", "", "## Evidence ledger", ""]
        for claim in run.claims:
            lines.append(f"- {claim.text} {markers[claim.id]}")
        return self._finalize_report("\n".join(lines), run)


class SearchWritePipeline(BaselineResearchPipeline):
    """B0: direct question search followed by one structured report call."""

    def run(self, question: str, *, decision_context: str | None = None) -> RunState:
        run_id = uuid.uuid4().hex
        state = RunState(
            run_id=run_id,
            task=Task(id=f"task-{run_id[:8]}", question=question, decision_context=decision_context),
            variant="b0_search_write",
            budget_limits=self._budget_limits.model_copy(deep=True),
        )
        store = RunStore(self._output_dir, run_id)
        state.transition(RunStatus.RUNNING, "B0 Search-Write started")
        store.save(state)
        try:
            state.evidence = self._collector.collect([question], self._max_evidence)
            self._add_collector_trace(state, len(state.evidence))
            if not state.evidence:
                raise RuntimeError("collector returned no evidence for the user question")
            store.save(state)
            draft = self._direct_write(question, state)
            known_evidence = {evidence.id for evidence in state.evidence}
            if any(not set(claim.evidence_ids).issubset(known_evidence) for claim in draft.claims):
                raise ValueError("direct-write claim cites evidence outside this run")
            state.claims = draft.claims
            state.citations = self._citations_for(draft.claims)
            report = self._compile_direct_report(draft, state.citations)
            report = self._finalize_report(report, state)
            state.report_path = str(store.write_report(report))
            state.transition(RunStatus.SUCCEEDED, "B0 Search-Write completed")
            store.save(state)
            return state
        except Exception as error:
            state.error = str(error)
            state.transition(RunStatus.FAILED, "B0 Search-Write failed")
            store.save(state)
            raise

    def _direct_write(self, question: str, state: RunState) -> DirectWriteDraft:
        prompt = json.dumps(
            {
                "instruction": (
                    "Answer the question from the supplied evidence in one pass. Return json only with claims and a markdown report. "
                    "Every material claim must use evidence_ids from the input. Cite each claim in the report with the exact placeholder [[claim-id]]. "
                    "Honor definitions and existing capabilities in the decision context. Do not recommend an already implemented "
                    "capability. Do not create numeric citation markers; the harness compiles them deterministically."
                ),
                "json_example": {
                    "claims": [
                        {
                            "id": "claim-1",
                            "text": "Evidence-backed claim.",
                            "evidence_ids": ["evidence-id"],
                            "support": "direct",
                        }
                    ],
                    "report": "# Research report\\n\\nEvidence-backed claim. [[claim-1]]",
                },
                "question": question,
                "decision_context": state.task.decision_context,
                "evidence": [item.model_dump(mode="json") for item in state.evidence],
            }
        )
        completion = self._complete("direct_write", prompt, state, json_output=True)
        return DirectWriteDraft.model_validate_json(completion)

    @staticmethod
    def _compile_direct_report(draft: DirectWriteDraft, citations: list[Citation]) -> str:
        report = draft.report
        for citation in citations:
            placeholder = f"[[{citation.claim_id}]]"
            if placeholder not in report:
                raise ValueError(f"direct-write report omitted placeholder {placeholder}")
            report = report.replace(placeholder, citation.marker)
        return report


class ObligationEvidenceDebtPipeline(BaselineResearchPipeline):
    """B2: make answer obligations and unresolved evidence needs explicit without another LLM call."""

    variant = "b2_obligation_evidence_debt"

    def _plan(self, question: str, state: RunState) -> Plan:
        prompt = json.dumps(
            {
                "instruction": (
                    "Create an exhaustive but concise answer contract for the question. Return json only. "
                    "Hard limit: return 3-5 obligations total and never exceed five. Define non-overlapping, "
                    "domain-specific evidence questions. Cover every distinct workflow stage, "
                    "comparison axis, failure mode, and hard constraint stated or directly implied by the question and decision "
                    "context. Use the input's concrete concepts in obligation descriptions and queries. Do not use generic "
                    "benefit, risk, constraint, or trade-off slots unless the input explicitly asks for those categories. "
                    "When the question names multiple projects, products, or organizations, create one source obligation "
                    "for each named entity before adding cross-cutting obligations. Derive comparisons from those primary "
                    "sources instead of searching for a third-party comparison page. Keep each public-web query focused "
                    "on one entity and one information need, preferably its official repository or documentation, and use "
                    "4-10 searchable terms. Do not create a separate comparison obligation; the writer derives comparisons "
                    "from the named entities' primary sources. If the requested comparison includes design implications, "
                    "do not create a separate search obligation for them; the writer must synthesize them from the named "
                    "entities' primary sources. Combine secondary evaluation needs into remaining slots only when necessary. "
                    "Each obligation must ask for evidence that can substantively inform one answer component; do not require "
                    "a source to state the final recommendation. Do not invent legal, regulatory, policy, product, user-preference, "
                    "or metric requirements unless the input names them."
                    " Search queries should work on the public web; retain concrete English product and project names "
                    "when they are more searchable than translations."
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "json_example": {
                    "steps": [{"id": "contract", "objective": "Identify every answer obligation."}],
                    "obligations": [
                        {
                            "id": "criterion-id",
                            "description": "Decision criterion that requires evidence.",
                            "search_query": "question subject criterion evidence",
                        }
                    ],
                },
            }
        )
        completion = self._complete("plan", prompt, state, json_output=True)
        draft = ObligationPlanDraft.model_validate_json(completion)
        return Plan(
            steps=draft.steps,
            obligations=draft.obligations,
            search_queries=[item.search_query for item in draft.obligations],
        )

    def _build_ledger(self, question: str, evidence: list[Evidence], state: RunState) -> list[Claim]:
        if state.plan is None or not state.plan.obligations:
            raise ValueError("B2 requires planned answer obligations")
        prompt = json.dumps(
            {
                "instruction": (
                    "Build an obligation-linked claim ledger from the supplied evidence. Return json only. "
                    "For every obligation, emit exactly one evidence_debt entry. Mark it resolved only when selected evidence "
                    "directly supports one or more atomic claims; otherwise mark it open and explain the missing evidence. "
                    "Omit topical distractors, honor definitions and existing capabilities in the decision context, and do not add facts."
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "obligations": [item.model_dump() for item in state.plan.obligations],
                "json_example": {
                    "claims": [
                        {
                            "id": "claim-evidence-id",
                            "text": "Exact evidence-backed claim.",
                            "evidence_ids": ["evidence-id"],
                            "support": "direct",
                        }
                    ],
                    "evidence_debts": [
                        {
                            "obligation_id": "criterion-id",
                            "status": "resolved",
                            "evidence_ids": ["evidence-id"],
                            "claim_ids": ["claim-evidence-id"],
                            "detail": "Direct evidence found.",
                        }
                    ],
                },
                "evidence": [item.model_dump(mode="json") for item in evidence],
            }
        )
        payload = json.loads(self._complete("ledger", prompt, state, json_output=True))
        claims = [Claim.model_validate(item) for item in payload["claims"]]
        debts = [EvidenceDebt.model_validate(item) for item in payload["evidence_debts"]]
        self._validate_debt_ledger(state.plan, evidence, claims, debts)
        state.evidence_debts = debts
        referenced_claim_ids = {claim_id for debt in debts for claim_id in debt.claim_ids}
        return [claim for claim in claims if claim.id in referenced_claim_ids]

    def _write_report(self, question: str, run: RunState, state: RunState) -> str:
        markers = {citation.claim_id: citation.marker for citation in run.citations}
        prompt = json.dumps(
            {
                "instruction": (
                    "Write a concise markdown decision report using only resolved supplied claims. "
                    "Preserve each citation marker after its claim. Do not silently invent support for open evidence debt. "
                    "Never relabel a nearby concept as a missing requested axis. Label unmeasured expected effects as planned. "
                    "Do not render an evidence-debt or references section; the harness appends both deterministically. "
                    f"{self._language_instruction()}"
                ),
                "question": question,
                "decision_context": state.task.decision_context,
                "claims": [claim.model_dump() for claim in run.claims],
                "citations": markers,
                "evidence_debts": [debt.model_dump() for debt in run.evidence_debts],
            }
        )
        report = self._complete("write", prompt, state)
        open_debts = [debt for debt in run.evidence_debts if debt.status == "open"]
        if not open_debts:
            return self._finalize_report(report, run)
        obligation_by_id = {item.id: item for item in run.plan.obligations} if run.plan else {}
        chinese = self._report_language == "Simplified Chinese"
        lines = [report.rstrip(), "", "## 待补证据" if chinese else "## Open evidence debt", ""]
        for debt in open_debts:
            description = obligation_by_id[debt.obligation_id].description
            suffix = "本次运行未找到直接支持证据。" if chinese else "No directly supporting evidence was identified in this run."
            lines.append(f"- **{debt.obligation_id}:** {description} {suffix}")
        return self._finalize_report("\n".join(lines), run)

    @staticmethod
    def _validate_debt_ledger(
        plan: Plan,
        evidence: list[Evidence],
        claims: list[Claim],
        debts: list[EvidenceDebt],
    ) -> None:
        obligation_ids = {item.id for item in plan.obligations}
        debt_ids = {item.obligation_id for item in debts}
        if debt_ids != obligation_ids or len(debts) != len(obligation_ids):
            raise ValueError("evidence debt ledger must classify every planned obligation exactly once")
        evidence_ids = {item.id for item in evidence}
        claim_by_id = {item.id: item for item in claims}
        if len(claim_by_id) != len(claims):
            raise ValueError("claim ids must be unique")
        for claim in claims:
            if not set(claim.evidence_ids).issubset(evidence_ids):
                raise ValueError(f"claim {claim.id} cites evidence outside this run")
        for debt in debts:
            if not set(debt.evidence_ids).issubset(evidence_ids):
                raise ValueError(f"evidence debt {debt.obligation_id} cites evidence outside this run")
            if not set(debt.claim_ids).issubset(claim_by_id):
                raise ValueError(f"evidence debt {debt.obligation_id} cites an unknown claim")
            linked_evidence = {
                evidence_id
                for claim_id in debt.claim_ids
                for evidence_id in claim_by_id[claim_id].evidence_ids
            }
            if debt.status == "resolved" and not set(debt.evidence_ids).issubset(linked_evidence):
                raise ValueError(f"resolved evidence debt {debt.obligation_id} is not linked through its claims")
