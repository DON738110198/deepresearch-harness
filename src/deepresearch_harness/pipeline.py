from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Protocol

from .contracts import BudgetLimits, Citation, Claim, DirectWriteDraft, Evidence, Plan, RunState, RunStatus, Task, TraceEvent
from .providers import LLMProvider
from .storage import RunStore


class EvidenceCollector(Protocol):
    def collect(self, queries: list[str], max_evidence: int) -> list[Evidence]: ...


class LocalCorpusCollector:
    """Deterministic lexical collector for an explicitly supplied corpus snapshot."""

    def __init__(self, corpus_path: Path) -> None:
        self._records = json.loads(corpus_path.read_text(encoding="utf-8"))

    def collect(self, queries: list[str], max_evidence: int) -> list[Evidence]:
        terms = set(re.findall(r"[a-z0-9]+", " ".join(queries).lower()))
        scored: list[tuple[int, dict[str, str], str]] = []
        for item in self._records:
            haystack = f"{item['title']} {item['snippet']}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, item, queries[0]))
        scored.sort(key=lambda row: (-row[0], row[1]["id"]))
        return [
            Evidence(id=item["id"], title=item["title"], url=item["url"], excerpt=item["snippet"], query=query)
            for _, item, query in scored[:max_evidence]
        ]


class BudgetExceeded(RuntimeError):
    pass


class BaselineResearchPipeline:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        collector: EvidenceCollector,
        output_dir: Path,
        max_evidence: int = 6,
        budget_limits: BudgetLimits | None = None,
    ) -> None:
        self._provider = provider
        self._collector = collector
        self._output_dir = output_dir
        self._max_evidence = max_evidence
        self._budget_limits = budget_limits or BudgetLimits()

    def run(self, question: str) -> RunState:
        run_id = uuid.uuid4().hex
        state = RunState(
            run_id=run_id,
            task=Task(id=f"task-{run_id[:8]}", question=question),
            variant="b1_plan_search_ledger_write",
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
                    "Omit topical distractors. Return json only and do not add facts."
                ),
                "question": question,
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
        claims = [Claim.model_validate(item) for item in json.loads(completion)["claims"]]
        evidence_ids = {item.id for item in evidence}
        for claim in claims:
            if not set(claim.evidence_ids).issubset(evidence_ids):
                raise ValueError(f"claim {claim.id} cites evidence outside this run")
        return claims

    def _write_report(self, question: str, run: RunState, state: RunState) -> str:
        markers = {citation.claim_id: citation.marker for citation in run.citations}
        prompt = json.dumps(
            {
                "instruction": "Write a concise markdown decision report using only the supplied claims. Preserve each citation marker after its claim.",
                "question": question,
                "claims": [claim.model_dump() for claim in run.claims],
                "citations": markers,
                "evidence": [item.model_dump(mode="json") for item in run.evidence],
            }
        )
        return self._complete("write", prompt, state)

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
        state.add_trace(TraceEvent(stage="collect", provider="local_corpus", model="lexical-v1", latency_ms=0, outcome="ok", detail=f"collected={count}"))

    @staticmethod
    def _citations_for(claims: list[Claim]) -> list[Citation]:
        citations: list[Citation] = []
        for index, claim in enumerate(claims, start=1):
            citations.append(Citation(id=f"citation-{index}", claim_id=claim.id, evidence_id=claim.evidence_ids[0], marker=f"[{index}]"))
        return citations


class SearchWritePipeline(BaselineResearchPipeline):
    """B0: direct question search followed by one structured report call."""

    def run(self, question: str) -> RunState:
        run_id = uuid.uuid4().hex
        state = RunState(
            run_id=run_id,
            task=Task(id=f"task-{run_id[:8]}", question=question),
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
                    "Do not create numeric citation markers; the harness compiles them deterministically."
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
