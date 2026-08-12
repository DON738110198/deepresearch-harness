from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Protocol

from .contracts import Citation, Claim, Evidence, Plan, RunState, RunStatus, Task, TraceEvent
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


class BaselineResearchPipeline:
    def __init__(self, *, provider: LLMProvider, collector: EvidenceCollector, output_dir: Path, max_evidence: int = 6) -> None:
        self._provider = provider
        self._collector = collector
        self._output_dir = output_dir
        self._max_evidence = max_evidence

    def run(self, question: str) -> RunState:
        run_id = uuid.uuid4().hex
        state = RunState(run_id=run_id, task=Task(id=f"task-{run_id[:8]}", question=question))
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
            state.claims = self._build_ledger(state.evidence, state)
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
        prompt = f"Create JSON with steps (id, objective) and search_queries for this question: {question}"
        completion = self._complete("plan", prompt, state)
        return Plan.model_validate_json(completion)

    def _build_ledger(self, evidence: list[Evidence], state: RunState) -> list[Claim]:
        prompt = json.dumps({"evidence": [item.model_dump(mode="json") for item in evidence]})
        completion = self._complete("ledger", prompt, state)
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
                "question": question,
                "claims": [claim.model_dump() for claim in run.claims],
                "citations": markers,
                "evidence": [item.model_dump(mode="json") for item in run.evidence],
            }
        )
        return self._complete("write", prompt, state)

    def _complete(self, stage: str, prompt: str, state: RunState) -> str:
        try:
            completion = self._provider.complete(stage=stage, prompt=prompt)
            state.add_trace(
                TraceEvent(stage=stage, provider=self._provider.name, model=self._provider.model, latency_ms=completion.latency_ms, usage=completion.usage, outcome="ok")
            )
            return completion.text
        except Exception as error:
            state.add_trace(TraceEvent(stage=stage, provider=self._provider.name, model=self._provider.model, latency_ms=0, outcome="error", detail=str(error)))
            raise

    def _add_collector_trace(self, state: RunState, count: int) -> None:
        state.add_trace(TraceEvent(stage="collect", provider="local_corpus", model="lexical-v1", latency_ms=0, outcome="ok", detail=f"collected={count}"))

    @staticmethod
    def _citations_for(claims: list[Claim]) -> list[Citation]:
        citations: list[Citation] = []
        for index, claim in enumerate(claims, start=1):
            citations.append(Citation(id=f"citation-{index}", claim_id=claim.id, evidence_id=claim.evidence_ids[0], marker=f"[{index}]"))
        return citations

