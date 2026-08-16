from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .atomic_clue_frontier import AtomicClueCaseResult, AtomicClueResult
from .contracts import Usage
from .hypothesis_firewall import FirewallProbeResult
from .providers import LLMProvider


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaseSourceReference(StrictContract):
    query_id: str = Field(min_length=1)
    request: ArtifactReference
    run: ArtifactReference


class CorpusBridgeFixedContract(StrictContract):
    model: str = Field(min_length=1)
    thinking_mode: Literal["disabled"] = "disabled"
    system_prompt_policy: str = Field(min_length=1)
    retriever_id: str = Field(min_length=1)
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    context_fields: tuple[
        Literal[
            "question",
            "unresolved_obligation",
            "unverified_prior_subject",
            "near_miss_packet",
        ],
        ...,
    ]
    forbidden_context_fields: tuple[
        Literal["draft_answer", "prior_exact_answer", "prior_query", "gold_docids"],
        ...,
    ]
    documents_per_clue: Literal[2] = 2
    snippet_character_cap: Literal[1200] = 1200
    bridges_per_case: Literal[3] = 3
    prior_search_calls: int = Field(ge=0)
    final_search_calls_per_case: Literal[3] = 3
    maximum_cumulative_search_calls: int = Field(ge=1)
    max_search_results: int = Field(ge=1, le=100)
    max_planner_output_tokens: int = Field(ge=1, le=512)
    provider_attempts_per_case: Literal[1] = 1
    sealed_holdout_access: Literal["forbidden"] = "forbidden"

    @model_validator(mode="after")
    def context_and_budget_are_exact(self) -> "CorpusBridgeFixedContract":
        expected = (
            "question",
            "unresolved_obligation",
            "unverified_prior_subject",
            "near_miss_packet",
        )
        if self.context_fields != expected:
            raise ValueError("corpus bridge context boundary changed")
        forbidden = {"draft_answer", "prior_exact_answer", "prior_query", "gold_docids"}
        if set(self.forbidden_context_fields) != forbidden:
            raise ValueError("corpus bridge forbidden context is incomplete")
        return self


class CorpusBridgeAcceptance(StrictContract):
    minimum_candidate_gold_hit_cases: int = Field(ge=1)
    minimum_gold_hit_case_delta: int = Field(ge=1)
    parse_failures_must_equal: Literal[0] = 0
    provider_request_failures_must_equal: Literal[0] = 0
    search_failures_must_equal: Literal[0] = 0
    maximum_provider_cost_usd: float = Field(gt=0)


class CorpusBridgeRegistration(StrictContract):
    schema_version: Literal["corpus-grounded-bridge-registration-v0"] = (
        "corpus-grounded-bridge-registration-v0"
    )
    status: Literal["registered_pre_provider"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    frontier_source: ArtifactReference
    baseline_probe: ArtifactReference
    case_sources: tuple[CaseSourceReference, ...] = Field(min_length=1)
    gold_slice: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    fixed_contract: CorpusBridgeFixedContract
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: CorpusBridgeAcceptance
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def ids_and_gates_match_cases(self) -> "CorpusBridgeRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("corpus bridge IDs must be unique")
        if tuple(item.query_id for item in self.case_sources) != self.query_ids:
            raise ValueError("case source order must exactly match query IDs")
        if self.acceptance.minimum_candidate_gold_hit_cases > len(self.query_ids):
            raise ValueError("corpus bridge gold-hit gate exceeds registered cases")
        expected_total = (
            self.fixed_contract.prior_search_calls
            + len(self.query_ids) * self.fixed_contract.final_search_calls_per_case
        )
        if expected_total != self.fixed_contract.maximum_cumulative_search_calls:
            raise ValueError("corpus bridge cumulative search cap is inconsistent")
        return self


class NearMissDocument(StrictContract):
    docid: str
    score: float
    snippet: str


class NearMissGroup(StrictContract):
    clue_query: str
    documents: tuple[NearMissDocument, ...] = Field(min_length=1, max_length=2)


class CorpusBridge(StrictContract):
    type: Literal["entity", "domain", "topic", "geography", "work", "event"]
    term: str = Field(min_length=1, max_length=80)
    source_docids: tuple[str, ...] = Field(min_length=1, max_length=3)
    query: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def bridge_is_compact(self) -> "CorpusBridge":
        term_tokens = _tokens(self.term)
        if not 1 <= len(term_tokens) <= 5:
            raise ValueError("corpus bridge term must contain one to five terms")
        query_tokens = set(_tokens(self.query))
        if not 5 <= len(_tokens(self.query)) <= 18:
            raise ValueError("corpus bridge query must contain 5 to 18 terms")
        if any(token not in query_tokens for token in term_tokens):
            raise ValueError("corpus bridge query omits its bridge term")
        if len(self.source_docids) != len(set(self.source_docids)):
            raise ValueError("corpus bridge source docids must be unique")
        return self


class CorpusBridgeSlate(StrictContract):
    bridges: tuple[CorpusBridge, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def bridge_terms_are_distinct(self) -> "CorpusBridgeSlate":
        signatures = tuple(_tokens(item.term) for item in self.bridges)
        if len(set(signatures)) != 3:
            raise ValueError("corpus bridge terms must be distinct")
        return self


class CorpusBridgeCase(StrictContract):
    query_id: str
    question: str
    unresolved_obligation: str
    unverified_prior_subject: str
    near_miss_packet: tuple[NearMissGroup, ...] = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)


class CorpusBridgeSearch(StrictContract):
    bridge_index: int = Field(ge=0, le=2)
    term: str
    query: str
    returned_docids: tuple[str, ...]
    gold_hits: tuple[str, ...]
    gold_hit_ranks: tuple[int, ...]
    latency_ms: int = Field(ge=0)


class CorpusBridgeItem(StrictContract):
    query_id: str
    status: Literal["succeeded", "failed"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_fields: tuple[str, ...]
    baseline_gold_hits: tuple[str, ...]
    slate: CorpusBridgeSlate | None
    searches: tuple[CorpusBridgeSearch, ...]
    case_gold_hits: tuple[str, ...]
    raw_completion_sha256: str | None
    raw_completion_text: str | None
    usage: Usage
    provider_latency_ms: int = Field(ge=0)
    error: str | None


class DecisionGate(StrictContract):
    gate_id: str
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class CorpusBridgeResult(StrictContract):
    schema_version: Literal["corpus-grounded-bridge-probe-v0"] = (
        "corpus-grounded-bridge-probe-v0"
    )
    created_at: str
    status: Literal["succeeded", "failed"]
    decision: Literal["pass", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    provider_attempts: int = Field(ge=0)
    prior_search_calls: int = Field(ge=0)
    final_search_calls: int = Field(ge=0)
    cumulative_search_calls: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    provider_request_failures: int = Field(ge=0)
    search_failures: int = Field(ge=0)
    baseline_gold_hit_cases: int = Field(ge=0)
    candidate_gold_hit_cases: int = Field(ge=0)
    gold_hit_case_delta: int
    total_usage: Usage
    provider_cost_observability: Literal["complete", "lower_bound_after_failure"]
    context_isolation_verified: bool
    items: tuple[CorpusBridgeItem, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "preregister_corpus_bridge_stage_replay",
        "close_without_end_to_end_expansion",
    ]
    claim_boundary: str


def _validate_artifacts(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")


def load_corpus_bridge_registration(path: Path) -> CorpusBridgeRegistration:
    registration = CorpusBridgeRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    case_artifacts = tuple(
        artifact
        for source in registration.case_sources
        for artifact in (source.request, source.run)
    )
    _validate_artifacts(
        root,
        (
            registration.frontier_source,
            registration.baseline_probe,
            *case_artifacts,
            registration.gold_slice,
            *registration.frozen_artifacts,
        ),
    )
    return registration


def _obligation_for_question(question: str) -> str:
    lowered = question.casefold()
    if "first name and surname" in lowered or lowered.lstrip().startswith("who"):
        return "Identify the exact person and retrieve a source connecting the clue conjunction."
    if ("what" in lowered and "year" in lowered) or "tell me the year" in lowered:
        return "Identify the exact year and retrieve a source connecting it to the event."
    if "subtitle" in lowered:
        return "Identify the exact book subtitle and retrieve a source connecting the work and clues."
    return "Identify the exact requested answer and retrieve a source connecting all key clues."


def build_near_miss_packet(
    *, frontier_item: AtomicClueCaseResult, documents_per_clue: int, character_cap: int
) -> tuple[NearMissGroup, ...]:
    groups: list[NearMissGroup] = []
    for search in frontier_item.searches:
        documents = tuple(
            NearMissDocument(
                docid=item.docid,
                score=item.score,
                snippet=item.snippet[:character_cap],
            )
            for item in search.documents[:documents_per_clue]
        )
        if documents:
            groups.append(NearMissGroup(clue_query=search.query, documents=documents))
    if not groups:
        raise ValueError(f"frontier case has no near-miss documents: {frontier_item.query_id}")
    return tuple(groups)


def load_corpus_bridge_cases(
    *, repository_root: Path, registration: CorpusBridgeRegistration
) -> tuple[CorpusBridgeCase, ...]:
    frontier = AtomicClueResult.model_validate_json(
        (repository_root / registration.frontier_source.path).read_text(encoding="utf-8")
    )
    if frontier.search_calls != registration.fixed_contract.prior_search_calls:
        raise ValueError("frontier search count differs from corpus bridge registration")
    if any(item.gold_hits for item in frontier.items):
        raise ValueError("corpus bridge source is not a pure near-miss frontier")
    frontier_by_id = {item.query_id: item for item in frontier.items}
    gold = json.loads(
        (repository_root / registration.gold_slice.path).read_text(encoding="utf-8")
    )
    gold_by_id = {str(item["query_id"]): item for item in gold["rows"]}
    sources = {item.query_id: item for item in registration.case_sources}
    cases: list[CorpusBridgeCase] = []
    for query_id in registration.query_ids:
        source = sources[query_id]
        request = json.loads(
            (repository_root / source.request.path).read_text(encoding="utf-8")
        )
        run = json.loads((repository_root / source.run.path).read_text(encoding="utf-8"))
        audit = run.get("answer_first_audit")
        if not isinstance(audit, dict) or audit.get("audit_status") != "open":
            raise ValueError(f"corpus bridge source has no open debt: {query_id}")
        seed = audit.get("subject_hypothesis")
        if not isinstance(seed, str) or not seed.strip():
            raise ValueError(f"corpus bridge source has no subject seed: {query_id}")
        question = str(request["question"])
        cases.append(
            CorpusBridgeCase(
                query_id=query_id,
                question=question,
                unresolved_obligation=_obligation_for_question(question),
                unverified_prior_subject=seed,
                near_miss_packet=build_near_miss_packet(
                    frontier_item=frontier_by_id[query_id],
                    documents_per_clue=registration.fixed_contract.documents_per_clue,
                    character_cap=registration.fixed_contract.snippet_character_cap,
                ),
                gold_docids=tuple(str(item) for item in gold_by_id[query_id]["gold_docids"]),
            )
        )
    return tuple(cases)


def build_corpus_bridge_prompt(case: CorpusBridgeCase) -> str:
    return json.dumps(
        {
            "task": (
                "Return exactly one compact JSON object. Use the near-miss corpus packet to "
                "induce three answer-linked bridge terms and three next-hop retrieval queries."
            ),
            "epistemic_rules": [
                "The prior subject is unverified and may be wrong.",
                "Packet documents are retrieval leads, not proof of the final answer.",
                "Each bridge must cite one to three packet docids that motivated the pivot.",
            ],
            "rules": [
                "Return exactly three distinct bridges.",
                "Prefer concrete entities, geographies, domains, works, topics, or events that connect multiple clues.",
                "Do not merely repeat a clue sentence or the unverified subject.",
                "Each term contains 1 to 5 words; each query contains 5 to 18 terms and includes its term.",
                "Use world knowledge only to connect vocabulary observed in the packet to the question.",
                "Do not answer the question and do not include rationales or extra keys.",
            ],
            "input": {
                "question": case.question,
                "unresolved_obligation": case.unresolved_obligation,
                "unverified_prior_subject": case.unverified_prior_subject,
                "near_miss_packet": [
                    {
                        "clue_query": group.clue_query,
                        "documents": [doc.model_dump(mode="json") for doc in group.documents],
                    }
                    for group in case.near_miss_packet
                ],
            },
            "output_schema": {
                "bridges": [
                    {
                        "type": "entity | domain | topic | geography | work | event",
                        "term": "1 to 5 words",
                        "source_docids": ["one to three packet docids"],
                        "query": "5 to 18 terms including term",
                    }
                ]
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_slate_for_case(*, case: CorpusBridgeCase, slate: CorpusBridgeSlate) -> None:
    packet_docids = {
        doc.docid for group in case.near_miss_packet for doc in group.documents
    }
    anchored_tokens = set(_tokens(case.question)) | set(_tokens(case.unverified_prior_subject))
    for bridge in slate.bridges:
        if any(docid not in packet_docids for docid in bridge.source_docids):
            raise ValueError("corpus bridge cites a docid outside the near-miss packet")
        if not (set(_tokens(bridge.term)) - anchored_tokens):
            raise ValueError("corpus bridge adds no vocabulary beyond question and prior subject")


def _search(
    search_url: str, run_id: str, query: str, timeout_seconds: int
) -> tuple[tuple[str, ...], int]:
    request = Request(
        search_url,
        data=json.dumps({"run_id": run_id, "query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"corpus bridge retrieval failed: {error}") from error
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("corpus bridge retrieval response has no results array")
    return (
        tuple(str(item["docid"]) for item in results),
        round((time.perf_counter() - started) * 1000),
    )


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        input_cache_hit_tokens=left.input_cache_hit_tokens + right.input_cache_hit_tokens,
        input_cache_miss_tokens=left.input_cache_miss_tokens + right.input_cache_miss_tokens,
        estimated_cost_usd=left.estimated_cost_usd + right.estimated_cost_usd,
    )


def run_corpus_bridge_probe(
    *,
    registration_path: Path,
    provider: LLMProvider,
    output_path: Path,
    timeout_seconds: int = 60,
) -> CorpusBridgeResult:
    registration = load_corpus_bridge_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("corpus bridge output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("corpus bridge output already exists")
    if provider.model != registration.fixed_contract.model:
        raise ValueError("provider model differs from the corpus bridge registration")
    cases = load_corpus_bridge_cases(repository_root=root, registration=registration)
    baseline = FirewallProbeResult.model_validate_json(
        (root / registration.baseline_probe.path).read_text(encoding="utf-8")
    )
    baseline_by_id = {item.query_id: item.case_gold_hits for item in baseline.items}
    items: list[CorpusBridgeItem] = []
    total_usage = Usage()
    provider_attempts = final_search_calls = parse_failures = 0
    provider_request_failures = search_failures = 0
    observability: Literal["complete", "lower_bound_after_failure"] = "complete"
    for case in cases:
        packet_json = json.dumps(
            [group.model_dump(mode="json") for group in case.near_miss_packet],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = build_corpus_bridge_prompt(case)
        raw_text: str | None = None
        raw_hash: str | None = None
        usage = Usage()
        provider_latency = 0
        slate: CorpusBridgeSlate | None = None
        searches: list[CorpusBridgeSearch] = []
        error_text: str | None = None
        try:
            provider_attempts += 1
            completion = provider.complete(
                stage="corpus_grounded_bridge_induction",
                prompt=prompt,
                json_output=True,
                max_output_tokens=registration.fixed_contract.max_planner_output_tokens,
            )
            raw_text = completion.text
            raw_hash = _sha256_text(raw_text)
            usage = completion.usage
            provider_latency = completion.latency_ms
            total_usage = _add_usage(total_usage, usage)
        except Exception as error:
            provider_request_failures += 1
            observability = "lower_bound_after_failure"
            error_text = f"provider_request_error: {type(error).__name__}: {error}"
        if error_text is None:
            try:
                slate = CorpusBridgeSlate.model_validate_json(raw_text)
                validate_slate_for_case(case=case, slate=slate)
            except Exception as error:
                parse_failures += 1
                error_text = f"slate_parse_error: {type(error).__name__}: {error}"
        if slate is not None and error_text is None:
            for index, bridge in enumerate(slate.bridges):
                try:
                    final_search_calls += 1
                    docids, latency = _search(
                        registration.fixed_contract.search_url,
                        f"corpus-grounded-bridge-{case.query_id}-{index}",
                        bridge.query,
                        timeout_seconds,
                    )
                    if len(docids) > registration.fixed_contract.max_search_results:
                        raise ValueError("corpus bridge retriever exceeded the result cap")
                    hits = tuple(item for item in docids if item in case.gold_docids)
                    ranks = tuple(
                        rank + 1 for rank, item in enumerate(docids) if item in case.gold_docids
                    )
                    searches.append(
                        CorpusBridgeSearch(
                            bridge_index=index,
                            term=bridge.term,
                            query=bridge.query,
                            returned_docids=docids,
                            gold_hits=hits,
                            gold_hit_ranks=ranks,
                            latency_ms=latency,
                        )
                    )
                except Exception as error:
                    search_failures += 1
                    error_text = f"search_error: {type(error).__name__}: {error}"
                    break
        case_hits = tuple(dict.fromkeys(hit for item in searches for hit in item.gold_hits))
        items.append(
            CorpusBridgeItem(
                query_id=case.query_id,
                status="succeeded" if error_text is None else "failed",
                prompt_sha256=_sha256_text(prompt),
                packet_sha256=_sha256_text(packet_json),
                context_fields=registration.fixed_contract.context_fields,
                baseline_gold_hits=baseline_by_id[case.query_id],
                slate=slate,
                searches=tuple(searches),
                case_gold_hits=case_hits,
                raw_completion_sha256=raw_hash,
                raw_completion_text=raw_text,
                usage=usage,
                provider_latency_ms=provider_latency,
                error=error_text,
            )
        )
    baseline_hit_cases = sum(bool(item.baseline_gold_hits) for item in items)
    candidate_hit_cases = sum(bool(item.case_gold_hits) for item in items)
    delta = candidate_hit_cases - baseline_hit_cases
    prior_search_calls = registration.fixed_contract.prior_search_calls
    cumulative_search_calls = prior_search_calls + final_search_calls
    expected_cases = len(cases)
    expected_final_searches = (
        expected_cases * registration.fixed_contract.final_search_calls_per_case
    )
    acceptance = registration.acceptance
    completed = sum(item.status == "succeeded" for item in items)
    gates = (
        DecisionGate(gate_id="completed_cases", observed=completed, operator="eq", threshold=expected_cases, passed=completed == expected_cases),
        DecisionGate(gate_id="provider_attempts", observed=provider_attempts, operator="eq", threshold=expected_cases, passed=provider_attempts == expected_cases),
        DecisionGate(gate_id="final_search_calls", observed=final_search_calls, operator="eq", threshold=expected_final_searches, passed=final_search_calls == expected_final_searches),
        DecisionGate(gate_id="cumulative_search_calls", observed=cumulative_search_calls, operator="le", threshold=registration.fixed_contract.maximum_cumulative_search_calls, passed=cumulative_search_calls <= registration.fixed_contract.maximum_cumulative_search_calls),
        DecisionGate(gate_id="parse_failures", observed=parse_failures, operator="eq", threshold=0, passed=parse_failures == 0),
        DecisionGate(gate_id="provider_request_failures", observed=provider_request_failures, operator="eq", threshold=0, passed=provider_request_failures == 0),
        DecisionGate(gate_id="search_failures", observed=search_failures, operator="eq", threshold=0, passed=search_failures == 0),
        DecisionGate(gate_id="candidate_gold_hit_cases", observed=candidate_hit_cases, operator="ge", threshold=acceptance.minimum_candidate_gold_hit_cases, passed=candidate_hit_cases >= acceptance.minimum_candidate_gold_hit_cases),
        DecisionGate(gate_id="gold_hit_case_delta", observed=delta, operator="ge", threshold=acceptance.minimum_gold_hit_case_delta, passed=delta >= acceptance.minimum_gold_hit_case_delta),
        DecisionGate(gate_id="provider_cost_usd", observed=total_usage.estimated_cost_usd, operator="le", threshold=acceptance.maximum_provider_cost_usd, passed=total_usage.estimated_cost_usd <= acceptance.maximum_provider_cost_usd),
    )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = CorpusBridgeResult(
        created_at=_utc_now(),
        status=(
            "succeeded"
            if parse_failures == provider_request_failures == search_failures == 0
            else "failed"
        ),
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=expected_cases,
        provider_attempts=provider_attempts,
        prior_search_calls=prior_search_calls,
        final_search_calls=final_search_calls,
        cumulative_search_calls=cumulative_search_calls,
        parse_failures=parse_failures,
        provider_request_failures=provider_request_failures,
        search_failures=search_failures,
        baseline_gold_hit_cases=baseline_hit_cases,
        candidate_gold_hit_cases=candidate_hit_cases,
        gold_hit_case_delta=delta,
        total_usage=total_usage,
        provider_cost_observability=observability,
        context_isolation_verified=all(
            item.context_fields == registration.fixed_contract.context_fields for item in items
        ),
        items=tuple(items),
        gates=gates,
        next_action=(
            "preregister_corpus_bridge_stage_replay"
            if decision == "pass"
            else "close_without_end_to_end_expansion"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
