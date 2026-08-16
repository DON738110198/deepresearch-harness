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

from .contracts import Usage
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


class CaseRequestReference(ArtifactReference):
    query_id: str = Field(min_length=1)


class CounterFixedContract(StrictContract):
    model: str = Field(min_length=1)
    thinking_mode: Literal["disabled"] = "disabled"
    system_prompt_policy: str = Field(min_length=1)
    retriever_id: str = Field(min_length=1)
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    context_fields: tuple[Literal["question", "unresolved_obligation"], ...]
    forbidden_context_fields: tuple[
        Literal[
            "draft_answer",
            "prior_candidate",
            "prior_query",
            "gold_docids",
        ],
        ...,
    ]
    bridges_per_case: Literal[3] = 3
    max_search_results: int = Field(ge=1, le=100)
    max_planner_output_tokens: int = Field(ge=1, le=256)
    provider_attempts_per_case: Literal[1] = 1
    search_calls_per_case: Literal[1] = 1
    sealed_holdout_access: Literal["forbidden"] = "forbidden"

    @model_validator(mode="after")
    def context_boundary_is_exact(self) -> "CounterFixedContract":
        if self.context_fields != ("question", "unresolved_obligation"):
            raise ValueError("counter-hypothesis context boundary changed")
        required_forbidden = {
            "draft_answer",
            "prior_candidate",
            "prior_query",
            "gold_docids",
        }
        if set(self.forbidden_context_fields) != required_forbidden:
            raise ValueError("counter-hypothesis forbidden context is incomplete")
        return self


class CounterAcceptance(StrictContract):
    minimum_candidate_gold_hit_cases: int = Field(ge=1)
    minimum_gold_hit_case_delta: int = Field(ge=1)
    parse_failures_must_equal: Literal[0] = 0
    request_failures_must_equal: Literal[0] = 0
    maximum_provider_cost_usd: float = Field(gt=0)


class CounterProbeRegistration(StrictContract):
    schema_version: Literal["draft-blind-counter-hypothesis-registration-v0"] = (
        "draft-blind-counter-hypothesis-registration-v0"
    )
    status: Literal["registered_pre_provider"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    source_summary: ArtifactReference
    case_requests: tuple[CaseRequestReference, ...] = Field(min_length=1)
    gold_slice: ArtifactReference
    baseline_probe: ArtifactReference
    oracle_replay: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    fixed_contract: CounterFixedContract
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: CounterAcceptance
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def ids_and_gates_match_cases(self) -> "CounterProbeRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("counter-hypothesis IDs must be unique")
        request_ids = tuple(item.query_id for item in self.case_requests)
        if request_ids != self.query_ids:
            raise ValueError("case request order must exactly match query IDs")
        if self.acceptance.minimum_candidate_gold_hit_cases > len(self.query_ids):
            raise ValueError("counter-hypothesis gold-hit gate exceeds registered cases")
        return self


class BridgeCandidate(StrictContract):
    type: Literal["domain", "geography", "topic", "entity"]
    term: str = Field(min_length=1, max_length=60)

    @model_validator(mode="after")
    def term_is_compact(self) -> "BridgeCandidate":
        if not 1 <= len(_tokens(self.term)) <= 4:
            raise ValueError("bridge term must contain one to four terms")
        return self


class BridgePacket(StrictContract):
    bridges: tuple[BridgeCandidate, ...] = Field(min_length=3, max_length=3)
    selected: int = Field(ge=0, le=2)
    query: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def packet_is_contrastive_and_compact(self) -> "BridgePacket":
        bridge_types = tuple(item.type for item in self.bridges)
        if len(set(bridge_types)) != 3:
            raise ValueError("bridge candidates must use three distinct types")
        signatures = tuple(_tokens(item.term) for item in self.bridges)
        if len(set(signatures)) != 3:
            raise ValueError("bridge candidates must use distinct terms")
        if not 5 <= len(_tokens(self.query)) <= 18:
            raise ValueError("bridge query must contain 5 to 18 terms")
        return self


class CounterHypothesisCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    unresolved_obligation: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)


class CounterProbeItem(StrictContract):
    query_id: str
    status: Literal["succeeded", "failed"]
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_fields: tuple[str, ...]
    baseline_gold_hits: tuple[str, ...]
    packet: BridgePacket | None
    selected_novel_terms: tuple[str, ...]
    candidate_docids: tuple[str, ...]
    candidate_gold_hits: tuple[str, ...]
    candidate_gold_hit_ranks: tuple[int, ...]
    raw_completion_sha256: str | None
    raw_completion_text: str | None
    usage: Usage
    provider_latency_ms: int = Field(ge=0)
    search_latency_ms: int = Field(ge=0)
    error: str | None


class DecisionGate(StrictContract):
    gate_id: str
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class CounterProbeResult(StrictContract):
    schema_version: Literal["draft-blind-counter-hypothesis-probe-v0"] = (
        "draft-blind-counter-hypothesis-probe-v0"
    )
    created_at: str
    status: Literal["succeeded", "failed"]
    decision: Literal["pass", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    provider_attempts: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    request_failures: int = Field(ge=0)
    baseline_gold_hit_cases: int = Field(ge=0)
    candidate_gold_hit_cases: int = Field(ge=0)
    gold_hit_case_delta: int
    total_usage: Usage
    provider_cost_observability: Literal["complete", "lower_bound_after_failure"]
    context_isolation_verified: bool
    items: tuple[CounterProbeItem, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "preregister_counter_hypothesis_stage_replay",
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


def load_counter_registration(path: Path) -> CounterProbeRegistration:
    registration = CounterProbeRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            registration.source_summary,
            *registration.case_requests,
            registration.gold_slice,
            registration.baseline_probe,
            registration.oracle_replay,
            *registration.frozen_artifacts,
        ),
    )
    return registration


def _obligation_for_question(question: str) -> str:
    lowered = question.casefold()
    if "first name and surname" in lowered or lowered.lstrip().startswith("who"):
        return "Identify the exact person and retrieve a source connecting the clue conjunction."
    if "what" in lowered and "year" in lowered or "tell me the year" in lowered:
        return "Identify the exact year and retrieve a source connecting it to the named event."
    if "subtitle" in lowered:
        return "Identify the exact book subtitle and retrieve a source connecting the author and clues."
    return "Identify the exact requested answer and retrieve a source connecting all key clues."


def load_counter_cases(
    *, repository_root: Path, registration: CounterProbeRegistration
) -> tuple[CounterHypothesisCase, ...]:
    summary = json.loads(
        (repository_root / registration.source_summary.path).read_text(encoding="utf-8")
    )
    gold = json.loads(
        (repository_root / registration.gold_slice.path).read_text(encoding="utf-8")
    )
    summary_items = {str(item["query_id"]): item for item in summary["items"]}
    gold_rows = {str(item["query_id"]): item for item in gold["rows"]}
    requests = {item.query_id: item for item in registration.case_requests}
    cases: list[CounterHypothesisCase] = []
    for query_id in registration.query_ids:
        summary_item = summary_items.get(query_id)
        gold_row = gold_rows.get(query_id)
        if summary_item is None or gold_row is None:
            raise ValueError(f"registered query is absent from source or gold: {query_id}")
        if summary_item["status"] != "succeeded":
            raise ValueError(f"counter-hypothesis source query did not succeed: {query_id}")
        request = json.loads(
            (repository_root / requests[query_id].path).read_text(encoding="utf-8")
        )
        question = str(request["question"])
        cases.append(
            CounterHypothesisCase(
                query_id=query_id,
                question=question,
                unresolved_obligation=_obligation_for_question(question),
                gold_docids=tuple(str(item) for item in gold_row["gold_docids"]),
            )
        )
    return tuple(cases)


def build_counter_hypothesis_prompt(*, question: str, unresolved_obligation: str) -> str:
    return json.dumps(
        {
            "task": (
                "Return exactly one compact JSON object. Work independently from any prior "
                "answer. Propose three counter-hypothesis bridge terms, select one, and form "
                "one retrieval query."
            ),
            "definition": (
                "A bridge is source vocabulary not stated in the question that could connect "
                "the full clue conjunction to a relevant document."
            ),
            "rules": [
                "Return exactly three bridges with three distinct types.",
                "Each bridge term must contain one to four words.",
                "The selected bridge must add factual vocabulary absent from the question.",
                "The query must include the selected bridge and contain 5 to 18 terms.",
                "Do not answer the question and do not include rationales or extra keys.",
            ],
            "input": {
                "question": question,
                "unresolved_obligation": unresolved_obligation,
            },
            "output_schema": {
                "bridges": [
                    {"type": "domain | geography | topic | entity", "term": "1 to 4 words"}
                ],
                "selected": "0, 1, or 2",
                "query": "5 to 18 terms including the selected bridge",
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_packet_for_question(
    *, question: str, packet: BridgePacket
) -> tuple[str, ...]:
    selected_tokens = _tokens(packet.bridges[packet.selected].term)
    query_tokens = set(_tokens(packet.query))
    if any(token not in query_tokens for token in selected_tokens):
        raise ValueError("bridge query omits the selected bridge term")
    question_tokens = set(_tokens(question))
    novel = tuple(token for token in selected_tokens if token not in question_tokens)
    if not novel:
        raise ValueError("selected bridge adds no vocabulary beyond the question")
    non_bridge_terms = {
        "answer",
        "biography",
        "dame",
        "dr",
        "evidence",
        "individual",
        "lady",
        "lord",
        "mr",
        "mrs",
        "ms",
        "person",
        "sir",
    }
    if set(novel).issubset(non_bridge_terms):
        raise ValueError("selected bridge adds only generic or honorific vocabulary")
    return novel


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
        raise RuntimeError(f"counter-hypothesis retrieval failed: {error}") from error
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("counter-hypothesis retrieval response has no results array")
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


def run_counter_probe(
    *,
    registration_path: Path,
    provider: LLMProvider,
    output_path: Path,
    timeout_seconds: int = 60,
) -> CounterProbeResult:
    registration = load_counter_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("counter-hypothesis output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("counter-hypothesis output already exists")
    if provider.model != registration.fixed_contract.model:
        raise ValueError("provider model differs from the counter-hypothesis registration")
    cases = load_counter_cases(repository_root=root, registration=registration)
    baseline = json.loads(
        (root / registration.baseline_probe.path).read_text(encoding="utf-8")
    )
    baseline_by_id = {str(item["query_id"]): item for item in baseline["items"]}
    items: list[CounterProbeItem] = []
    total_usage = Usage()
    provider_attempts = search_calls = parse_failures = request_failures = 0
    observability: Literal["complete", "lower_bound_after_failure"] = "complete"
    for case in cases:
        baseline_hits = tuple(
            str(item) for item in baseline_by_id[case.query_id]["candidate_gold_hits"]
        )
        prompt = build_counter_hypothesis_prompt(
            question=case.question,
            unresolved_obligation=case.unresolved_obligation,
        )
        question_hash = _sha256_text(case.question)
        prompt_hash = _sha256_text(prompt)
        usage = Usage()
        provider_latency = search_latency = 0
        raw_text: str | None = None
        raw_hash: str | None = None
        try:
            provider_attempts += 1
            completion = provider.complete(
                stage="counter_hypothesis_packet",
                prompt=prompt,
                json_output=True,
                max_output_tokens=registration.fixed_contract.max_planner_output_tokens,
            )
            raw_text = completion.text
            raw_hash = _sha256_text(raw_text)
            usage = completion.usage
            provider_latency = completion.latency_ms
            total_usage = _add_usage(total_usage, usage)
            try:
                packet = BridgePacket.model_validate_json(raw_text)
                novel = validate_packet_for_question(
                    question=case.question,
                    packet=packet,
                )
            except Exception as error:
                parse_failures += 1
                items.append(
                    CounterProbeItem(
                        query_id=case.query_id,
                        status="failed",
                        question_sha256=question_hash,
                        prompt_sha256=prompt_hash,
                        context_fields=registration.fixed_contract.context_fields,
                        baseline_gold_hits=baseline_hits,
                        packet=None,
                        selected_novel_terms=(),
                        candidate_docids=(),
                        candidate_gold_hits=(),
                        candidate_gold_hit_ranks=(),
                        raw_completion_sha256=raw_hash,
                        raw_completion_text=raw_text,
                        usage=usage,
                        provider_latency_ms=provider_latency,
                        search_latency_ms=0,
                        error=f"packet_parse_error: {type(error).__name__}: {error}",
                    )
                )
                continue
            search_calls += 1
            docids, search_latency = _search(
                registration.fixed_contract.search_url,
                f"draft-blind-counter-hypothesis-{case.query_id}",
                packet.query,
                timeout_seconds,
            )
            if len(docids) > registration.fixed_contract.max_search_results:
                raise ValueError("counter-hypothesis retriever exceeded the result cap")
            hits = tuple(item for item in docids if item in case.gold_docids)
            ranks = tuple(
                index + 1 for index, item in enumerate(docids) if item in case.gold_docids
            )
            items.append(
                CounterProbeItem(
                    query_id=case.query_id,
                    status="succeeded",
                    question_sha256=question_hash,
                    prompt_sha256=prompt_hash,
                    context_fields=registration.fixed_contract.context_fields,
                    baseline_gold_hits=baseline_hits,
                    packet=packet,
                    selected_novel_terms=novel,
                    candidate_docids=docids,
                    candidate_gold_hits=hits,
                    candidate_gold_hit_ranks=ranks,
                    raw_completion_sha256=raw_hash,
                    raw_completion_text=raw_text,
                    usage=usage,
                    provider_latency_ms=provider_latency,
                    search_latency_ms=search_latency,
                    error=None,
                )
            )
        except Exception as error:
            request_failures += 1
            observability = "lower_bound_after_failure"
            items.append(
                CounterProbeItem(
                    query_id=case.query_id,
                    status="failed",
                    question_sha256=question_hash,
                    prompt_sha256=prompt_hash,
                    context_fields=registration.fixed_contract.context_fields,
                    baseline_gold_hits=baseline_hits,
                    packet=None,
                    selected_novel_terms=(),
                    candidate_docids=(),
                    candidate_gold_hits=(),
                    candidate_gold_hit_ranks=(),
                    raw_completion_sha256=raw_hash,
                    raw_completion_text=raw_text,
                    usage=usage,
                    provider_latency_ms=provider_latency,
                    search_latency_ms=search_latency,
                    error=f"request_error: {type(error).__name__}: {error}",
                )
            )
    baseline_hit_cases = sum(bool(item.baseline_gold_hits) for item in items)
    candidate_hit_cases = sum(bool(item.candidate_gold_hits) for item in items)
    delta = candidate_hit_cases - baseline_hit_cases
    completed = sum(item.status == "succeeded" for item in items)
    acceptance = registration.acceptance
    expected = len(cases)
    gates = (
        DecisionGate(gate_id="completed_cases", observed=completed, operator="eq", threshold=expected, passed=completed == expected),
        DecisionGate(gate_id="provider_attempts", observed=provider_attempts, operator="eq", threshold=expected, passed=provider_attempts == expected),
        DecisionGate(gate_id="search_calls", observed=search_calls, operator="eq", threshold=expected, passed=search_calls == expected),
        DecisionGate(gate_id="parse_failures", observed=parse_failures, operator="eq", threshold=0, passed=parse_failures == 0),
        DecisionGate(gate_id="request_failures", observed=request_failures, operator="eq", threshold=0, passed=request_failures == 0),
        DecisionGate(gate_id="candidate_gold_hit_cases", observed=candidate_hit_cases, operator="ge", threshold=acceptance.minimum_candidate_gold_hit_cases, passed=candidate_hit_cases >= acceptance.minimum_candidate_gold_hit_cases),
        DecisionGate(gate_id="gold_hit_case_delta", observed=delta, operator="ge", threshold=acceptance.minimum_gold_hit_case_delta, passed=delta >= acceptance.minimum_gold_hit_case_delta),
        DecisionGate(gate_id="provider_cost_usd", observed=total_usage.estimated_cost_usd, operator="le", threshold=acceptance.maximum_provider_cost_usd, passed=total_usage.estimated_cost_usd <= acceptance.maximum_provider_cost_usd),
    )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = CounterProbeResult(
        created_at=_utc_now(),
        status="succeeded" if parse_failures == 0 and request_failures == 0 else "failed",
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=expected,
        provider_attempts=provider_attempts,
        search_calls=search_calls,
        parse_failures=parse_failures,
        request_failures=request_failures,
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
            "preregister_counter_hypothesis_stage_replay"
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
