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
from .counter_candidate_replay import CandidateReplayResult
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


class FirewallFixedContract(StrictContract):
    model: str = Field(min_length=1)
    thinking_mode: Literal["disabled"] = "disabled"
    system_prompt_policy: str = Field(min_length=1)
    retriever_id: str = Field(min_length=1)
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    context_fields: tuple[
        Literal["question", "unresolved_obligation", "unverified_prior_subject"], ...
    ]
    forbidden_context_fields: tuple[
        Literal["draft_answer", "prior_exact_answer", "prior_query", "gold_docids"],
        ...,
    ]
    anchors_per_case: Literal[3] = 3
    max_search_results: int = Field(ge=1, le=100)
    max_planner_output_tokens: int = Field(ge=1, le=512)
    provider_attempts_per_case: Literal[1] = 1
    search_calls_per_case: Literal[3] = 3
    sealed_holdout_access: Literal["forbidden"] = "forbidden"

    @model_validator(mode="after")
    def context_boundary_is_exact(self) -> "FirewallFixedContract":
        expected = ("question", "unresolved_obligation", "unverified_prior_subject")
        if self.context_fields != expected:
            raise ValueError("hypothesis firewall context boundary changed")
        required_forbidden = {
            "draft_answer",
            "prior_exact_answer",
            "prior_query",
            "gold_docids",
        }
        if set(self.forbidden_context_fields) != required_forbidden:
            raise ValueError("hypothesis firewall forbidden context is incomplete")
        return self


class FirewallAcceptance(StrictContract):
    minimum_candidate_gold_hit_cases: int = Field(ge=1)
    minimum_gold_hit_case_delta: int = Field(ge=1)
    parse_failures_must_equal: Literal[0] = 0
    provider_request_failures_must_equal: Literal[0] = 0
    search_failures_must_equal: Literal[0] = 0
    maximum_provider_cost_usd: float = Field(gt=0)


class FirewallRegistration(StrictContract):
    schema_version: Literal["hypothesis-firewall-slate-registration-v0"] = (
        "hypothesis-firewall-slate-registration-v0"
    )
    status: Literal["registered_pre_provider"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    source_summary: ArtifactReference
    case_sources: tuple[CaseSourceReference, ...] = Field(min_length=1)
    gold_slice: ArtifactReference
    baseline_replay: ArtifactReference
    oracle_replay: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    fixed_contract: FirewallFixedContract
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: FirewallAcceptance
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def ids_and_gates_match_cases(self) -> "FirewallRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("hypothesis firewall IDs must be unique")
        if tuple(item.query_id for item in self.case_sources) != self.query_ids:
            raise ValueError("case source order must exactly match query IDs")
        if self.acceptance.minimum_candidate_gold_hit_cases > len(self.query_ids):
            raise ValueError("hypothesis firewall gold-hit gate exceeds registered cases")
        return self


class FirewallAnchor(StrictContract):
    role: Literal["seed_expansion", "counter_hypothesis"]
    type: Literal["entity", "domain", "topic", "geography", "work", "event"]
    term: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def anchor_is_compact(self) -> "FirewallAnchor":
        if not 1 <= len(_tokens(self.term)) <= 5:
            raise ValueError("firewall anchor term must contain one to five terms")
        if not 5 <= len(_tokens(self.query)) <= 18:
            raise ValueError("firewall query must contain 5 to 18 terms")
        if any(token not in set(_tokens(self.query)) for token in _tokens(self.term)):
            raise ValueError("firewall query omits its anchor term")
        return self


class HypothesisFirewallSlate(StrictContract):
    anchors: tuple[FirewallAnchor, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def roles_and_terms_are_distinct(self) -> "HypothesisFirewallSlate":
        roles = tuple(item.role for item in self.anchors)
        if roles != ("seed_expansion", "counter_hypothesis", "counter_hypothesis"):
            raise ValueError("firewall slate must contain one seed expansion then two counters")
        signatures = tuple(_tokens(item.term) for item in self.anchors)
        if len(set(signatures)) != 3:
            raise ValueError("firewall slate anchor terms must be distinct")
        return self


class FirewallCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    unresolved_obligation: str = Field(min_length=1)
    unverified_prior_subject: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)


class FirewallSearch(StrictContract):
    anchor_index: int = Field(ge=0, le=2)
    role: str
    type: str
    term: str
    query: str
    returned_docids: tuple[str, ...]
    gold_hits: tuple[str, ...]
    gold_hit_ranks: tuple[int, ...]
    latency_ms: int = Field(ge=0)


class FirewallProbeItem(StrictContract):
    query_id: str
    status: Literal["succeeded", "failed"]
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_fields: tuple[str, ...]
    baseline_gold_hits: tuple[str, ...]
    slate: HypothesisFirewallSlate | None
    searches: tuple[FirewallSearch, ...]
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


class FirewallProbeResult(StrictContract):
    schema_version: Literal["hypothesis-firewall-slate-probe-v0"] = (
        "hypothesis-firewall-slate-probe-v0"
    )
    created_at: str
    status: Literal["succeeded", "failed"]
    decision: Literal["pass", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    provider_attempts: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    provider_request_failures: int = Field(ge=0)
    search_failures: int = Field(ge=0)
    baseline_gold_hit_cases: int = Field(ge=0)
    candidate_gold_hit_cases: int = Field(ge=0)
    gold_hit_case_delta: int
    total_usage: Usage
    provider_cost_observability: Literal["complete", "lower_bound_after_failure"]
    context_isolation_verified: bool
    items: tuple[FirewallProbeItem, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "preregister_firewall_stage_replay",
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


def load_firewall_registration(path: Path) -> FirewallRegistration:
    registration = FirewallRegistration.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[2]
    case_artifacts = tuple(
        artifact
        for source in registration.case_sources
        for artifact in (source.request, source.run)
    )
    _validate_artifacts(
        root,
        (
            registration.source_summary,
            *case_artifacts,
            registration.gold_slice,
            registration.baseline_replay,
            registration.oracle_replay,
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


def load_firewall_cases(
    *, repository_root: Path, registration: FirewallRegistration
) -> tuple[FirewallCase, ...]:
    summary = json.loads(
        (repository_root / registration.source_summary.path).read_text(encoding="utf-8")
    )
    summary_by_id = {str(item["query_id"]): item for item in summary["items"]}
    gold = json.loads(
        (repository_root / registration.gold_slice.path).read_text(encoding="utf-8")
    )
    gold_by_id = {str(item["query_id"]): item for item in gold["rows"]}
    sources = {item.query_id: item for item in registration.case_sources}
    cases: list[FirewallCase] = []
    for query_id in registration.query_ids:
        if query_id not in summary_by_id or query_id not in gold_by_id:
            raise ValueError(f"firewall case is absent from source or gold: {query_id}")
        if summary_by_id[query_id]["status"] != "succeeded":
            raise ValueError(f"firewall source query did not succeed: {query_id}")
        source = sources[query_id]
        request = json.loads(
            (repository_root / source.request.path).read_text(encoding="utf-8")
        )
        run = json.loads((repository_root / source.run.path).read_text(encoding="utf-8"))
        audit = run.get("answer_first_audit")
        if not isinstance(audit, dict) or audit.get("audit_status") != "open":
            raise ValueError(f"firewall source has no open evidence debt: {query_id}")
        seed = audit.get("subject_hypothesis")
        if not isinstance(seed, str) or not seed.strip():
            raise ValueError(f"firewall source has no subject hypothesis: {query_id}")
        question = str(request["question"])
        cases.append(
            FirewallCase(
                query_id=query_id,
                question=question,
                unresolved_obligation=_obligation_for_question(question),
                unverified_prior_subject=seed,
                gold_docids=tuple(str(item) for item in gold_by_id[query_id]["gold_docids"]),
            )
        )
    return tuple(cases)


def build_firewall_prompt(
    *, question: str, unresolved_obligation: str, unverified_prior_subject: str
) -> str:
    return json.dumps(
        {
            "task": (
                "Return exactly one compact JSON object. Build a retrieval slate behind a "
                "hypothesis firewall: preserve the tagged prior subject only as an unverified "
                "seed and generate two independent counter-hypotheses."
            ),
            "epistemic_rule": (
                "The prior subject is not evidence and may be wrong. Never treat it as the answer."
            ),
            "rules": [
                "Return exactly three anchors in order: one seed_expansion, then two counter_hypothesis anchors.",
                "Each counter term must add a concrete entity, domain, geography, work, topic, or event absent from both the question and prior seed.",
                "Counter terms must not merely paraphrase the prior seed or surface clues.",
                "Each term contains 1 to 5 words; each query contains 5 to 18 terms and includes its term.",
                "Use world knowledge to propose answer-linked source vocabulary, but do not answer the question.",
                "Do not include rationales or extra keys.",
            ],
            "input": {
                "question": question,
                "unresolved_obligation": unresolved_obligation,
                "unverified_prior_subject": unverified_prior_subject,
            },
            "output_schema": {
                "anchors": [
                    {
                        "role": "seed_expansion | counter_hypothesis",
                        "type": "entity | domain | topic | geography | work | event",
                        "term": "1 to 5 words",
                        "query": "5 to 18 terms including term",
                    }
                ]
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_slate_for_case(
    *, question: str, unverified_prior_subject: str, slate: HypothesisFirewallSlate
) -> None:
    question_tokens = set(_tokens(question))
    seed_tokens = set(_tokens(unverified_prior_subject))
    generic = {
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
        "subject",
    }
    if not (set(_tokens(slate.anchors[0].query)) & seed_tokens):
        raise ValueError("seed expansion query does not preserve any prior-subject token")
    for anchor in slate.anchors[1:]:
        novel = set(_tokens(anchor.term)) - question_tokens - seed_tokens
        if not novel:
            raise ValueError("counter-hypothesis term adds no vocabulary beyond question and seed")
        if novel.issubset(generic):
            raise ValueError("counter-hypothesis adds only generic or honorific vocabulary")


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
        raise RuntimeError(f"hypothesis firewall retrieval failed: {error}") from error
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("hypothesis firewall retrieval response has no results array")
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


def run_firewall_probe(
    *,
    registration_path: Path,
    provider: LLMProvider,
    output_path: Path,
    timeout_seconds: int = 60,
) -> FirewallProbeResult:
    registration = load_firewall_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("hypothesis firewall output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("hypothesis firewall output already exists")
    if provider.model != registration.fixed_contract.model:
        raise ValueError("provider model differs from the hypothesis firewall registration")
    cases = load_firewall_cases(repository_root=root, registration=registration)
    baseline = CandidateReplayResult.model_validate_json(
        (root / registration.baseline_replay.path).read_text(encoding="utf-8")
    )
    baseline_hits_by_id = {
        query_id: tuple(
            dict.fromkeys(
                hit
                for item in baseline.items
                if item.query_id == query_id
                for hit in item.gold_hits
            )
        )
        for query_id in registration.query_ids
    }
    items: list[FirewallProbeItem] = []
    total_usage = Usage()
    provider_attempts = search_calls = parse_failures = 0
    provider_request_failures = search_failures = 0
    observability: Literal["complete", "lower_bound_after_failure"] = "complete"
    for case in cases:
        prompt = build_firewall_prompt(
            question=case.question,
            unresolved_obligation=case.unresolved_obligation,
            unverified_prior_subject=case.unverified_prior_subject,
        )
        raw_text: str | None = None
        raw_hash: str | None = None
        usage = Usage()
        provider_latency = 0
        slate: HypothesisFirewallSlate | None = None
        searches: list[FirewallSearch] = []
        error_text: str | None = None
        try:
            provider_attempts += 1
            completion = provider.complete(
                stage="hypothesis_firewall_slate",
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
                slate = HypothesisFirewallSlate.model_validate_json(raw_text)
                validate_slate_for_case(
                    question=case.question,
                    unverified_prior_subject=case.unverified_prior_subject,
                    slate=slate,
                )
            except Exception as error:
                parse_failures += 1
                error_text = f"slate_parse_error: {type(error).__name__}: {error}"
        if slate is not None and error_text is None:
            for index, anchor in enumerate(slate.anchors):
                try:
                    search_calls += 1
                    docids, latency = _search(
                        registration.fixed_contract.search_url,
                        f"hypothesis-firewall-{case.query_id}-{index}",
                        anchor.query,
                        timeout_seconds,
                    )
                    if len(docids) > registration.fixed_contract.max_search_results:
                        raise ValueError("hypothesis firewall retriever exceeded the result cap")
                    hits = tuple(item for item in docids if item in case.gold_docids)
                    ranks = tuple(
                        rank + 1 for rank, item in enumerate(docids) if item in case.gold_docids
                    )
                    searches.append(
                        FirewallSearch(
                            anchor_index=index,
                            role=anchor.role,
                            type=anchor.type,
                            term=anchor.term,
                            query=anchor.query,
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
            FirewallProbeItem(
                query_id=case.query_id,
                status="succeeded" if error_text is None else "failed",
                question_sha256=_sha256_text(case.question),
                seed_sha256=_sha256_text(case.unverified_prior_subject),
                prompt_sha256=_sha256_text(prompt),
                context_fields=registration.fixed_contract.context_fields,
                baseline_gold_hits=baseline_hits_by_id[case.query_id],
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
    completed = sum(item.status == "succeeded" for item in items)
    acceptance = registration.acceptance
    expected_cases = len(cases)
    expected_searches = expected_cases * registration.fixed_contract.search_calls_per_case
    gates = (
        DecisionGate(gate_id="completed_cases", observed=completed, operator="eq", threshold=expected_cases, passed=completed == expected_cases),
        DecisionGate(gate_id="provider_attempts", observed=provider_attempts, operator="eq", threshold=expected_cases, passed=provider_attempts == expected_cases),
        DecisionGate(gate_id="search_calls", observed=search_calls, operator="eq", threshold=expected_searches, passed=search_calls == expected_searches),
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
    result = FirewallProbeResult(
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
        search_calls=search_calls,
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
            "preregister_firewall_stage_replay"
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
