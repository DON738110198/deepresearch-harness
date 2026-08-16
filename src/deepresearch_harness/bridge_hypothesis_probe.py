from __future__ import annotations

import json
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
from .repair_query_probe import RepairQueryCase, load_probe_cases


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple((value.casefold().replace("-", " ").replace("/", " ")).split())


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BridgeFixedContract(StrictContract):
    model: str = Field(min_length=1)
    thinking_mode: Literal["disabled"] = "disabled"
    system_prompt_policy: str = Field(min_length=1)
    retriever_id: str = Field(min_length=1)
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    hypotheses_per_case: Literal[3] = 3
    max_search_results: int = Field(ge=1, le=100)
    max_planner_output_tokens: int = Field(ge=1, le=512)
    provider_attempts_per_case: Literal[1] = 1
    search_calls_per_case: Literal[1] = 1
    sealed_holdout_access: Literal["forbidden"] = "forbidden"


class BridgeAcceptance(StrictContract):
    minimum_oracle_gold_hit_cases: int = Field(ge=1)
    minimum_candidate_gold_hit_cases: int = Field(ge=1)
    minimum_gold_hit_case_delta: int = Field(ge=1)
    parse_failures_must_equal: Literal[0] = 0
    request_failures_must_equal: Literal[0] = 0
    maximum_provider_cost_usd: float = Field(gt=0)


class BridgeProbeRegistration(StrictContract):
    schema_version: Literal["contrastive-bridge-probe-registration-v0"] = (
        "contrastive-bridge-probe-registration-v0"
    )
    status: Literal["registered_pre_provider"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    source_summary: ArtifactReference
    gold_slice: ArtifactReference
    baseline_probe: ArtifactReference
    oracle_replay: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    fixed_contract: BridgeFixedContract
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: BridgeAcceptance
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def boundaries_match_cases(self) -> "BridgeProbeRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("bridge probe IDs must be unique")
        case_count = len(self.query_ids)
        if self.acceptance.minimum_candidate_gold_hit_cases > case_count:
            raise ValueError("bridge gold-hit gate exceeds registered cases")
        if self.acceptance.minimum_oracle_gold_hit_cases > case_count:
            raise ValueError("oracle gold-hit gate exceeds registered cases")
        return self


class BridgeHypothesis(StrictContract):
    bridge_type: Literal["domain", "geography", "topic", "entity"]
    bridge_terms: tuple[str, ...] = Field(min_length=1, max_length=3)
    rationale: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def terms_are_compact(self) -> "BridgeHypothesis":
        if any(not term.strip() or len(term) > 60 for term in self.bridge_terms):
            raise ValueError("bridge terms must be non-empty and compact")
        return self


class BridgeQueryPlan(StrictContract):
    unresolved_obligation: str = Field(min_length=1, max_length=400)
    hypotheses: tuple[BridgeHypothesis, ...] = Field(min_length=3, max_length=3)
    selected_index: int = Field(ge=0, le=2)
    query: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def hypotheses_are_distinct_and_query_is_compact(self) -> "BridgeQueryPlan":
        bridge_types = [item.bridge_type for item in self.hypotheses]
        if len(bridge_types) != len(set(bridge_types)):
            raise ValueError("bridge hypotheses must use distinct types")
        signatures = [
            tuple(token for term in item.bridge_terms for token in _tokens(term))
            for item in self.hypotheses
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("bridge hypotheses must use distinct terms")
        if not 5 <= len(self.query.split()) <= 18:
            raise ValueError("bridge query must contain 5 to 18 terms")
        return self


class BridgeProbeItem(StrictContract):
    query_id: str
    status: Literal["succeeded", "failed"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_query: str
    baseline_gold_hits: tuple[str, ...]
    plan: BridgeQueryPlan | None
    selected_novel_terms: tuple[str, ...]
    candidate_docids: tuple[str, ...]
    candidate_gold_hits: tuple[str, ...]
    candidate_gold_hit_ranks: tuple[int, ...]
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


class BridgeProbeResult(StrictContract):
    schema_version: Literal["contrastive-bridge-probe-v0"] = (
        "contrastive-bridge-probe-v0"
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
    oracle_gold_hit_cases: int = Field(ge=0)
    candidate_gold_hit_cases: int = Field(ge=0)
    gold_hit_case_delta: int
    total_usage: Usage
    provider_cost_observability: Literal["complete", "lower_bound_after_failure"]
    items: tuple[BridgeProbeItem, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "integrate_and_preregister_stage_replay",
        "diagnose_bridge_generation_or_selection",
    ]
    claim_boundary: str


class OracleCase(StrictContract):
    query_id: str
    bridge_type: Literal["domain", "geography", "topic", "entity"]
    bridge_terms: tuple[str, ...] = Field(min_length=1)
    query: str = Field(min_length=1)
    provenance: Literal["manual_after_gold_inspection"]


class BridgeOracleSpec(StrictContract):
    schema_version: Literal["contrastive-bridge-oracle-spec-v0"] = (
        "contrastive-bridge-oracle-spec-v0"
    )
    status: Literal["posthoc_not_preregistered"]
    source_summary: ArtifactReference
    gold_slice: ArtifactReference
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    retriever_id: str
    max_search_results: int = Field(ge=1, le=100)
    cases: tuple[OracleCase, ...] = Field(min_length=1)
    claim_boundary: str


class OracleItem(StrictContract):
    query_id: str
    bridge_type: str
    bridge_terms: tuple[str, ...]
    query: str
    returned_docids: tuple[str, ...]
    gold_hits: tuple[str, ...]
    gold_hit_ranks: tuple[int, ...]
    search_latency_ms: int


class BridgeOracleResult(StrictContract):
    schema_version: Literal["contrastive-bridge-oracle-replay-v0"] = (
        "contrastive-bridge-oracle-replay-v0"
    )
    created_at: str
    status: Literal["posthoc_oracle_diagnostic"]
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int
    search_calls: int
    gold_hit_cases: int
    items: tuple[OracleItem, ...]
    claim_boundary: str


def _validate_artifacts(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")


def load_bridge_registration(path: Path) -> BridgeProbeRegistration:
    registration = BridgeProbeRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            registration.source_summary,
            registration.gold_slice,
            registration.baseline_probe,
            registration.oracle_replay,
            *registration.frozen_artifacts,
        ),
    )
    return registration


def build_bridge_prompt(case: RepairQueryCase, baseline_query: str) -> str:
    return json.dumps(
        {
            "task": (
                "Return exactly one JSON object. Generate three typed bridge hypotheses before "
                "selecting one late-stage retrieval query for the unresolved obligation."
            ),
            "definition": (
                "A bridge is a domain, geography, topic, or entity concept implied by the clue "
                "conjunction but absent from the failed query. It connects surface clues to the "
                "vocabulary likely used by a relevant source."
            ),
            "rules": [
                "Generate exactly three hypotheses with three distinct bridge types.",
                "At least one selected bridge term must be absent from the question, draft, and failed query.",
                "Do not select a refusal sentence or merely recombine the original clue words.",
                "The final query must include every term from the selected bridge hypothesis.",
                "Use 5 to 18 whitespace-delimited terms and do not answer the question.",
            ],
            "input": {
                "question": case.question,
                "audited_draft": case.draft_answer,
                "exact_answer_hypothesis": case.exact_answer,
                "subject_hypothesis": case.subject_hypothesis,
                "audit_reasons": list(case.audit_reasons),
                "failed_constraint_query": baseline_query,
            },
            "output_schema": {
                "unresolved_obligation": "one sentence",
                "hypotheses": [
                    {
                        "bridge_type": "domain | geography | topic | entity",
                        "bridge_terms": ["one to three compact terms"],
                        "rationale": "why this vocabulary connects the clues to a source",
                    }
                ],
                "selected_index": "0, 1, or 2",
                "query": "5 to 18 terms including every selected bridge term",
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_plan_for_case(
    *, case: RepairQueryCase, baseline_query: str, plan: BridgeQueryPlan
) -> tuple[str, ...]:
    if plan.query.casefold() == baseline_query.casefold():
        raise ValueError("bridge planner repeated the failed constraint query")
    selected = plan.hypotheses[plan.selected_index]
    selected_tokens = tuple(
        token for term in selected.bridge_terms for token in _tokens(term)
    )
    query_tokens = set(_tokens(plan.query))
    if not selected_tokens or any(token not in query_tokens for token in selected_tokens):
        raise ValueError("bridge query omits a selected bridge term")
    anchored_tokens = set(
        _tokens(f"{case.question}\n{case.draft_answer}\n{baseline_query}")
    )
    novel = tuple(token for token in selected_tokens if token not in anchored_tokens)
    if not novel:
        raise ValueError("selected bridge adds no term beyond question, draft, and failed query")
    return novel


def _search(search_url: str, run_id: str, query: str, timeout_seconds: int) -> tuple[tuple[str, ...], int]:
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
        raise RuntimeError(f"bridge retrieval failed: {error}") from error
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("bridge retrieval response has no results array")
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


def run_oracle_replay(*, spec_path: Path, output_path: Path, timeout_seconds: int = 60) -> BridgeOracleResult:
    spec = BridgeOracleSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    root = spec_path.resolve().parents[2]
    _validate_artifacts(root, (spec.source_summary, spec.gold_slice))
    if output_path.exists():
        raise ValueError("bridge oracle output already exists")
    gold = json.loads((root / spec.gold_slice.path).read_text(encoding="utf-8"))
    gold_by_id = {
        str(row["query_id"]): tuple(str(item) for item in row["gold_docids"])
        for row in gold["rows"]
    }
    items: list[OracleItem] = []
    for case in spec.cases:
        if case.query_id not in gold_by_id:
            raise ValueError(f"oracle case is absent from gold: {case.query_id}")
        docids, latency = _search(
            spec.search_url,
            f"contrastive-bridge-oracle-{case.query_id}",
            case.query,
            timeout_seconds,
        )
        if len(docids) > spec.max_search_results:
            raise ValueError("oracle retriever exceeded the result cap")
        gold_docids = gold_by_id[case.query_id]
        hits = tuple(item for item in docids if item in gold_docids)
        ranks = tuple(index + 1 for index, item in enumerate(docids) if item in gold_docids)
        items.append(
            OracleItem(
                query_id=case.query_id,
                bridge_type=case.bridge_type,
                bridge_terms=case.bridge_terms,
                query=case.query,
                returned_docids=docids,
                gold_hits=hits,
                gold_hit_ranks=ranks,
                search_latency_ms=latency,
            )
        )
    result = BridgeOracleResult(
        created_at=_utc_now(),
        status="posthoc_oracle_diagnostic",
        spec_sha256=_sha256_file(spec_path),
        query_count=len(items),
        search_calls=len(items),
        gold_hit_cases=sum(bool(item.gold_hits) for item in items),
        items=tuple(items),
        claim_boundary=spec.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_bridge_probe(
    *, registration_path: Path, provider: LLMProvider, output_path: Path, timeout_seconds: int = 60
) -> BridgeProbeResult:
    registration = load_bridge_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("bridge probe output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("bridge probe output already exists")
    if provider.model != registration.fixed_contract.model:
        raise ValueError("provider model differs from the bridge registration")
    cases = load_probe_cases(repository_root=root, registration=registration)  # type: ignore[arg-type]
    baseline = json.loads((root / registration.baseline_probe.path).read_text(encoding="utf-8"))
    baseline_by_id = {str(item["query_id"]): item for item in baseline["items"]}
    oracle = BridgeOracleResult.model_validate_json(
        (root / registration.oracle_replay.path).read_text(encoding="utf-8")
    )
    oracle_ids = {item.query_id for item in oracle.items if item.gold_hits}
    items: list[BridgeProbeItem] = []
    total_usage = Usage()
    provider_attempts = search_calls = parse_failures = request_failures = 0
    observability: Literal["complete", "lower_bound_after_failure"] = "complete"
    for case in cases:
        baseline_item = baseline_by_id[case.query_id]
        baseline_query = str(baseline_item["plan"]["query"])
        baseline_hits = tuple(str(item) for item in baseline_item["candidate_gold_hits"])
        prompt = build_bridge_prompt(case, baseline_query)
        prompt_hash = _sha256_text(prompt)
        usage = Usage()
        provider_latency = search_latency = 0
        try:
            provider_attempts += 1
            completion = provider.complete(
                stage="bridge_hypothesis",
                prompt=prompt,
                json_output=True,
                max_output_tokens=registration.fixed_contract.max_planner_output_tokens,
            )
            usage = completion.usage
            provider_latency = completion.latency_ms
            total_usage = _add_usage(total_usage, usage)
            try:
                plan = BridgeQueryPlan.model_validate_json(completion.text)
                novel = validate_plan_for_case(
                    case=case, baseline_query=baseline_query, plan=plan
                )
            except Exception as error:
                parse_failures += 1
                items.append(
                    BridgeProbeItem(
                        query_id=case.query_id,
                        status="failed",
                        prompt_sha256=prompt_hash,
                        baseline_query=baseline_query,
                        baseline_gold_hits=baseline_hits,
                        plan=None,
                        selected_novel_terms=(),
                        candidate_docids=(),
                        candidate_gold_hits=(),
                        candidate_gold_hit_ranks=(),
                        usage=usage,
                        provider_latency_ms=provider_latency,
                        search_latency_ms=0,
                        error=f"plan_parse_error: {type(error).__name__}: {error}",
                    )
                )
                continue
            search_calls += 1
            docids, search_latency = _search(
                registration.fixed_contract.search_url,
                f"contrastive-bridge-probe-{case.query_id}",
                plan.query,
                timeout_seconds,
            )
            if len(docids) > registration.fixed_contract.max_search_results:
                raise ValueError("bridge retriever exceeded the registered result cap")
            hits = tuple(item for item in docids if item in case.gold_docids)
            ranks = tuple(index + 1 for index, item in enumerate(docids) if item in case.gold_docids)
            items.append(
                BridgeProbeItem(
                    query_id=case.query_id,
                    status="succeeded",
                    prompt_sha256=prompt_hash,
                    baseline_query=baseline_query,
                    baseline_gold_hits=baseline_hits,
                    plan=plan,
                    selected_novel_terms=novel,
                    candidate_docids=docids,
                    candidate_gold_hits=hits,
                    candidate_gold_hit_ranks=ranks,
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
                BridgeProbeItem(
                    query_id=case.query_id,
                    status="failed",
                    prompt_sha256=prompt_hash,
                    baseline_query=baseline_query,
                    baseline_gold_hits=baseline_hits,
                    plan=None,
                    selected_novel_terms=(),
                    candidate_docids=(),
                    candidate_gold_hits=(),
                    candidate_gold_hit_ranks=(),
                    usage=usage,
                    provider_latency_ms=provider_latency,
                    search_latency_ms=search_latency,
                    error=f"request_error: {type(error).__name__}: {error}",
                )
            )
    baseline_hits = sum(bool(item.baseline_gold_hits) for item in items)
    candidate_hits = sum(bool(item.candidate_gold_hits) for item in items)
    delta = candidate_hits - baseline_hits
    completed = sum(item.status == "succeeded" for item in items)
    acceptance = registration.acceptance
    gates = (
        DecisionGate(gate_id="completed_cases", observed=completed, operator="eq", threshold=len(cases), passed=completed == len(cases)),
        DecisionGate(gate_id="provider_attempts", observed=provider_attempts, operator="eq", threshold=len(cases), passed=provider_attempts == len(cases)),
        DecisionGate(gate_id="search_calls", observed=search_calls, operator="eq", threshold=len(cases), passed=search_calls == len(cases)),
        DecisionGate(gate_id="parse_failures", observed=parse_failures, operator="eq", threshold=0, passed=parse_failures == 0),
        DecisionGate(gate_id="request_failures", observed=request_failures, operator="eq", threshold=0, passed=request_failures == 0),
        DecisionGate(gate_id="oracle_gold_hit_cases", observed=len(oracle_ids), operator="ge", threshold=acceptance.minimum_oracle_gold_hit_cases, passed=len(oracle_ids) >= acceptance.minimum_oracle_gold_hit_cases),
        DecisionGate(gate_id="candidate_gold_hit_cases", observed=candidate_hits, operator="ge", threshold=acceptance.minimum_candidate_gold_hit_cases, passed=candidate_hits >= acceptance.minimum_candidate_gold_hit_cases),
        DecisionGate(gate_id="gold_hit_case_delta", observed=delta, operator="ge", threshold=acceptance.minimum_gold_hit_case_delta, passed=delta >= acceptance.minimum_gold_hit_case_delta),
        DecisionGate(gate_id="provider_cost_usd", observed=total_usage.estimated_cost_usd, operator="le", threshold=acceptance.maximum_provider_cost_usd, passed=total_usage.estimated_cost_usd <= acceptance.maximum_provider_cost_usd),
    )
    decision: Literal["pass", "reject"] = "pass" if all(item.passed for item in gates) else "reject"
    result = BridgeProbeResult(
        created_at=_utc_now(),
        status="succeeded" if parse_failures == 0 and request_failures == 0 else "failed",
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=len(cases),
        provider_attempts=provider_attempts,
        search_calls=search_calls,
        parse_failures=parse_failures,
        request_failures=request_failures,
        baseline_gold_hit_cases=baseline_hits,
        oracle_gold_hit_cases=len(oracle_ids),
        candidate_gold_hit_cases=candidate_hits,
        gold_hit_case_delta=delta,
        total_usage=total_usage,
        provider_cost_observability=observability,
        items=tuple(items),
        gates=gates,
        next_action="integrate_and_preregister_stage_replay" if decision == "pass" else "diagnose_bridge_generation_or_selection",
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
