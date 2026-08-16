from __future__ import annotations

import json
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


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProbeFixedContract(StrictContract):
    model: str = Field(min_length=1)
    thinking_mode: Literal["disabled"] = "disabled"
    system_prompt_policy: str = Field(min_length=1)
    retriever_id: str = Field(min_length=1)
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    max_search_results: int = Field(ge=1, le=100)
    max_planner_output_tokens: int = Field(ge=1, le=512)
    provider_attempts_per_case: Literal[1] = 1
    search_calls_per_case: Literal[1] = 1
    sealed_holdout_access: Literal["forbidden"] = "forbidden"


class ProbeAcceptance(StrictContract):
    minimum_candidate_gold_hit_cases: int = Field(ge=1)
    minimum_gold_hit_case_delta: int = Field(ge=1)
    parse_failures_must_equal: Literal[0] = 0
    request_failures_must_equal: Literal[0] = 0
    maximum_provider_cost_usd: float = Field(gt=0)


class ProbeOperationalRecovery(StrictContract):
    failure_artifact: ArtifactReference
    failure_class: Literal["deepseek_json_mode_instruction_missing"]
    official_contract_url: str = Field(pattern=r"^https://api-docs\.deepseek\.com/")
    validation_change: str = Field(min_length=1)
    previous_provider_attempts: int = Field(gt=0)
    maximum_additional_provider_attempts: int = Field(gt=0)
    prior_cost_observability: Literal["unknown_http_400_usage"]


class RepairQueryProbeRegistration(StrictContract):
    schema_version: Literal["obligation-repair-query-probe-registration-v0"] = (
        "obligation-repair-query-probe-registration-v0"
    )
    status: Literal["registered_pre_provider"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    source_summary: ArtifactReference
    gold_slice: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    fixed_contract: ProbeFixedContract
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: ProbeAcceptance
    operational_recovery: ProbeOperationalRecovery | None = None
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def query_ids_are_unique(self) -> "RepairQueryProbeRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("repair-query probe IDs must be unique")
        if self.acceptance.minimum_candidate_gold_hit_cases > len(self.query_ids):
            raise ValueError("candidate gold-hit gate exceeds registered cases")
        if (
            self.operational_recovery is not None
            and self.operational_recovery.maximum_additional_provider_attempts
            != len(self.query_ids)
        ):
            raise ValueError("recovery attempt cap must equal the registered case count")
        return self


class RepairQueryPlan(StrictContract):
    query: str = Field(min_length=1, max_length=300)
    unresolved_obligation: str = Field(min_length=1, max_length=400)
    strategy: Literal[
        "constraint_recombination",
        "counter_hypothesis",
        "entity_verification",
        "relation_verification",
    ]

    @model_validator(mode="after")
    def query_is_search_shaped(self) -> "RepairQueryPlan":
        terms = self.query.split()
        if not 5 <= len(terms) <= 18:
            raise ValueError("repair query must contain 5 to 18 whitespace-delimited terms")
        lowered = self.query.casefold()
        forbidden = (
            "cannot be determined",
            "unable to determine",
            "available evidence",
            "insufficient evidence",
        )
        if any(phrase in lowered for phrase in forbidden):
            raise ValueError("repair query must not search a refusal or uncertainty sentence")
        return self


class RepairQueryCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    draft_answer: str = Field(min_length=1)
    draft_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_answer: str = Field(min_length=1)
    subject_hypothesis: str | None
    audit_reasons: tuple[str, ...] = Field(min_length=1)
    uncertainty_phrases: tuple[str, ...]
    failed_repair_query: str = Field(min_length=1)
    baseline_docids: tuple[str, ...]
    gold_docids: tuple[str, ...] = Field(min_length=1)


class ProbeItem(StrictContract):
    query_id: str
    status: Literal["succeeded", "failed"]
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan: RepairQueryPlan | None
    baseline_query: str
    baseline_docids: tuple[str, ...]
    baseline_gold_hits: tuple[str, ...]
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


class RepairQueryProbeResult(StrictContract):
    schema_version: Literal["obligation-repair-query-probe-v0"] = (
        "obligation-repair-query-probe-v0"
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
    recovery_status: Literal["none", "json_instruction_protocol_recovery"]
    previous_provider_attempts: int = Field(ge=0)
    total_provider_attempts_including_prior: int = Field(ge=0)
    items: tuple[ProbeItem, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "integrate_and_preregister_stage_replay",
        "diagnose_query_planner_or_retriever",
    ]
    claim_boundary: str


def load_registration(path: Path) -> RepairQueryProbeRegistration:
    registration = RepairQueryProbeRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    artifacts = [
        registration.source_summary,
        registration.gold_slice,
        *registration.frozen_artifacts,
    ]
    if registration.operational_recovery is not None:
        artifacts.append(registration.operational_recovery.failure_artifact)
    for artifact in artifacts:
        artifact_path = (root / artifact.path).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError(f"registered artifact escapes repository: {artifact.path}")
        if not artifact_path.is_file():
            raise ValueError(f"registered artifact is missing: {artifact.path}")
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")
    return registration


def load_probe_cases(
    *,
    repository_root: Path,
    registration: RepairQueryProbeRegistration,
) -> tuple[RepairQueryCase, ...]:
    summary_path = repository_root / registration.source_summary.path
    gold_path = repository_root / registration.gold_slice.path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    summary_items = {str(item["query_id"]): item for item in summary["items"]}
    gold_rows = {str(item["query_id"]): item for item in gold["rows"]}
    source_dir = summary_path.parent
    cases: list[RepairQueryCase] = []
    for query_id in registration.query_ids:
        summary_item = summary_items.get(query_id)
        gold_row = gold_rows.get(query_id)
        if summary_item is None or gold_row is None:
            raise ValueError(f"registered query is absent from source or gold: {query_id}")
        if summary_item["status"] != "succeeded":
            raise ValueError(f"probe source query did not succeed: {query_id}")
        run_path = source_dir / query_id / "run.json"
        request_path = source_dir / query_id / "request.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        request = json.loads(request_path.read_text(encoding="utf-8"))
        audit = run.get("answer_first_audit")
        if not isinstance(audit, dict) or audit.get("audit_status") != "open":
            raise ValueError(f"probe source does not have open evidence debt: {query_id}")
        draft_hash = str(audit["draft_answer_sha256"])
        draft_candidates: list[str] = []
        for message in run.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for content in message.get("content", []):
                if content.get("type") == "text" and isinstance(content.get("text"), str):
                    text = content["text"]
                    if _sha256_bytes(text.encode("utf-8")) == draft_hash:
                        draft_candidates.append(text)
        if len(draft_candidates) != 1:
            raise ValueError(f"could not bind exactly one audited draft: {query_id}")
        failed_query = audit.get("repair_query")
        if not isinstance(failed_query, str) or not failed_query.strip():
            raise ValueError(f"probe source has no executed repair query: {query_id}")
        cases.append(
            RepairQueryCase(
                query_id=query_id,
                question=str(request["question"]),
                draft_answer=draft_candidates[0],
                draft_answer_sha256=draft_hash,
                exact_answer=str(audit["exact_answer"]),
                subject_hypothesis=audit.get("subject_hypothesis"),
                audit_reasons=tuple(str(item) for item in audit["reasons"]),
                uncertainty_phrases=tuple(
                    str(item) for item in audit["explicit_uncertainty_phrases"]
                ),
                failed_repair_query=failed_query,
                baseline_docids=tuple(str(item) for item in audit["repair_returned_docids"]),
                gold_docids=tuple(str(item) for item in gold_row["gold_docids"]),
            )
        )
    return tuple(cases)


def build_repair_query_prompt(case: RepairQueryCase) -> str:
    payload = {
        "task": (
            "Return exactly one JSON object containing one late-stage retrieval query for the "
            "unresolved answer-critical obligation. The draft and its entity/domain guesses "
            "are hypotheses, not evidence."
        ),
        "rules": [
            "Never search a refusal, uncertainty sentence, or the phrase available evidence.",
            "Do not merely repeat the failed repair query.",
            "Use the original question to identify the requested field and two rare constraints.",
            "When the draft may have chosen the wrong entity or profession, prefer a plausible counter-hypothesis.",
            "Use discovered names only as tentative anchors and combine them with the unresolved relation.",
            "Return 5 to 18 whitespace-delimited search terms, not prose and not an answer.",
        ],
        "input": {
            "question": case.question,
            "audited_draft": case.draft_answer,
            "exact_answer_hypothesis": case.exact_answer,
            "subject_hypothesis": case.subject_hypothesis,
            "audit_reasons": list(case.audit_reasons),
            "uncertainty_phrases": list(case.uncertainty_phrases),
            "failed_repair_query": case.failed_repair_query,
        },
        "output_schema": {
            "query": "5 to 18 search terms",
            "unresolved_obligation": "one short sentence",
            "strategy": (
                "constraint_recombination | counter_hypothesis | entity_verification | "
                "relation_verification"
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _search(
    *, search_url: str, run_id: str, query: str, timeout_seconds: int
) -> tuple[tuple[str, ...], int]:
    request = Request(
        search_url,
        data=json.dumps({"run_id": run_id, "query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = datetime.now(timezone.utc)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"repair-query retrieval failed: {error}") from error
    elapsed = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("repair-query retrieval response has no results array")
    docids = tuple(str(item["docid"]) for item in results)
    return docids, elapsed


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        input_cache_hit_tokens=(
            left.input_cache_hit_tokens + right.input_cache_hit_tokens
        ),
        input_cache_miss_tokens=(
            left.input_cache_miss_tokens + right.input_cache_miss_tokens
        ),
        estimated_cost_usd=left.estimated_cost_usd + right.estimated_cost_usd,
    )


def run_probe(
    *,
    registration_path: Path,
    provider: LLMProvider,
    output_path: Path,
    timeout_seconds: int = 60,
) -> RepairQueryProbeResult:
    registration = load_registration(registration_path)
    repository_root = registration_path.resolve().parents[2]
    output_resolved = output_path.resolve()
    if not output_resolved.is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("repair-query probe output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("repair-query probe output already exists")
    if provider.model != registration.fixed_contract.model:
        raise ValueError("provider model differs from the registered repair-query model")
    cases = load_probe_cases(
        repository_root=repository_root,
        registration=registration,
    )
    items: list[ProbeItem] = []
    total_usage = Usage()
    provider_attempts = 0
    search_calls = 0
    parse_failures = 0
    request_failures = 0
    cost_observability: Literal["complete", "lower_bound_after_failure"] = "complete"
    if registration.operational_recovery is not None:
        cost_observability = "lower_bound_after_failure"
    for case in cases:
        prompt = build_repair_query_prompt(case)
        prompt_hash = _sha256_bytes(prompt.encode("utf-8"))
        usage = Usage()
        provider_latency_ms = 0
        search_latency_ms = 0
        try:
            provider_attempts += 1
            completion = provider.complete(
                stage="repair_query",
                prompt=prompt,
                json_output=True,
                max_output_tokens=registration.fixed_contract.max_planner_output_tokens,
            )
            usage = completion.usage
            provider_latency_ms = completion.latency_ms
            total_usage = _add_usage(total_usage, usage)
            try:
                plan = RepairQueryPlan.model_validate_json(completion.text)
            except Exception as error:
                parse_failures += 1
                items.append(
                    ProbeItem(
                        query_id=case.query_id,
                        status="failed",
                        prompt_sha256=prompt_hash,
                        plan=None,
                        baseline_query=case.failed_repair_query,
                        baseline_docids=case.baseline_docids,
                        baseline_gold_hits=tuple(
                            item for item in case.baseline_docids if item in case.gold_docids
                        ),
                        candidate_docids=(),
                        candidate_gold_hits=(),
                        candidate_gold_hit_ranks=(),
                        usage=usage,
                        provider_latency_ms=provider_latency_ms,
                        search_latency_ms=0,
                        error=f"plan_parse_error: {type(error).__name__}: {error}",
                    )
                )
                continue
            if plan.query.casefold() == case.failed_repair_query.casefold():
                raise ValueError("planner repeated the failed repair query")
            search_calls += 1
            docids, search_latency_ms = _search(
                search_url=registration.fixed_contract.search_url,
                run_id=f"obligation-repair-query-probe-{case.query_id}",
                query=plan.query,
                timeout_seconds=timeout_seconds,
            )
            if len(docids) > registration.fixed_contract.max_search_results:
                raise ValueError("retriever returned more results than the registered cap")
            hits = tuple(item for item in docids if item in case.gold_docids)
            ranks = tuple(index + 1 for index, item in enumerate(docids) if item in case.gold_docids)
            items.append(
                ProbeItem(
                    query_id=case.query_id,
                    status="succeeded",
                    prompt_sha256=prompt_hash,
                    plan=plan,
                    baseline_query=case.failed_repair_query,
                    baseline_docids=case.baseline_docids,
                    baseline_gold_hits=tuple(
                        item for item in case.baseline_docids if item in case.gold_docids
                    ),
                    candidate_docids=docids,
                    candidate_gold_hits=hits,
                    candidate_gold_hit_ranks=ranks,
                    usage=usage,
                    provider_latency_ms=provider_latency_ms,
                    search_latency_ms=search_latency_ms,
                    error=None,
                )
            )
        except Exception as error:
            request_failures += 1
            cost_observability = "lower_bound_after_failure"
            items.append(
                ProbeItem(
                    query_id=case.query_id,
                    status="failed",
                    prompt_sha256=prompt_hash,
                    plan=None,
                    baseline_query=case.failed_repair_query,
                    baseline_docids=case.baseline_docids,
                    baseline_gold_hits=tuple(
                        item for item in case.baseline_docids if item in case.gold_docids
                    ),
                    candidate_docids=(),
                    candidate_gold_hits=(),
                    candidate_gold_hit_ranks=(),
                    usage=usage,
                    provider_latency_ms=provider_latency_ms,
                    search_latency_ms=search_latency_ms,
                    error=f"request_error: {type(error).__name__}: {error}",
                )
            )
    baseline_hits = sum(bool(item.baseline_gold_hits) for item in items)
    candidate_hits = sum(bool(item.candidate_gold_hits) for item in items)
    delta = candidate_hits - baseline_hits
    acceptance = registration.acceptance
    gates = (
        DecisionGate(
            gate_id="completed_cases",
            observed=sum(item.status == "succeeded" for item in items),
            operator="eq",
            threshold=len(cases),
            passed=all(item.status == "succeeded" for item in items),
        ),
        DecisionGate(
            gate_id="provider_attempts",
            observed=provider_attempts,
            operator="eq",
            threshold=len(cases),
            passed=provider_attempts == len(cases),
        ),
        DecisionGate(
            gate_id="search_calls",
            observed=search_calls,
            operator="eq",
            threshold=len(cases),
            passed=search_calls == len(cases),
        ),
        DecisionGate(
            gate_id="parse_failures",
            observed=parse_failures,
            operator="eq",
            threshold=acceptance.parse_failures_must_equal,
            passed=parse_failures == acceptance.parse_failures_must_equal,
        ),
        DecisionGate(
            gate_id="request_failures",
            observed=request_failures,
            operator="eq",
            threshold=acceptance.request_failures_must_equal,
            passed=request_failures == acceptance.request_failures_must_equal,
        ),
        DecisionGate(
            gate_id="candidate_gold_hit_cases",
            observed=candidate_hits,
            operator="ge",
            threshold=acceptance.minimum_candidate_gold_hit_cases,
            passed=candidate_hits >= acceptance.minimum_candidate_gold_hit_cases,
        ),
        DecisionGate(
            gate_id="gold_hit_case_delta",
            observed=delta,
            operator="ge",
            threshold=acceptance.minimum_gold_hit_case_delta,
            passed=delta >= acceptance.minimum_gold_hit_case_delta,
        ),
        DecisionGate(
            gate_id="provider_cost_usd",
            observed=total_usage.estimated_cost_usd,
            operator="le",
            threshold=acceptance.maximum_provider_cost_usd,
            passed=(
                total_usage.estimated_cost_usd
                <= acceptance.maximum_provider_cost_usd
            ),
        ),
    )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = RepairQueryProbeResult(
        created_at=_utc_now(),
        status="succeeded" if request_failures == 0 and parse_failures == 0 else "failed",
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=len(cases),
        provider_attempts=provider_attempts,
        search_calls=search_calls,
        parse_failures=parse_failures,
        request_failures=request_failures,
        baseline_gold_hit_cases=baseline_hits,
        candidate_gold_hit_cases=candidate_hits,
        gold_hit_case_delta=delta,
        total_usage=total_usage,
        provider_cost_observability=cost_observability,
        recovery_status=(
            "json_instruction_protocol_recovery"
            if registration.operational_recovery is not None
            else "none"
        ),
        previous_provider_attempts=(
            registration.operational_recovery.previous_provider_attempts
            if registration.operational_recovery is not None
            else 0
        ),
        total_provider_attempts_including_prior=(
            provider_attempts
            + (
                registration.operational_recovery.previous_provider_attempts
                if registration.operational_recovery is not None
                else 0
            )
        ),
        items=tuple(items),
        gates=gates,
        next_action=(
            "integrate_and_preregister_stage_replay"
            if decision == "pass"
            else "diagnose_query_planner_or_retriever"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return result
