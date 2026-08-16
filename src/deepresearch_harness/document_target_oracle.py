from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch_harness.evidence_span_oracle import (
    ArtifactReference,
    DecisionGate,
    derive_answer_obligation_query_v1,
)


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "been",
    "before",
    "does",
    "for",
    "from",
    "give",
    "have",
    "into",
    "name",
    "that",
    "the",
    "their",
    "this",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _terms(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in _TOKEN.findall(value.casefold())
            if (len(token) >= 3 or token.isdigit()) and token not in _STOPWORDS
        )
    )


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetSlateSearchCall(StrictContract):
    search_call_index: int = Field(ge=1)
    query: str = Field(min_length=1)
    matched_obligation_terms: tuple[str, ...]
    obligation_term_coverage: float = Field(ge=0.0, le=1.0)


class TargetDocument(StrictContract):
    docid: str = Field(min_length=1)
    channel: Literal["bm25_anchor", "dense_lead"]
    search_call_index: int = Field(ge=1)
    result_rank: int = Field(ge=1)
    search_query: str = Field(min_length=1)
    matched_obligation_terms: tuple[str, ...]
    obligation_term_coverage: float = Field(ge=0.0, le=1.0)


class ObligationTargetSlate(StrictContract):
    selector_id: Literal["obligation_channel_slate_v0"] = (
        "obligation_channel_slate_v0"
    )
    question: str = Field(min_length=1)
    obligation_query: str = Field(min_length=1)
    selected_search_calls: tuple[TargetSlateSearchCall, ...] = Field(min_length=1)
    targets: tuple[TargetDocument, ...] = Field(min_length=1)


class DocumentTargetOracleAcceptance(StrictContract):
    minimum_gold_target_hit_cases: int = Field(ge=1)
    maximum_targets_per_case: int = Field(ge=1)


class DocumentTargetOracleRegistration(StrictContract):
    schema_version: Literal["document-target-oracle-registration-v0"] = (
        "document-target-oracle-registration-v0"
    )
    status: Literal["posthoc_registered_for_new_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    candidate_run_root: str = Field(min_length=1)
    gold_slice: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    selector_id: Literal["obligation_channel_slate_v0"] = (
        "obligation_channel_slate_v0"
    )
    maximum_search_calls: int = Field(ge=1, le=8)
    slots_per_channel: int = Field(ge=1, le=3)
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: DocumentTargetOracleAcceptance
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def gate_is_valid(self) -> "DocumentTargetOracleRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("document-target query IDs must be unique")
        if self.acceptance.minimum_gold_target_hit_cases > len(self.query_ids):
            raise ValueError("document-target hit gate exceeds registered cases")
        maximum_possible_targets = (
            self.maximum_search_calls * self.slots_per_channel * 2
        )
        if self.acceptance.maximum_targets_per_case < maximum_possible_targets:
            raise ValueError("target cap is lower than the selector's configured maximum")
        return self


class DocumentTargetOracleCaseResult(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    selected_gold_docids: tuple[str, ...]
    gold_target_hit: bool
    slate: ObligationTargetSlate


class DocumentTargetOracleResult(StrictContract):
    schema_version: Literal["document-target-oracle-v0"] = (
        "document-target-oracle-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded"] = "succeeded"
    decision: Literal["pass", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    gold_target_hit_cases: int = Field(ge=0)
    total_selected_targets: int = Field(ge=0)
    maximum_selected_targets_in_case: int = Field(ge=0)
    provider_calls: Literal[0] = 0
    new_search_calls: Literal[0] = 0
    document_open_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    items: tuple[DocumentTargetOracleCaseResult, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "preregister_bounded_post_run_overlay",
        "reject_target_slate_and_rediagnose",
    ]
    claim_boundary: str = Field(min_length=1)


def select_obligation_target_slate(
    run: dict[str, object],
    *,
    maximum_search_calls: int = 2,
    slots_per_channel: int = 1,
) -> ObligationTargetSlate:
    if maximum_search_calls < 1 or slots_per_channel < 1:
        raise ValueError("target-slate budgets must be positive")
    question = _extract_question(run)
    obligation = derive_answer_obligation_query_v1(question)
    obligation_terms = _terms(obligation)
    if not obligation_terms:
        obligation_terms = _terms(question)
    raw_search_calls = run.get("search_calls")
    if not isinstance(raw_search_calls, list) or not raw_search_calls:
        raise ValueError("run trace has no recorded search calls")

    ranked_calls: list[tuple[int, float, int, dict[str, object], tuple[str, ...]]] = []
    for zero_based_index, raw_call in enumerate(raw_search_calls):
        if not isinstance(raw_call, dict) or raw_call.get("outcome") != "ok":
            raise ValueError("target slate requires only successful recorded searches")
        query = raw_call.get("query")
        results = raw_call.get("results")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("recorded search query is missing")
        if not isinstance(results, list) or not results:
            raise ValueError("successful recorded search has no results")
        matched = tuple(term for term in obligation_terms if term in set(_terms(query)))
        coverage = len(matched) / len(obligation_terms)
        ranked_calls.append(
            (len(matched), coverage, zero_based_index, raw_call, matched)
        )
    ranked_calls.sort(key=lambda row: (-row[0], -row[1], row[2]))
    selected_calls = ranked_calls[:maximum_search_calls]

    seen_docids: set[str] = set()
    selected_targets: list[TargetDocument] = []
    call_records: list[TargetSlateSearchCall] = []
    for _matched_count, coverage, zero_based_index, raw_call, matched in selected_calls:
        query = str(raw_call["query"])
        call_records.append(
            TargetSlateSearchCall(
                search_call_index=zero_based_index + 1,
                query=query,
                matched_obligation_terms=matched,
                obligation_term_coverage=coverage,
            )
        )
        results = raw_call["results"]
        assert isinstance(results, list)
        by_channel: dict[str, list[tuple[int, dict[str, object]]]] = {
            "bm25_anchor": [],
            "dense_lead": [],
        }
        for zero_based_rank, result in enumerate(results):
            if not isinstance(result, dict):
                raise ValueError("recorded search result is not an object")
            snippet = result.get("snippet")
            if not isinstance(snippet, str):
                raise ValueError("recorded search result has no snippet")
            channel = _result_channel(snippet)
            by_channel[channel].append((zero_based_rank, result))
        for channel in ("bm25_anchor", "dense_lead"):
            added = 0
            for zero_based_rank, result in by_channel[channel]:
                docid = str(result.get("docid", "")).strip()
                if not docid:
                    raise ValueError("recorded search result has no docid")
                if docid in seen_docids:
                    continue
                selected_targets.append(
                    TargetDocument(
                        docid=docid,
                        channel=channel,
                        search_call_index=zero_based_index + 1,
                        result_rank=zero_based_rank + 1,
                        search_query=query,
                        matched_obligation_terms=matched,
                        obligation_term_coverage=coverage,
                    )
                )
                seen_docids.add(docid)
                added += 1
                if added >= slots_per_channel:
                    break
    if not selected_targets:
        raise ValueError("target slate selected no documents")
    return ObligationTargetSlate(
        question=question,
        obligation_query=obligation,
        selected_search_calls=tuple(call_records),
        targets=tuple(selected_targets),
    )


def load_document_target_oracle_registration(
    path: Path,
) -> DocumentTargetOracleRegistration:
    registration = DocumentTargetOracleRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(root, (registration.gold_slice, *registration.frozen_artifacts))
    run_root = (root / registration.candidate_run_root).resolve()
    if not run_root.is_relative_to(root) or not run_root.is_dir():
        raise ValueError("candidate run root is missing or escapes the repository")
    return registration


def run_document_target_oracle(
    *, registration_path: Path, output_path: Path
) -> DocumentTargetOracleResult:
    registration = load_document_target_oracle_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("document-target output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("document-target output already exists")
    gold = json.loads((root / registration.gold_slice.path).read_text(encoding="utf-8"))
    gold_by_id = {str(row["query_id"]): row for row in gold["rows"]}

    items: list[DocumentTargetOracleCaseResult] = []
    for query_id in registration.query_ids:
        if query_id not in gold_by_id:
            raise ValueError(f"registered target case is absent from gold slice: {query_id}")
        run_path = root / registration.candidate_run_root / query_id / "run.json"
        if not run_path.is_file():
            raise ValueError(f"candidate run is missing: {run_path}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        slate = select_obligation_target_slate(
            run,
            maximum_search_calls=registration.maximum_search_calls,
            slots_per_channel=registration.slots_per_channel,
        )
        gold_docids = tuple(str(value) for value in gold_by_id[query_id]["gold_docids"])
        selected_docids = {target.docid for target in slate.targets}
        selected_gold_docids = tuple(
            docid for docid in gold_docids if docid in selected_docids
        )
        items.append(
            DocumentTargetOracleCaseResult(
                query_id=query_id,
                gold_docids=gold_docids,
                selected_gold_docids=selected_gold_docids,
                gold_target_hit=bool(selected_gold_docids),
                slate=slate,
            )
        )

    hit_cases = sum(item.gold_target_hit for item in items)
    total_targets = sum(len(item.slate.targets) for item in items)
    maximum_targets = max((len(item.slate.targets) for item in items), default=0)
    acceptance = registration.acceptance
    target_cap_violations = sum(
        len(item.slate.targets) > acceptance.maximum_targets_per_case
        for item in items
    )
    gates = (
        DecisionGate(
            gate_id="gold_target_hit_cases",
            observed=hit_cases,
            operator="ge",
            threshold=acceptance.minimum_gold_target_hit_cases,
            passed=hit_cases >= acceptance.minimum_gold_target_hit_cases,
        ),
        DecisionGate(
            gate_id="target_cap_violations",
            observed=target_cap_violations,
            operator="eq",
            threshold=0,
            passed=target_cap_violations == 0,
        ),
    )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = DocumentTargetOracleResult(
        created_at=_utc_now(),
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=len(items),
        gold_target_hit_cases=hit_cases,
        total_selected_targets=total_targets,
        maximum_selected_targets_in_case=maximum_targets,
        items=tuple(items),
        gates=gates,
        next_action=(
            "preregister_bounded_post_run_overlay"
            if decision == "pass"
            else "reject_target_slate_and_rediagnose"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _extract_question(run: dict[str, object]) -> str:
    messages = run.get("messages")
    if not isinstance(messages, list):
        raise ValueError("run trace has no messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = str(part.get("text", ""))
            if "Question:" not in text:
                continue
            question = text.split("Question:", 1)[1].split(
                "Your response should be", 1
            )[0].strip()
            if question:
                return question
    raise ValueError("could not extract question from run trace")


def _result_channel(snippet: str) -> Literal["bm25_anchor", "dense_lead"]:
    if snippet.startswith("[BM25 anchor:"):
        return "bm25_anchor"
    if snippet.startswith("[Dense lead preview;"):
        return "dense_lead"
    raise ValueError("recorded search result has an unknown retrieval channel")


def _validate_artifacts(root: Path, artifacts: Sequence[ArtifactReference]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")
