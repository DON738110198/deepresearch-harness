from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_span_oracle import ArtifactReference
from .passage_index_gate import IndexFileDigest, sha256_file


Analyze = Callable[[str], Sequence[str]]
DocumentFrequency = Callable[[str], int]
DocumentSearch = Callable[[str, int], Sequence[str]]

_LEADING_WRAPPER = re.compile(r"\A\s*\[[^\r\n]*\]\s*(?:\r?\n)?")
_FRONTMATTER = re.compile(r"\A\s*---\s*\r?\n.*?\r?\n---\s*(?:\r?\n)?", re.DOTALL)
_CAPITALIZED_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9'-]{2,39}\b")


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectorInput(StrictContract):
    question_source: Literal["saved_request_json"]
    query_and_evidence_source: Literal["saved_successful_run_search_calls"]
    gold_documents_available_to_selector: Literal[False]
    gold_answer_available_to_selector: Literal[False]


class BodyExtraction(StrictContract):
    strip_leading_bracket_wrapper: Literal[True]
    strip_yaml_frontmatter: Literal[True]
    frontmatter_only_candidates_must_equal: Literal[0]


class CandidatePolicy(StrictContract):
    surface_pattern_id: Literal["ascii_capitalized_token_v0"]
    analyzer_id: Literal["pyserini_default_english_v1.2.0"]
    minimum_surface_characters: int = Field(ge=1, le=40)
    maximum_surface_characters: int = Field(ge=1, le=80)
    require_single_analyzed_term: Literal[True]
    maximum_document_frequency: int = Field(ge=2)
    exclude_question_and_recorded_query_terms: Literal[True]
    candidate_order: Literal[
        "document_frequency_ascending_then_support_descending_then_first_seen"
    ]
    slate_size_per_case: int = Field(ge=1, le=8)
    source_query_policy: Literal["first_supporting_query"]

    @model_validator(mode="after")
    def lengths_are_consistent(self) -> "CandidatePolicy":
        if self.minimum_surface_characters > self.maximum_surface_characters:
            raise ValueError("gold-blind pivot surface bounds are reversed")
        return self


class SlateRetrieval(StrictContract):
    composition: Literal["append_surface_form_to_provenance_query_v0"]
    analyzer_id: Literal["pyserini_default_english_v1.2.0"]
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
    top_k: Literal[20]
    maximum_pivot_queries_per_case: int = Field(ge=1, le=8)


class SlateAcceptance(StrictContract):
    minimum_selector_gold_doc_recall_at20_cases: int = Field(ge=1)
    minimum_retained_oracle_rescue_cases: int = Field(ge=1)
    required_selection_failures: Literal[0]
    required_frontmatter_only_selected_candidates: Literal[0]


class SlateBudgets(StrictContract):
    maximum_offline_bm25_queries: int = Field(gt=0)
    maximum_provider_calls: Literal[0]
    maximum_online_search_calls: Literal[0]
    maximum_judge_calls: Literal[0]


class GoldBlindPivotRegistration(StrictContract):
    schema_version: Literal["gold-blind-visible-pivot-slate-registration-v0"]
    status: Literal["posthoc_registered_after_oracle"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    source_registration: ArtifactReference
    oracle_result: ArtifactReference
    baseline_run_root: str = Field(min_length=1)
    query_ids: tuple[str, ...] = Field(min_length=1)
    selector_input: SelectorInput
    body_extraction: BodyExtraction
    candidate_policy: CandidatePolicy
    retrieval: SlateRetrieval
    acceptance: SlateAcceptance
    budgets: SlateBudgets
    sealed_holdout_access: Literal["forbidden"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def registration_is_consistent(self) -> "GoldBlindPivotRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("gold-blind pivot query IDs must be unique")
        if self.candidate_policy.slate_size_per_case != (
            self.retrieval.maximum_pivot_queries_per_case
        ):
            raise ValueError("gold-blind pivot slate and query caps must match")
        if self.budgets.maximum_offline_bm25_queries != (
            len(self.query_ids) * self.retrieval.maximum_pivot_queries_per_case
        ):
            raise ValueError("gold-blind pivot total query budget must match the case grid")
        for threshold in (
            self.acceptance.minimum_selector_gold_doc_recall_at20_cases,
            self.acceptance.minimum_retained_oracle_rescue_cases,
        ):
            if threshold > len(self.query_ids):
                raise ValueError("gold-blind pivot acceptance exceeds registered cases")
        return self


class PivotOccurrence(StrictContract):
    surface_form: str = Field(min_length=1)
    docid: str = Field(min_length=1)
    source_query: str = Field(min_length=1)
    source_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_index: int = Field(ge=1)
    result_index: int = Field(ge=1)
    body_character_offset: int = Field(ge=0)
    source_region: Literal["body"] = "body"


class SelectedPivot(StrictContract):
    analyzed_term: str = Field(min_length=1)
    surface_form: str = Field(min_length=1)
    document_frequency: int = Field(ge=2)
    distinct_visible_documents: int = Field(ge=1)
    source_query: str = Field(min_length=1)
    source_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_docid: str = Field(min_length=1)
    source_call_index: int = Field(ge=1)
    source_result_index: int = Field(ge=1)
    source_body_character_offset: int = Field(ge=0)
    source_region: Literal["body"] = "body"
    occurrences: tuple[PivotOccurrence, ...] = Field(min_length=1)


class PivotSlateCase(StrictContract):
    query_id: str = Field(min_length=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    successful_search_calls: int = Field(gt=0)
    saved_result_snippets: int = Field(gt=0)
    wrapper_stripped_snippets: int = Field(ge=0)
    frontmatter_stripped_snippets: int = Field(ge=0)
    empty_body_snippets: int = Field(ge=0)
    candidate_terms_before_slate: int = Field(ge=0)
    selected: tuple[SelectedPivot, ...]
    selection_failure: bool

    @model_validator(mode="after")
    def selection_is_consistent(self) -> "PivotSlateCase":
        if self.selection_failure != (not self.selected):
            raise ValueError("gold-blind pivot selection failure differs from slate")
        return self


class GoldBlindPivotSlate(StrictContract):
    schema_version: Literal["gold-blind-visible-pivot-slate-v0"] = (
        "gold-blind-visible-pivot-slate-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["selected_without_gold"] = "selected_without_gold"
    registration: ArtifactReference
    query_count: int = Field(gt=0)
    selection_failures: int = Field(ge=0)
    selected_pivot_count: int = Field(ge=0)
    contains_gold_documents: Literal[False] = False
    contains_gold_answers: Literal[False] = False
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[PivotSlateCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def totals_are_consistent(self) -> "GoldBlindPivotSlate":
        if self.query_count != len(self.items):
            raise ValueError("gold-blind pivot slate query count differs from items")
        if self.selection_failures != sum(item.selection_failure for item in self.items):
            raise ValueError("gold-blind pivot selection failures differ from items")
        if self.selected_pivot_count != sum(len(item.selected) for item in self.items):
            raise ValueError("gold-blind pivot count differs from items")
        return self


class SelectorAttempt(StrictContract):
    analyzed_term: str = Field(min_length=1)
    surface_form: str = Field(min_length=1)
    source_docid: str = Field(min_length=1)
    source_query: str = Field(min_length=1)
    source_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    composed_query: str = Field(min_length=1)
    composed_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top20_docids: tuple[str, ...] = Field(max_length=20)
    gold_hit: bool
    matched_gold_docid: str | None = None
    matched_gold_rank: int | None = Field(default=None, ge=1, le=20)
    latency_ms: float = Field(ge=0)


class SelectorScoreCase(StrictContract):
    query_id: str = Field(min_length=1)
    oracle_rescued: bool
    selected_pivot_count: int = Field(ge=0)
    attempts: tuple[SelectorAttempt, ...]
    selector_gold_hit_at20: bool
    retained_oracle_rescue: bool


class SelectorGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int = Field(ge=0)
    operator: Literal["eq", "ge", "le"]
    threshold: int = Field(ge=0)
    passed: bool


class GoldBlindPivotResult(StrictContract):
    schema_version: Literal["gold-blind-visible-pivot-slate-result-v0"] = (
        "gold-blind-visible-pivot-slate-result-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    decision: Literal["gold_blind_pivot_candidate", "freeze_gold_blind_pivot_branch"]
    registration: ArtifactReference
    slate: ArtifactReference
    query_count: int = Field(gt=0)
    oracle_rescue_cases: int = Field(ge=0)
    selector_gold_hit_cases_at20: int = Field(ge=0)
    retained_oracle_rescue_cases: int = Field(ge=0)
    selection_failures: int = Field(ge=0)
    frontmatter_only_selected_candidates: int = Field(ge=0)
    answer_string_leak_cases: int = Field(ge=0)
    gold_docid_leak_cases: int = Field(ge=0)
    offline_bm25_queries: int = Field(ge=0)
    search_latency_ms: float = Field(ge=0)
    execution_device: Literal["cpu"] = "cpu"
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[SelectorScoreCase, ...] = Field(min_length=1)
    gates: tuple[SelectorGate, SelectorGate, SelectorGate, SelectorGate]
    next_action: Literal[
        "preregister_fixed_budget_unseen_pivot_comparison",
        "freeze_gold_blind_pivot_and_reassess_entity_linking",
    ]
    claim_boundary: str = Field(min_length=1)


def load_gold_blind_pivot_registration(path: Path) -> GoldBlindPivotRegistration:
    registration = GoldBlindPivotRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = _repository_root(path)
    _validate_artifact(root, registration.source_registration)
    _validate_directory(root, registration.baseline_run_root)
    source = _read_object(root / registration.source_registration.path)
    if tuple(str(value) for value in source.get("query_ids", ())) != registration.query_ids:
        raise ValueError("gold-blind pivot cases differ from source registration")
    if source.get("baseline_run_root") != registration.baseline_run_root:
        raise ValueError("gold-blind pivot baseline root differs from source registration")
    index_relative = source.get("document_index_path")
    if not isinstance(index_relative, str):
        raise ValueError("gold-blind pivot source index path is invalid")
    index_path = _validate_directory(root, index_relative)
    raw_files = source.get("document_index_files")
    if not isinstance(raw_files, list):
        raise ValueError("gold-blind pivot source index files are invalid")
    expected_files = tuple(IndexFileDigest.model_validate(item) for item in raw_files)
    _validate_index_files(index_path, expected_files)
    return registration


def build_gold_blind_pivot_slate(
    *,
    registration_path: Path,
    output_path: Path,
    analyze: Analyze,
    document_frequency: DocumentFrequency,
) -> GoldBlindPivotSlate:
    registration = load_gold_blind_pivot_registration(registration_path)
    root = _repository_root(registration_path)
    output_path = output_path.resolve()
    _require_run_output(output_path, root)
    if output_path.exists():
        raise ValueError("gold-blind pivot slate already exists")

    items: list[PivotSlateCase] = []
    frequency_cache: dict[str, int] = {}
    for query_id in registration.query_ids:
        case_dir = root / registration.baseline_run_root / query_id
        request_path = case_dir / "request.json"
        run_path = case_dir / "run.json"
        request = _read_object(request_path)
        run = _read_object(run_path)
        if str(request.get("query_id")) != query_id or str(run.get("query_id")) != query_id:
            raise ValueError(f"gold-blind pivot source ID changed: {query_id}")
        question = request.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"gold-blind pivot question is blank: {query_id}")
        calls = run.get("search_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"gold-blind pivot run has no search calls: {query_id}")
        if any(not isinstance(call, dict) or call.get("outcome") != "ok" for call in calls):
            raise ValueError(f"gold-blind pivot run has failed search: {query_id}")

        excluded = set(_normalize_terms(analyze(question)))
        for call in calls:
            query = call.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"gold-blind pivot source query is blank: {query_id}")
            excluded.update(_normalize_terms(analyze(query)))

        support: dict[str, list[PivotOccurrence]] = defaultdict(list)
        wrapper_stripped = 0
        frontmatter_stripped = 0
        empty_body = 0
        snippet_count = 0
        for call_index, call in enumerate(calls, start=1):
            query = str(call["query"])
            query_hash = _text_sha256(query)
            results = call.get("results")
            if not isinstance(results, list) or not results:
                raise ValueError(f"gold-blind pivot search has no results: {query_id}")
            for result_index, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    raise ValueError(f"gold-blind pivot saved result is invalid: {query_id}")
                docid = result.get("docid")
                snippet = result.get("snippet")
                if not isinstance(docid, str) or not docid:
                    raise ValueError(f"gold-blind pivot saved docid is invalid: {query_id}")
                if not isinstance(snippet, str) or not snippet.strip():
                    raise ValueError(f"gold-blind pivot saved snippet is blank: {query_id}")
                snippet_count += 1
                body, removed_wrapper, removed_frontmatter = extract_visible_body(snippet)
                wrapper_stripped += removed_wrapper
                frontmatter_stripped += removed_frontmatter
                if not body.strip():
                    empty_body += 1
                    continue
                for match in _CAPITALIZED_TOKEN.finditer(body):
                    surface = match.group(0)
                    if not (
                        registration.candidate_policy.minimum_surface_characters
                        <= len(surface)
                        <= registration.candidate_policy.maximum_surface_characters
                    ):
                        continue
                    analyzed = _normalize_terms(analyze(surface))
                    if len(analyzed) != 1:
                        continue
                    term = analyzed[0]
                    if term in excluded:
                        continue
                    if term not in frequency_cache:
                        frequency_cache[term] = int(document_frequency(term))
                    frequency = frequency_cache[term]
                    if frequency < 2 or frequency > (
                        registration.candidate_policy.maximum_document_frequency
                    ):
                        continue
                    support[term].append(
                        PivotOccurrence(
                            surface_form=surface,
                            docid=docid,
                            source_query=query,
                            source_query_sha256=query_hash,
                            call_index=call_index,
                            result_index=result_index,
                            body_character_offset=match.start(),
                        )
                    )

        candidates: list[SelectedPivot] = []
        for term, occurrences in support.items():
            ordered = sorted(
                occurrences,
                key=lambda item: (
                    item.call_index,
                    item.result_index,
                    item.body_character_offset,
                    item.surface_form,
                ),
            )
            first = ordered[0]
            candidates.append(
                SelectedPivot(
                    analyzed_term=term,
                    surface_form=first.surface_form,
                    document_frequency=frequency_cache[term],
                    distinct_visible_documents=len({item.docid for item in ordered}),
                    source_query=first.source_query,
                    source_query_sha256=first.source_query_sha256,
                    source_docid=first.docid,
                    source_call_index=first.call_index,
                    source_result_index=first.result_index,
                    source_body_character_offset=first.body_character_offset,
                    occurrences=tuple(ordered),
                )
            )
        candidates.sort(
            key=lambda item: (
                item.document_frequency,
                -item.distinct_visible_documents,
                item.source_call_index,
                item.source_result_index,
                item.source_body_character_offset,
                item.analyzed_term,
            )
        )
        selected = tuple(candidates[: registration.candidate_policy.slate_size_per_case])
        items.append(
            PivotSlateCase(
                query_id=query_id,
                request_sha256=sha256_file(request_path),
                run_sha256=sha256_file(run_path),
                question_sha256=_text_sha256(question),
                successful_search_calls=len(calls),
                saved_result_snippets=snippet_count,
                wrapper_stripped_snippets=wrapper_stripped,
                frontmatter_stripped_snippets=frontmatter_stripped,
                empty_body_snippets=empty_body,
                candidate_terms_before_slate=len(candidates),
                selected=selected,
                selection_failure=not selected,
            )
        )
    slate = GoldBlindPivotSlate(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(registration_path),
        ),
        query_count=len(items),
        selection_failures=sum(item.selection_failure for item in items),
        selected_pivot_count=sum(len(item.selected) for item in items),
        items=tuple(items),
    )
    _atomic_write(output_path, slate.model_dump(mode="json"))
    return slate


def score_gold_blind_pivot_slate(
    *,
    registration_path: Path,
    slate_path: Path,
    output_path: Path,
    search: DocumentSearch,
    analyze: Analyze,
) -> GoldBlindPivotResult:
    registration = load_gold_blind_pivot_registration(registration_path)
    root = _repository_root(registration_path)
    _validate_artifact(root, registration.oracle_result)
    output_path = output_path.resolve()
    _require_run_output(output_path, root)
    if output_path.exists():
        raise ValueError("gold-blind pivot score already exists")
    slate_path = slate_path.resolve()
    _require_run_output(slate_path, root)
    slate = GoldBlindPivotSlate.model_validate_json(
        slate_path.read_text(encoding="utf-8")
    )
    if slate.registration.sha256 != sha256_file(registration_path):
        raise ValueError("gold-blind pivot slate targets another registration")
    if tuple(item.query_id for item in slate.items) != registration.query_ids:
        raise ValueError("gold-blind pivot slate cases differ from registration")

    source = _read_object(root / registration.source_registration.path)
    gold_reference = source.get("gold_slice")
    if not isinstance(gold_reference, dict):
        raise ValueError("gold-blind pivot source gold reference is invalid")
    gold_artifact = ArtifactReference.model_validate(gold_reference)
    _validate_artifact(root, gold_artifact)
    gold = _read_object(root / gold_artifact.path)
    gold_by_id = _gold_rows(gold, registration.query_ids)
    oracle = _read_object(root / registration.oracle_result.path)
    oracle_by_id = {
        str(item["query_id"]): bool(item["rescued"])
        for item in _items(oracle, label="oracle")
    }
    if tuple(oracle_by_id) != registration.query_ids:
        raise ValueError("gold-blind pivot oracle cases differ from registration")

    items: list[SelectorScoreCase] = []
    offline_queries = 0
    latency = 0.0
    answer_leaks = 0
    docid_leaks = 0
    for slate_item in slate.items:
        gold_row = gold_by_id[slate_item.query_id]
        answer = str(gold_row["answer"])
        answer_normalized = " ".join(_normalize_terms(analyze(answer)))
        gold_docids = {str(value) for value in gold_row["gold_docids"]}
        case_answer_leak = False
        case_docid_leak = False
        attempts: list[SelectorAttempt] = []
        for selected in slate_item.selected:
            if selected.source_region != "body":
                raise ValueError("gold-blind pivot selected a non-body candidate")
            candidate_normalized = " ".join(_normalize_terms(analyze(selected.surface_form)))
            if answer_normalized and candidate_normalized == answer_normalized:
                case_answer_leak = True
            if selected.source_docid in gold_docids:
                case_docid_leak = True
            composed = f"{selected.source_query} {selected.surface_form}"
            started = perf_counter()
            hits = tuple(str(docid) for docid in search(composed, registration.retrieval.top_k))
            elapsed = (perf_counter() - started) * 1000
            latency += elapsed
            offline_queries += 1
            if len(hits) > registration.retrieval.top_k:
                raise ValueError("gold-blind pivot search exceeded registered top-k")
            if len(hits) != len(set(hits)):
                raise ValueError("gold-blind pivot search returned duplicate documents")
            matched = next((docid for docid in hits if docid in gold_docids), None)
            attempts.append(
                SelectorAttempt(
                    analyzed_term=selected.analyzed_term,
                    surface_form=selected.surface_form,
                    source_docid=selected.source_docid,
                    source_query=selected.source_query,
                    source_query_sha256=selected.source_query_sha256,
                    composed_query=composed,
                    composed_query_sha256=_text_sha256(composed),
                    top20_docids=hits,
                    gold_hit=matched is not None,
                    matched_gold_docid=matched,
                    matched_gold_rank=(hits.index(matched) + 1 if matched else None),
                    latency_ms=elapsed,
                )
            )
        answer_leaks += case_answer_leak
        docid_leaks += case_docid_leak
        hit = any(attempt.gold_hit for attempt in attempts)
        oracle_rescued = oracle_by_id[slate_item.query_id]
        items.append(
            SelectorScoreCase(
                query_id=slate_item.query_id,
                oracle_rescued=oracle_rescued,
                selected_pivot_count=len(slate_item.selected),
                attempts=tuple(attempts),
                selector_gold_hit_at20=hit,
                retained_oracle_rescue=hit and oracle_rescued,
            )
        )
    if answer_leaks or docid_leaks:
        raise ValueError(
            "gold-blind pivot slate leaked an exact answer or gold document into selection"
        )
    if offline_queries > registration.budgets.maximum_offline_bm25_queries:
        raise ValueError("gold-blind pivot offline query budget exceeded")

    selector_hits = sum(item.selector_gold_hit_at20 for item in items)
    retained = sum(item.retained_oracle_rescue for item in items)
    selection_failures = slate.selection_failures
    frontmatter_selected = sum(
        selected.source_region != "body"
        for item in slate.items
        for selected in item.selected
    )
    pass_selector = (
        selector_hits
        >= registration.acceptance.minimum_selector_gold_doc_recall_at20_cases
    )
    pass_retention = (
        retained >= registration.acceptance.minimum_retained_oracle_rescue_cases
    )
    pass_failures = (
        selection_failures == registration.acceptance.required_selection_failures
    )
    pass_frontmatter = (
        frontmatter_selected
        == registration.acceptance.required_frontmatter_only_selected_candidates
    )
    accepted = pass_selector and pass_retention and pass_failures and pass_frontmatter
    gates = (
        SelectorGate(
            gate_id="selector_gold_doc_recall_at20_cases",
            observed=selector_hits,
            operator="ge",
            threshold=registration.acceptance.minimum_selector_gold_doc_recall_at20_cases,
            passed=pass_selector,
        ),
        SelectorGate(
            gate_id="retained_oracle_rescue_cases",
            observed=retained,
            operator="ge",
            threshold=registration.acceptance.minimum_retained_oracle_rescue_cases,
            passed=pass_retention,
        ),
        SelectorGate(
            gate_id="selection_failures",
            observed=selection_failures,
            operator="eq",
            threshold=registration.acceptance.required_selection_failures,
            passed=pass_failures,
        ),
        SelectorGate(
            gate_id="frontmatter_only_selected_candidates",
            observed=frontmatter_selected,
            operator="eq",
            threshold=(
                registration.acceptance.required_frontmatter_only_selected_candidates
            ),
            passed=pass_frontmatter,
        ),
    )
    result = GoldBlindPivotResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=(
            "gold_blind_pivot_candidate"
            if accepted
            else "freeze_gold_blind_pivot_branch"
        ),
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(registration_path),
        ),
        slate=ArtifactReference(
            path=slate_path.relative_to(root).as_posix(),
            sha256=sha256_file(slate_path),
        ),
        query_count=len(items),
        oracle_rescue_cases=sum(oracle_by_id.values()),
        selector_gold_hit_cases_at20=selector_hits,
        retained_oracle_rescue_cases=retained,
        selection_failures=selection_failures,
        frontmatter_only_selected_candidates=frontmatter_selected,
        answer_string_leak_cases=answer_leaks,
        gold_docid_leak_cases=docid_leaks,
        offline_bm25_queries=offline_queries,
        search_latency_ms=latency,
        items=tuple(items),
        gates=gates,
        next_action=(
            "preregister_fixed_budget_unseen_pivot_comparison"
            if accepted
            else "freeze_gold_blind_pivot_and_reassess_entity_linking"
        ),
        claim_boundary=registration.claim_boundary,
    )
    _atomic_write(output_path, result.model_dump(mode="json"))
    return result


def extract_visible_body(text: str) -> tuple[str, bool, bool]:
    without_wrapper, wrapper_count = _LEADING_WRAPPER.subn("", text, count=1)
    body, frontmatter_count = _FRONTMATTER.subn("", without_wrapper, count=1)
    return body, bool(wrapper_count), bool(frontmatter_count)


def _normalize_terms(terms: Sequence[str]) -> tuple[str, ...]:
    return tuple(term.strip().casefold() for term in terms if term.strip())


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate gold-blind pivot repository root")


def _require_run_output(path: Path, root: Path) -> None:
    if not path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("gold-blind pivot output must stay under ignored runs/")


def _validate_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"gold-blind pivot directory is missing or escapes root: {relative}")
    return path


def _validate_artifact(root: Path, artifact: ArtifactReference) -> None:
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"gold-blind pivot artifact is missing: {artifact.path}")
    if sha256_file(path) != artifact.sha256:
        raise ValueError(f"gold-blind pivot artifact hash changed: {artifact.path}")


def _validate_index_files(index_path: Path, expected: Sequence[IndexFileDigest]) -> None:
    actual_names = {
        path.name for path in index_path.iterdir() if path.is_file() and path.name != "write.lock"
    }
    expected_names = {item.name for item in expected}
    if actual_names != expected_names:
        raise ValueError("gold-blind pivot index file set changed")
    for item in expected:
        path = index_path / item.name
        if path.stat().st_size != item.bytes or sha256_file(path) != item.sha256:
            raise ValueError(f"gold-blind pivot index file changed: {item.name}")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"gold-blind pivot artifact is not an object: {path}")
    return value


def _items(value: dict[str, object], *, label: str) -> list[dict[str, object]]:
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"gold-blind pivot {label} items are invalid")
    return items


def _gold_rows(
    value: dict[str, object], query_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("gold-blind pivot gold rows are invalid")
    by_id = {str(row.get("query_id")): row for row in rows}
    for query_id in query_ids:
        row = by_id.get(query_id)
        if row is None or not isinstance(row.get("gold_docids"), list):
            raise ValueError(f"gold-blind pivot gold row is missing: {query_id}")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"gold-blind pivot gold answer is invalid: {query_id}")
    return by_id


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
