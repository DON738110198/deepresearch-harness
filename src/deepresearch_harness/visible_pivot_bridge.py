from __future__ import annotations

import json
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
DocumentLoad = Callable[[str], str]
DocumentFrequency = Callable[[str], int]
DocumentSearch = Callable[[str, int], Sequence[str]]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PivotPolicy(StrictContract):
    analyzer_id: Literal["pyserini_default_english_v1.2.0"]
    minimum_term_characters: int = Field(ge=1, le=32)
    maximum_document_frequency: int = Field(ge=2)
    maximum_candidate_terms_per_case: int = Field(ge=1, le=256)
    candidate_order: Literal["document_frequency_ascending_then_term"]
    require_term_in_visible_non_gold_snippet: Literal[True]
    require_term_in_gold_document: Literal[True]
    excluded_vocabularies: tuple[
        Literal["raw_question", "recorded_generated_queries", "gold_answer"], ...
    ]

    @model_validator(mode="after")
    def exclusions_are_exact(self) -> "PivotPolicy":
        if self.excluded_vocabularies != (
            "raw_question",
            "recorded_generated_queries",
            "gold_answer",
        ):
            raise ValueError("visible-pivot exclusions must remain question, queries, answer")
        return self


class PivotRetrieval(StrictContract):
    base_query_source: Literal["trial1_recorded_successful_search_calls"]
    composition: Literal["append_single_analyzed_pivot_token_v0"]
    analyzer_id: Literal["pyserini_default_english_v1.2.0"]
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
    top_k: Literal[20]
    stop_after_first_case_rescue: Literal[True]


class PivotAcceptance(StrictContract):
    minimum_visible_pivot_gold_doc_recall_at20_cases: int = Field(ge=1)
    required_baseline_gold_doc_recall_at20_cases: Literal[0]


class PivotBudgets(StrictContract):
    maximum_offline_bm25_queries: int = Field(gt=0)
    maximum_provider_calls: Literal[0]
    maximum_online_search_calls: Literal[0]
    maximum_judge_calls: Literal[0]


class VisiblePivotRegistration(StrictContract):
    schema_version: Literal["visible-pivot-bridge-registration-v0"]
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    dense_rank_result: ArtifactReference
    passage_index_result: ArtifactReference
    gold_slice: ArtifactReference
    baseline_run_root: str = Field(min_length=1)
    query_ids: tuple[str, ...] = Field(min_length=1)
    document_index_path: str = Field(min_length=1)
    document_count: int = Field(gt=0)
    document_index_files: tuple[IndexFileDigest, ...] = Field(min_length=1)
    visible_evidence_source: Literal["trial1_saved_successful_search_result_snippets"]
    pivot_policy: PivotPolicy
    retrieval: PivotRetrieval
    acceptance: PivotAcceptance
    budgets: PivotBudgets
    sealed_holdout_access: Literal["forbidden"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def registration_is_consistent(self) -> "VisiblePivotRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("visible-pivot query IDs must be unique")
        if (
            self.acceptance.minimum_visible_pivot_gold_doc_recall_at20_cases
            > len(self.query_ids)
        ):
            raise ValueError("visible-pivot acceptance exceeds registered cases")
        names = [item.name for item in self.document_index_files]
        if len(names) != len(set(names)):
            raise ValueError("visible-pivot index filenames must be unique")
        return self


class FrozenPivotQuery(StrictContract):
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recorded_full_document_top20: tuple[str, ...] = Field(max_length=20)


class VisibleSnippet(StrictContract):
    docid: str = Field(min_length=1)
    text: str = Field(min_length=1)
    call_index: int = Field(ge=1)
    result_index: int = Field(ge=1)


class FrozenPivotCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    queries: tuple[FrozenPivotQuery, ...] = Field(min_length=1)
    visible_snippets: tuple[VisibleSnippet, ...] = Field(min_length=1)


class PivotCandidate(StrictContract):
    term: str = Field(min_length=1)
    document_frequency: int = Field(ge=2)
    visible_docids: tuple[str, ...] = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)


class PivotAttempt(StrictContract):
    pivot_term: str = Field(min_length=1)
    base_query: str = Field(min_length=1)
    base_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    composed_query: str = Field(min_length=1)
    composed_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top20_docids: tuple[str, ...] = Field(max_length=20)
    gold_hit: bool
    matched_gold_docid: str | None = None
    matched_gold_rank: int | None = Field(default=None, ge=1, le=20)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def hit_is_consistent(self) -> "PivotAttempt":
        if self.gold_hit != (self.matched_gold_docid is not None):
            raise ValueError("visible-pivot hit and matched document differ")
        if self.gold_hit != (self.matched_gold_rank is not None):
            raise ValueError("visible-pivot hit and matched rank differ")
        return self


class VisiblePivotCase(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    generated_query_count: int = Field(gt=0)
    visible_snippet_count: int = Field(gt=0)
    visible_document_count: int = Field(gt=0)
    visible_term_count: int = Field(ge=0)
    gold_term_count: int = Field(ge=0)
    excluded_term_count: int = Field(ge=0)
    candidate_term_count_before_cap: int = Field(ge=0)
    candidates: tuple[PivotCandidate, ...]
    attempts: tuple[PivotAttempt, ...]
    rescued: bool
    selected_pivot_term: str | None = None
    selected_base_query_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    matched_gold_docid: str | None = None
    matched_gold_rank: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def rescue_is_consistent(self) -> "VisiblePivotCase":
        selected = (
            self.selected_pivot_term,
            self.selected_base_query_sha256,
            self.matched_gold_docid,
            self.matched_gold_rank,
        )
        if self.rescued != all(value is not None for value in selected):
            raise ValueError("visible-pivot rescue fields are incomplete")
        if self.rescued and (not self.attempts or not self.attempts[-1].gold_hit):
            raise ValueError("visible-pivot rescued case must end with the successful attempt")
        if not self.rescued and any(attempt.gold_hit for attempt in self.attempts):
            raise ValueError("visible-pivot failed case contains a successful attempt")
        return self


class PivotGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int = Field(ge=0)
    operator: Literal["eq", "ge", "le"]
    threshold: int = Field(ge=0)
    passed: bool


class VisiblePivotResult(StrictContract):
    schema_version: Literal["visible-pivot-bridge-result-v0"] = (
        "visible-pivot-bridge-result-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    decision: Literal["visible_pivot_sufficient", "freeze_visible_pivot_branch"]
    registration: ArtifactReference
    query_count: int = Field(gt=0)
    generated_query_count: int = Field(gt=0)
    baseline_bm25_queries: int = Field(gt=0)
    pivot_bm25_queries: int = Field(ge=0)
    offline_bm25_queries: int = Field(gt=0)
    baseline_gold_hit_cases_at20: int = Field(ge=0)
    cases_with_candidate_pivots: int = Field(ge=0)
    visible_pivot_gold_hit_cases_at20: int = Field(ge=0)
    total_candidate_terms_before_cap: int = Field(ge=0)
    total_candidate_terms_after_cap: int = Field(ge=0)
    baseline_search_latency_ms: float = Field(ge=0)
    pivot_search_latency_ms: float = Field(ge=0)
    execution_device: Literal["cpu"] = "cpu"
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[VisiblePivotCase, ...] = Field(min_length=1)
    gates: tuple[PivotGate, PivotGate, PivotGate]
    next_action: Literal[
        "preregister_gold_blind_visible_pivot_selector",
        "freeze_visible_pivot_and_diagnose_external_entity_linking",
    ]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def totals_are_consistent(self) -> "VisiblePivotResult":
        if self.query_count != len(self.items):
            raise ValueError("visible-pivot query count differs from items")
        if self.generated_query_count != sum(item.generated_query_count for item in self.items):
            raise ValueError("visible-pivot generated query count differs from items")
        if self.offline_bm25_queries != self.baseline_bm25_queries + self.pivot_bm25_queries:
            raise ValueError("visible-pivot offline query accounting does not add up")
        if self.visible_pivot_gold_hit_cases_at20 != sum(item.rescued for item in self.items):
            raise ValueError("visible-pivot rescued case total differs from items")
        return self


def load_visible_pivot_registration(path: Path) -> VisiblePivotRegistration:
    registration = VisiblePivotRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = _repository_root(path)
    for artifact in (
        registration.dense_rank_result,
        registration.passage_index_result,
        registration.gold_slice,
    ):
        _validate_artifact(root, artifact)
    _validate_directory(root, registration.baseline_run_root)
    index_path = _validate_directory(root, registration.document_index_path)
    _validate_index_files(index_path, registration.document_index_files)

    dense = _read_object(root / registration.dense_rank_result.path)
    if dense.get("decision") != "freeze_dense_channel":
        raise ValueError("visible-pivot dense prerequisite selected another branch")
    if dense.get("dense_gold_hit_cases_at20") != 0:
        raise ValueError("visible-pivot dense top-20 prerequisite changed")
    if dense.get("dense_gold_hit_cases_at100") != 1:
        raise ValueError("visible-pivot dense top-100 prerequisite changed")
    if _item_ids(dense) != registration.query_ids:
        raise ValueError("visible-pivot cases differ from dense-rank result")

    passage = _read_object(root / registration.passage_index_result.path)
    if passage.get("decision") != "freeze_passage_index_branch":
        raise ValueError("visible-pivot passage prerequisite selected another branch")
    if passage.get("full_document_gold_hit_cases_at20") != 0:
        raise ValueError("visible-pivot BM25 baseline prerequisite changed")
    if _item_ids(passage) != registration.query_ids:
        raise ValueError("visible-pivot cases differ from passage-index result")
    _validate_gold_rows(
        _read_object(root / registration.gold_slice.path), registration.query_ids
    )
    return registration


def collect_frozen_pivot_cases(path: Path) -> tuple[FrozenPivotCase, ...]:
    registration = load_visible_pivot_registration(path)
    root = _repository_root(path)
    passage = _read_object(root / registration.passage_index_result.path)
    gold = _read_object(root / registration.gold_slice.path)
    passage_by_id = {
        str(item["query_id"]): item for item in _items(passage, label="passage")
    }
    gold_by_id = _validate_gold_rows(gold, registration.query_ids)
    frozen: list[FrozenPivotCase] = []
    for query_id in registration.query_ids:
        gold_row = gold_by_id[query_id]
        gold_docids = tuple(str(value) for value in gold_row["gold_docids"])
        passage_case = passage_by_id[query_id]
        if tuple(str(value) for value in passage_case["gold_docids"]) != gold_docids:
            raise ValueError(f"visible-pivot gold documents changed for query {query_id}")
        recorded_queries = passage_case.get("queries")
        if not isinstance(recorded_queries, list):
            raise ValueError(f"visible-pivot passage queries are invalid: {query_id}")

        run = _read_object(
            root / registration.baseline_run_root / query_id / "run.json"
        )
        if str(run.get("query_id")) != query_id:
            raise ValueError(f"visible-pivot baseline run ID changed: {query_id}")
        calls = run.get("search_calls")
        if not isinstance(calls, list) or len(calls) != len(recorded_queries):
            raise ValueError(f"visible-pivot search calls changed: {query_id}")

        queries: list[FrozenPivotQuery] = []
        snippets: list[VisibleSnippet] = []
        for call_index, (call, recorded) in enumerate(
            zip(calls, recorded_queries, strict=True), start=1
        ):
            if not isinstance(call, dict) or call.get("outcome") != "ok":
                raise ValueError(f"visible-pivot baseline search failed: {query_id}")
            if not isinstance(recorded, dict):
                raise ValueError(f"visible-pivot recorded query is invalid: {query_id}")
            query = call.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"visible-pivot baseline query is blank: {query_id}")
            query_hash = _text_sha256(query)
            if recorded.get("query") != query or recorded.get("query_sha256") != query_hash:
                raise ValueError(f"visible-pivot recorded query changed: {query_id}")
            recorded_top20 = recorded.get("full_document_top20")
            if not isinstance(recorded_top20, list):
                raise ValueError(f"visible-pivot recorded BM25 ranking is invalid: {query_id}")
            queries.append(
                FrozenPivotQuery(
                    query=query,
                    query_sha256=query_hash,
                    recorded_full_document_top20=tuple(str(value) for value in recorded_top20),
                )
            )
            results = call.get("results")
            if not isinstance(results, list) or not results:
                raise ValueError(f"visible-pivot saved search has no results: {query_id}")
            for result_index, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    raise ValueError(f"visible-pivot saved result is invalid: {query_id}")
                docid = result.get("docid")
                snippet = result.get("snippet")
                if not isinstance(docid, str) or not docid:
                    raise ValueError(f"visible-pivot saved docid is invalid: {query_id}")
                if not isinstance(snippet, str) or not snippet.strip():
                    raise ValueError(f"visible-pivot saved snippet is blank: {query_id}")
                if docid in set(gold_docids):
                    raise ValueError(f"visible-pivot case is no longer a retrieval miss: {query_id}")
                snippets.append(
                    VisibleSnippet(
                        docid=docid,
                        text=snippet,
                        call_index=call_index,
                        result_index=result_index,
                    )
                )
        question = gold_row.get("question")
        answer = gold_row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"visible-pivot question is blank: {query_id}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"visible-pivot answer is blank: {query_id}")
        frozen.append(
            FrozenPivotCase(
                query_id=query_id,
                question=question,
                answer=answer,
                gold_docids=gold_docids,
                queries=tuple(queries),
                visible_snippets=tuple(snippets),
            )
        )
    return tuple(frozen)


def run_visible_pivot_oracle(
    *,
    registration_path: Path,
    output_path: Path,
    analyze: Analyze,
    load_document: DocumentLoad,
    document_frequency: DocumentFrequency,
    search: DocumentSearch,
) -> VisiblePivotResult:
    registration = load_visible_pivot_registration(registration_path)
    root = _repository_root(registration_path)
    output_path = output_path.resolve()
    if not output_path.is_relative_to((root / "runs").resolve()):
        raise ValueError("visible-pivot output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("visible-pivot result already exists")
    frozen = collect_frozen_pivot_cases(registration_path)
    maximum_possible_queries = sum(
        len(item.queries)
        * registration.pivot_policy.maximum_candidate_terms_per_case
        for item in frozen
    ) + sum(len(item.queries) for item in frozen)
    if maximum_possible_queries > registration.budgets.maximum_offline_bm25_queries:
        raise ValueError("visible-pivot registered offline query budget is too small")

    baseline_queries = 0
    baseline_latency = 0.0
    baseline_hit_cases = 0
    for item in frozen:
        case_hit = False
        gold = set(item.gold_docids)
        for query in item.queries:
            started = perf_counter()
            hits = tuple(str(docid) for docid in search(query.query, registration.retrieval.top_k))
            baseline_latency += (perf_counter() - started) * 1000
            baseline_queries += 1
            if hits != query.recorded_full_document_top20:
                raise ValueError(
                    f"visible-pivot BM25 baseline ranking changed for query {item.query_id}"
                )
            case_hit = case_hit or bool(gold.intersection(hits))
        baseline_hit_cases += case_hit
    if baseline_hit_cases != (
        registration.acceptance.required_baseline_gold_doc_recall_at20_cases
    ):
        raise ValueError("visible-pivot BM25 baseline case count changed")

    items: list[VisiblePivotCase] = []
    pivot_queries = 0
    pivot_latency = 0.0
    for item in frozen:
        visible_support: dict[str, set[str]] = defaultdict(set)
        for snippet in item.visible_snippets:
            for term in set(_normalize_terms(analyze(snippet.text))):
                visible_support[term].add(snippet.docid)
        gold_support: dict[str, set[str]] = defaultdict(set)
        for docid in item.gold_docids:
            contents = load_document(docid)
            if not contents.strip():
                raise ValueError(f"visible-pivot gold document is blank: {docid}")
            for term in set(_normalize_terms(analyze(contents))):
                gold_support[term].add(docid)
        excluded = set(_normalize_terms(analyze(item.question)))
        excluded.update(_normalize_terms(analyze(item.answer)))
        for query in item.queries:
            excluded.update(_normalize_terms(analyze(query.query)))

        eligible: list[PivotCandidate] = []
        for term in set(visible_support).intersection(gold_support) - excluded:
            if len(term) < registration.pivot_policy.minimum_term_characters:
                continue
            frequency = int(document_frequency(term))
            if frequency < 2 or frequency > registration.pivot_policy.maximum_document_frequency:
                continue
            eligible.append(
                PivotCandidate(
                    term=term,
                    document_frequency=frequency,
                    visible_docids=tuple(sorted(visible_support[term])),
                    gold_docids=tuple(sorted(gold_support[term])),
                )
            )
        eligible.sort(key=lambda candidate: (candidate.document_frequency, candidate.term))
        candidate_count = len(eligible)
        candidates = tuple(
            eligible[: registration.pivot_policy.maximum_candidate_terms_per_case]
        )

        attempts: list[PivotAttempt] = []
        selected: PivotAttempt | None = None
        gold = set(item.gold_docids)
        for candidate in candidates:
            for query in item.queries:
                composed = f"{query.query} {candidate.term}"
                started = perf_counter()
                hits = tuple(
                    str(docid) for docid in search(composed, registration.retrieval.top_k)
                )
                elapsed = (perf_counter() - started) * 1000
                pivot_latency += elapsed
                pivot_queries += 1
                if len(hits) > registration.retrieval.top_k:
                    raise ValueError("visible-pivot search exceeded registered top-k")
                if len(hits) != len(set(hits)):
                    raise ValueError("visible-pivot search returned duplicate documents")
                matched_docid = next((docid for docid in hits if docid in gold), None)
                matched_rank = hits.index(matched_docid) + 1 if matched_docid else None
                attempt = PivotAttempt(
                    pivot_term=candidate.term,
                    base_query=query.query,
                    base_query_sha256=query.query_sha256,
                    composed_query=composed,
                    composed_query_sha256=_text_sha256(composed),
                    top20_docids=hits,
                    gold_hit=matched_docid is not None,
                    matched_gold_docid=matched_docid,
                    matched_gold_rank=matched_rank,
                    latency_ms=elapsed,
                )
                attempts.append(attempt)
                if attempt.gold_hit:
                    selected = attempt
                    break
            if selected is not None:
                break
        items.append(
            VisiblePivotCase(
                query_id=item.query_id,
                gold_docids=item.gold_docids,
                generated_query_count=len(item.queries),
                visible_snippet_count=len(item.visible_snippets),
                visible_document_count=len({snippet.docid for snippet in item.visible_snippets}),
                visible_term_count=len(visible_support),
                gold_term_count=len(gold_support),
                excluded_term_count=len(excluded),
                candidate_term_count_before_cap=candidate_count,
                candidates=candidates,
                attempts=tuple(attempts),
                rescued=selected is not None,
                selected_pivot_term=(selected.pivot_term if selected else None),
                selected_base_query_sha256=(
                    selected.base_query_sha256 if selected else None
                ),
                matched_gold_docid=(selected.matched_gold_docid if selected else None),
                matched_gold_rank=(selected.matched_gold_rank if selected else None),
            )
        )
    offline_queries = baseline_queries + pivot_queries
    if offline_queries > registration.budgets.maximum_offline_bm25_queries:
        raise ValueError("visible-pivot offline query budget exceeded")
    rescued = sum(item.rescued for item in items)
    accepted = (
        rescued
        >= registration.acceptance.minimum_visible_pivot_gold_doc_recall_at20_cases
    )
    gates = (
        PivotGate(
            gate_id="baseline_gold_doc_recall_at20_cases",
            observed=baseline_hit_cases,
            operator="eq",
            threshold=(
                registration.acceptance.required_baseline_gold_doc_recall_at20_cases
            ),
            passed=(
                baseline_hit_cases
                == registration.acceptance.required_baseline_gold_doc_recall_at20_cases
            ),
        ),
        PivotGate(
            gate_id="visible_pivot_gold_doc_recall_at20_cases",
            observed=rescued,
            operator="ge",
            threshold=(
                registration.acceptance.minimum_visible_pivot_gold_doc_recall_at20_cases
            ),
            passed=accepted,
        ),
        PivotGate(
            gate_id="offline_bm25_query_budget",
            observed=offline_queries,
            operator="le",
            threshold=registration.budgets.maximum_offline_bm25_queries,
            passed=offline_queries <= registration.budgets.maximum_offline_bm25_queries,
        ),
    )
    result = VisiblePivotResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=(
            "visible_pivot_sufficient" if accepted else "freeze_visible_pivot_branch"
        ),
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(registration_path),
        ),
        query_count=len(items),
        generated_query_count=sum(len(item.queries) for item in frozen),
        baseline_bm25_queries=baseline_queries,
        pivot_bm25_queries=pivot_queries,
        offline_bm25_queries=offline_queries,
        baseline_gold_hit_cases_at20=baseline_hit_cases,
        cases_with_candidate_pivots=sum(bool(item.candidates) for item in items),
        visible_pivot_gold_hit_cases_at20=rescued,
        total_candidate_terms_before_cap=sum(
            item.candidate_term_count_before_cap for item in items
        ),
        total_candidate_terms_after_cap=sum(len(item.candidates) for item in items),
        baseline_search_latency_ms=baseline_latency,
        pivot_search_latency_ms=pivot_latency,
        items=tuple(items),
        gates=gates,
        next_action=(
            "preregister_gold_blind_visible_pivot_selector"
            if accepted
            else "freeze_visible_pivot_and_diagnose_external_entity_linking"
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


def _normalize_terms(terms: Sequence[str]) -> tuple[str, ...]:
    return tuple(term.strip().casefold() for term in terms if term.strip())


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate visible-pivot repository root")


def _validate_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"visible-pivot directory is missing or escapes root: {relative}")
    return path


def _validate_artifact(root: Path, artifact: ArtifactReference) -> None:
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"visible-pivot artifact is missing or escapes root: {artifact.path}")
    if sha256_file(path) != artifact.sha256:
        raise ValueError(f"visible-pivot artifact hash changed: {artifact.path}")


def _validate_index_files(index_path: Path, expected: Sequence[IndexFileDigest]) -> None:
    actual_names = {
        path.name for path in index_path.iterdir() if path.is_file() and path.name != "write.lock"
    }
    expected_names = {item.name for item in expected}
    if actual_names != expected_names:
        raise ValueError("visible-pivot index file set changed")
    for item in expected:
        path = index_path / item.name
        if path.stat().st_size != item.bytes or sha256_file(path) != item.sha256:
            raise ValueError(f"visible-pivot index file changed: {item.name}")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"visible-pivot artifact is not an object: {path}")
    return value


def _items(value: dict[str, object], *, label: str) -> list[dict[str, object]]:
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"visible-pivot {label} items are invalid")
    return items


def _item_ids(value: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(item["query_id"]) for item in _items(value, label="prerequisite"))


def _validate_gold_rows(
    value: dict[str, object], query_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("visible-pivot gold rows are invalid")
    by_id = {str(row.get("query_id")): row for row in rows}
    if any(query_id not in by_id for query_id in query_ids):
        raise ValueError("visible-pivot gold rows are missing registered cases")
    for query_id in query_ids:
        row = by_id[query_id]
        docids = row.get("gold_docids")
        if not isinstance(docids, list) or not docids:
            raise ValueError(f"visible-pivot gold documents are invalid: {query_id}")
    return by_id


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
