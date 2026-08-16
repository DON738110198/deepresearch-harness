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
from .passage_index_gate import IndexFileDigest, SourceDocumentIndex, sha256_file


DocumentSearch = Callable[[str, int], Sequence[str]]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RrfPrerequisites(StrictContract):
    lexical_rank_result: ArtifactReference
    rejected_pivot_selector: ArtifactReference
    gold_slice: ArtifactReference
    source_index_registration: ArtifactReference


class RrfRetrieval(StrictContract):
    document_index_path: str = Field(min_length=1)
    analyzer_id: Literal["pyserini_default_english_v1.2.0"]
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
    maximum_rank_per_query: int = Field(ge=100, le=10_000)


class RrfFusion(StrictContract):
    method: Literal["reciprocal_rank_fusion_v0"]
    rrf_k: int = Field(ge=1, le=1_000)
    query_weighting: Literal["uniform"]
    duplicate_query_policy: Literal["reject"]
    candidate_score: Literal["sum_1_over_rrf_k_plus_one_based_rank"]
    tie_break: Literal[
        "best_individual_rank_then_occurrence_count_desc_then_docid_ascending"
    ]
    evaluation_depths: tuple[Literal[20], Literal[100]]


class RrfAcceptance(StrictContract):
    minimum_fused_gold_doc_recall_at20_cases: int = Field(ge=1)
    minimum_fused_gold_doc_recall_at100_cases_for_typed_reranker: int = Field(ge=1)


class RrfBudgets(StrictContract):
    expected_frozen_queries: int = Field(gt=0)
    maximum_offline_bm25_queries: int = Field(gt=0)
    maximum_provider_calls: Literal[0]
    maximum_online_search_calls: Literal[0]
    maximum_judge_calls: Literal[0]
    gpu_allowed: Literal[False]


class MultiQueryRrfRegistration(StrictContract):
    schema_version: Literal["multi-query-rrf-registration-v0"]
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    prerequisites: RrfPrerequisites
    baseline_run_root: str = Field(min_length=1)
    query_ids: tuple[str, ...] = Field(min_length=1)
    query_source: Literal["trial1_recorded_successful_search_calls"]
    retrieval: RrfRetrieval
    fusion: RrfFusion
    acceptance: RrfAcceptance
    budgets: RrfBudgets
    sealed_holdout_access: Literal["forbidden"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def registration_is_consistent(self) -> "MultiQueryRrfRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("RRF query IDs must be unique")
        if self.fusion.evaluation_depths != (20, 100):
            raise ValueError("RRF evaluation depths must remain 20 and 100")
        if self.retrieval.maximum_rank_per_query < self.fusion.evaluation_depths[-1]:
            raise ValueError("RRF retrieval depth does not cover evaluation")
        if (
            self.budgets.expected_frozen_queries
            != self.budgets.maximum_offline_bm25_queries
        ):
            raise ValueError("RRF frozen-query count and offline budget differ")
        for threshold in (
            self.acceptance.minimum_fused_gold_doc_recall_at20_cases,
            self.acceptance.minimum_fused_gold_doc_recall_at100_cases_for_typed_reranker,
        ):
            if threshold > len(self.query_ids):
                raise ValueError("RRF acceptance exceeds registered cases")
        return self


class RrfQueryRanking(StrictContract):
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    returned_hits: int = Field(gt=0)
    ranked_docids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ranking_is_consistent(self) -> "RrfQueryRanking":
        if self.returned_hits != len(self.ranked_docids):
            raise ValueError("RRF returned-hit count differs from ranking")
        if len(self.ranked_docids) != len(set(self.ranked_docids)):
            raise ValueError("RRF ranking contains duplicate documents")
        return self


class FusedCandidate(StrictContract):
    docid: str = Field(min_length=1)
    score: float = Field(gt=0)
    best_individual_rank: int = Field(ge=1)
    occurrence_count: int = Field(ge=1)
    source_query_indices: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sources_are_consistent(self) -> "FusedCandidate":
        if self.occurrence_count != len(self.source_query_indices):
            raise ValueError("RRF occurrence count differs from source queries")
        if len(self.source_query_indices) != len(set(self.source_query_indices)):
            raise ValueError("RRF candidate repeats a source query")
        return self


class RrfSlateCase(StrictContract):
    query_id: str = Field(min_length=1)
    run_path: str = Field(min_length=1)
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    queries: tuple[RrfQueryRanking, ...] = Field(min_length=1)
    fused_candidate_count: int = Field(gt=0)
    fused_top100: tuple[FusedCandidate, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def fused_count_covers_saved_candidates(self) -> "RrfSlateCase":
        if self.fused_candidate_count < len(self.fused_top100):
            raise ValueError("RRF fused candidate count is too small")
        return self


class MultiQueryRrfSlate(StrictContract):
    schema_version: Literal["multi-query-rrf-slate-v0"] = "multi-query-rrf-slate-v0"
    created_at: str = Field(min_length=1)
    status: Literal["built_gold_blind"] = "built_gold_blind"
    registration: ArtifactReference
    query_count: int = Field(gt=0)
    frozen_query_count: int = Field(gt=0)
    offline_bm25_queries: int = Field(gt=0)
    search_latency_ms: float = Field(ge=0)
    gold_inputs_opened: Literal[False] = False
    lexical_rank_result_opened: Literal[False] = False
    rejected_pivot_result_opened: Literal[False] = False
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    gpu_used: Literal[False] = False
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[RrfSlateCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> "MultiQueryRrfSlate":
        if self.query_count != len(self.items):
            raise ValueError("RRF case count differs from slate")
        actual_queries = sum(len(item.queries) for item in self.items)
        if self.frozen_query_count != actual_queries:
            raise ValueError("RRF frozen-query count differs from slate")
        if self.offline_bm25_queries != actual_queries:
            raise ValueError("RRF offline query count differs from slate")
        return self


class RrfScoreCase(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    best_fused_gold_rank: int | None = Field(default=None, ge=1, le=100)
    matched_gold_docid: str | None = None
    fused_gold_hit_at20: bool
    fused_gold_hit_at100: bool

    @model_validator(mode="after")
    def hit_flags_match_rank(self) -> "RrfScoreCase":
        if self.fused_gold_hit_at20 != (
            self.best_fused_gold_rank is not None and self.best_fused_gold_rank <= 20
        ):
            raise ValueError("RRF top-20 flag differs from rank")
        if self.fused_gold_hit_at100 != (self.best_fused_gold_rank is not None):
            raise ValueError("RRF top-100 flag differs from rank")
        if (self.matched_gold_docid is None) != (self.best_fused_gold_rank is None):
            raise ValueError("RRF matched gold document differs from rank")
        return self


class RrfGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int = Field(ge=0)
    operator: Literal["ge"] = "ge"
    threshold: int = Field(ge=1)
    passed: bool


class MultiQueryRrfResult(StrictContract):
    schema_version: Literal["multi-query-rrf-result-v0"] = "multi-query-rrf-result-v0"
    created_at: str = Field(min_length=1)
    status: Literal["succeeded"] = "succeeded"
    decision: Literal[
        "multi_query_rrf_candidate",
        "typed_reranker_candidate",
        "freeze_multi_query_rrf",
    ]
    registration: ArtifactReference
    slate: ArtifactReference
    query_count: int = Field(gt=0)
    frozen_query_count: int = Field(gt=0)
    single_query_gold_hit_cases_at20: int = Field(ge=0)
    single_query_gold_hit_cases_at100: int = Field(ge=0)
    fused_gold_hit_cases_at20: int = Field(ge=0)
    fused_gold_hit_cases_at100: int = Field(ge=0)
    fused_minus_single_cases_at20: int
    fused_minus_single_cases_at100: int
    offline_bm25_queries: int = Field(gt=0)
    search_latency_ms: float = Field(ge=0)
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    gpu_used: Literal[False] = False
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[RrfScoreCase, ...] = Field(min_length=1)
    gates: tuple[RrfGate, RrfGate]
    next_action: Literal[
        "preregister_unseen_fixed_budget_rrf_comparison",
        "preregister_typed_reranker_over_fused_top100",
        "freeze_rrf_and_reassess_candidate_representation",
    ]
    claim_boundary: str = Field(min_length=1)


def fuse_rankings(
    rankings: Sequence[Sequence[str]], *, rrf_k: int
) -> tuple[FusedCandidate, ...]:
    if rrf_k < 1:
        raise ValueError("RRF k must be positive")
    if not rankings:
        raise ValueError("RRF requires at least one ranking")
    scores: dict[str, float] = defaultdict(float)
    best_ranks: dict[str, int] = {}
    sources: dict[str, list[int]] = defaultdict(list)
    for query_index, ranking in enumerate(rankings):
        docids = tuple(str(docid) for docid in ranking)
        if not docids or any(not docid for docid in docids):
            raise ValueError("RRF ranking is blank")
        if len(docids) != len(set(docids)):
            raise ValueError("RRF ranking contains duplicate documents")
        for rank, docid in enumerate(docids, start=1):
            scores[docid] += 1.0 / (rrf_k + rank)
            best_ranks[docid] = min(best_ranks.get(docid, rank), rank)
            sources[docid].append(query_index)
    candidates = [
        FusedCandidate(
            docid=docid,
            score=score,
            best_individual_rank=best_ranks[docid],
            occurrence_count=len(sources[docid]),
            source_query_indices=tuple(sources[docid]),
        )
        for docid, score in scores.items()
    ]
    candidates.sort(
        key=lambda item: (
            -item.score,
            item.best_individual_rank,
            -item.occurrence_count,
            item.docid,
        )
    )
    return tuple(candidates)


def load_multi_query_rrf_registration(path: Path) -> MultiQueryRrfRegistration:
    registration = _parse_registration(path)
    root = path.resolve().parents[2]
    for artifact in (
        registration.prerequisites.lexical_rank_result,
        registration.prerequisites.rejected_pivot_selector,
        registration.prerequisites.gold_slice,
        registration.prerequisites.source_index_registration,
    ):
        _validate_artifact(root, artifact)
    _validate_source_index(root, registration)
    baseline = (root / registration.baseline_run_root).resolve()
    if not baseline.is_relative_to(root) or not baseline.is_dir():
        raise ValueError("RRF baseline run root is missing or escapes repository")
    return registration


def build_multi_query_rrf_slate(
    *,
    registration_path: Path,
    output_path: Path,
    search: DocumentSearch,
) -> MultiQueryRrfSlate:
    registration = _parse_registration(registration_path)
    root = registration_path.resolve().parents[2]
    output_path = output_path.resolve()
    _require_run_output(output_path, root)
    if output_path.exists():
        raise ValueError("RRF slate already exists")

    # Build validates only the non-gold index contract. Scoring artifacts stay unopened
    # until the complete fused slate has been atomically persisted.
    source_ref = registration.prerequisites.source_index_registration
    _validate_artifact(root, source_ref)
    _validate_source_index(root, registration)
    baseline = (root / registration.baseline_run_root).resolve()
    if not baseline.is_relative_to(root) or not baseline.is_dir():
        raise ValueError("RRF baseline run root is missing or escapes repository")

    items: list[RrfSlateCase] = []
    seen_queries: set[str] = set()
    offline_queries = 0
    latency_ms = 0.0
    for query_id in registration.query_ids:
        run_path = baseline / query_id / "run.json"
        run = _read_object(run_path)
        if str(run.get("query_id")) != query_id:
            raise ValueError(f"RRF baseline run ID changed: {query_id}")
        calls = run.get("search_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"RRF baseline run has no searches: {query_id}")
        observations: list[RrfQueryRanking] = []
        rankings: list[tuple[str, ...]] = []
        for call in calls:
            if not isinstance(call, dict) or call.get("outcome") != "ok":
                raise ValueError(f"RRF baseline run has a failed search: {query_id}")
            query = call.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"RRF baseline run has a blank query: {query_id}")
            if query in seen_queries:
                raise ValueError(f"RRF duplicate frozen query: {query}")
            seen_queries.add(query)
            started = perf_counter()
            docids = tuple(
                str(docid)
                for docid in search(query, registration.retrieval.maximum_rank_per_query)
            )
            latency_ms += (perf_counter() - started) * 1000
            offline_queries += 1
            if len(docids) > registration.retrieval.maximum_rank_per_query:
                raise ValueError("RRF search exceeded registered rank depth")
            if not docids:
                raise ValueError(f"RRF search returned no documents: {query_id}")
            observation = RrfQueryRanking(
                query=query,
                query_sha256=_text_sha256(query),
                returned_hits=len(docids),
                ranked_docids=docids,
            )
            observations.append(observation)
            rankings.append(docids)
        fused = fuse_rankings(rankings, rrf_k=registration.fusion.rrf_k)
        relative_run = run_path.resolve().relative_to(root).as_posix()
        items.append(
            RrfSlateCase(
                query_id=query_id,
                run_path=relative_run,
                run_sha256=sha256_file(run_path),
                queries=tuple(observations),
                fused_candidate_count=len(fused),
                fused_top100=fused[: registration.fusion.evaluation_depths[-1]],
            )
        )
    if offline_queries != registration.budgets.expected_frozen_queries:
        raise ValueError(
            "RRF frozen query count differs from registration: "
            f"{offline_queries} != {registration.budgets.expected_frozen_queries}"
        )
    if offline_queries > registration.budgets.maximum_offline_bm25_queries:
        raise ValueError("RRF offline BM25 budget exceeded")

    slate = MultiQueryRrfSlate(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(registration_path),
        ),
        query_count=len(items),
        frozen_query_count=offline_queries,
        offline_bm25_queries=offline_queries,
        search_latency_ms=latency_ms,
        items=tuple(items),
    )
    _atomic_write(output_path, slate.model_dump(mode="json"))
    return slate


def score_multi_query_rrf_slate(
    *, registration_path: Path, slate_path: Path, output_path: Path
) -> MultiQueryRrfResult:
    registration = load_multi_query_rrf_registration(registration_path)
    root = registration_path.resolve().parents[2]
    slate_path = slate_path.resolve()
    output_path = output_path.resolve()
    _require_run_output(slate_path, root)
    _require_run_output(output_path, root)
    if output_path.exists():
        raise ValueError("RRF result already exists")
    if not slate_path.is_file():
        raise ValueError("RRF slate is missing")
    slate = MultiQueryRrfSlate.model_validate_json(slate_path.read_text(encoding="utf-8"))
    expected_registration_path = registration_path.resolve().relative_to(root).as_posix()
    if (
        slate.registration.path != expected_registration_path
        or slate.registration.sha256 != sha256_file(registration_path)
    ):
        raise ValueError("RRF slate registration binding changed")
    if tuple(item.query_id for item in slate.items) != registration.query_ids:
        raise ValueError("RRF slate cases differ from registration")
    if slate.frozen_query_count != registration.budgets.expected_frozen_queries:
        raise ValueError("RRF slate query count differs from registration")

    lexical = _read_object(root / registration.prerequisites.lexical_rank_result.path)
    if lexical.get("decision") != "index_representation_diagnosis_required":
        raise ValueError("RRF lexical prerequisite selected another branch")
    pivot = _read_object(root / registration.prerequisites.rejected_pivot_selector.path)
    if pivot.get("decision") != "freeze_gold_blind_pivot_branch":
        raise ValueError("RRF pivot prerequisite selected another branch")
    lexical_items = _object_items(lexical, "lexical rank")
    lexical_by_id = {str(item.get("query_id")): item for item in lexical_items}
    if tuple(str(item.get("query_id")) for item in lexical_items) != registration.query_ids:
        raise ValueError("RRF lexical cases differ from registration")
    single20 = int(lexical.get("generated_query_top20_cases", -1))
    single100 = int(lexical.get("generated_query_top100_cases", -1))
    if single20 < 0 or single100 < 0:
        raise ValueError("RRF lexical baseline metrics are missing")

    gold = _read_object(root / registration.prerequisites.gold_slice.path)
    rows = gold.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("RRF gold rows are invalid")
    gold_by_id = {str(row.get("query_id")): row for row in rows}

    scored: list[RrfScoreCase] = []
    for slate_item in slate.items:
        run_path = (root / slate_item.run_path).resolve()
        if not run_path.is_relative_to(root) or not run_path.is_file():
            raise ValueError(f"RRF baseline run is missing: {slate_item.query_id}")
        if sha256_file(run_path) != slate_item.run_sha256:
            raise ValueError(f"RRF baseline run changed: {slate_item.query_id}")
        lexical_item = lexical_by_id[slate_item.query_id]
        recorded = lexical_item.get("generated_queries")
        if not isinstance(recorded, list) or len(recorded) != len(slate_item.queries):
            raise ValueError(f"RRF lexical query count changed: {slate_item.query_id}")
        for saved, expected in zip(slate_item.queries, recorded, strict=True):
            if not isinstance(expected, dict) or (
                expected.get("query") != saved.query
                or expected.get("query_sha256") != saved.query_sha256
            ):
                raise ValueError(f"RRF frozen query sequence changed: {slate_item.query_id}")
        gold_row = gold_by_id.get(slate_item.query_id)
        if gold_row is None or not isinstance(gold_row.get("gold_docids"), list):
            raise ValueError(f"RRF gold case is missing: {slate_item.query_id}")
        gold_docids = tuple(str(docid) for docid in gold_row["gold_docids"])
        if not gold_docids:
            raise ValueError(f"RRF gold documents are empty: {slate_item.query_id}")
        gold_set = set(gold_docids)
        match = next(
            (
                (rank, candidate.docid)
                for rank, candidate in enumerate(slate_item.fused_top100, start=1)
                if candidate.docid in gold_set
            ),
            None,
        )
        rank = match[0] if match else None
        matched = match[1] if match else None
        scored.append(
            RrfScoreCase(
                query_id=slate_item.query_id,
                gold_docids=gold_docids,
                best_fused_gold_rank=rank,
                matched_gold_docid=matched,
                fused_gold_hit_at20=rank is not None and rank <= 20,
                fused_gold_hit_at100=rank is not None,
            )
        )

    fused20 = sum(item.fused_gold_hit_at20 for item in scored)
    fused100 = sum(item.fused_gold_hit_at100 for item in scored)
    top20_pass = (
        fused20 >= registration.acceptance.minimum_fused_gold_doc_recall_at20_cases
    )
    top100_pass = (
        fused100
        >= registration.acceptance.minimum_fused_gold_doc_recall_at100_cases_for_typed_reranker
    )
    if top20_pass:
        decision = "multi_query_rrf_candidate"
        next_action = "preregister_unseen_fixed_budget_rrf_comparison"
    elif top100_pass:
        decision = "typed_reranker_candidate"
        next_action = "preregister_typed_reranker_over_fused_top100"
    else:
        decision = "freeze_multi_query_rrf"
        next_action = "freeze_rrf_and_reassess_candidate_representation"
    gates = (
        RrfGate(
            gate_id="fused_gold_doc_recall_at20_cases",
            observed=fused20,
            threshold=registration.acceptance.minimum_fused_gold_doc_recall_at20_cases,
            passed=top20_pass,
        ),
        RrfGate(
            gate_id="fused_gold_doc_recall_at100_cases_for_typed_reranker",
            observed=fused100,
            threshold=(
                registration.acceptance.minimum_fused_gold_doc_recall_at100_cases_for_typed_reranker
            ),
            passed=top100_pass,
        ),
    )
    result = MultiQueryRrfResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(registration_path),
        ),
        slate=ArtifactReference(
            path=slate_path.relative_to(root).as_posix(), sha256=sha256_file(slate_path)
        ),
        query_count=len(scored),
        frozen_query_count=slate.frozen_query_count,
        single_query_gold_hit_cases_at20=single20,
        single_query_gold_hit_cases_at100=single100,
        fused_gold_hit_cases_at20=fused20,
        fused_gold_hit_cases_at100=fused100,
        fused_minus_single_cases_at20=fused20 - single20,
        fused_minus_single_cases_at100=fused100 - single100,
        offline_bm25_queries=slate.offline_bm25_queries,
        search_latency_ms=slate.search_latency_ms,
        items=tuple(scored),
        gates=gates,
        next_action=next_action,
        claim_boundary=registration.claim_boundary,
    )
    _atomic_write(output_path, result.model_dump(mode="json"))
    return result


def _parse_registration(path: Path) -> MultiQueryRrfRegistration:
    return MultiQueryRrfRegistration.model_validate_json(path.read_text(encoding="utf-8"))


def _validate_source_index(root: Path, registration: MultiQueryRrfRegistration) -> None:
    source_path = root / registration.prerequisites.source_index_registration.path
    source_payload = _read_object(source_path)
    source = SourceDocumentIndex.model_validate(source_payload.get("source_document_index"))
    if source.path != registration.retrieval.document_index_path:
        raise ValueError("RRF source index path differs from registration")
    index_path = (root / source.path).resolve()
    if not index_path.is_relative_to(root) or not index_path.is_dir():
        raise ValueError("RRF source index is missing or escapes repository")
    actual_names = {
        path.name
        for path in index_path.iterdir()
        if path.is_file() and path.name != "write.lock"
    }
    expected_names = {item.name for item in source.files}
    if actual_names != expected_names:
        raise ValueError("RRF source index file set changed")
    for item in source.files:
        path = index_path / item.name
        if path.stat().st_size != item.bytes or sha256_file(path) != item.sha256:
            raise ValueError(f"RRF source index file changed: {item.name}")


def _validate_artifact(root: Path, artifact: ArtifactReference) -> None:
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"RRF artifact is missing or escapes repository: {artifact.path}")
    if sha256_file(path) != artifact.sha256:
        raise ValueError(f"RRF artifact hash changed: {artifact.path}")


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"RRF artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"RRF artifact is not an object: {path}")
    return value


def _object_items(value: dict[str, object], label: str) -> list[dict[str, object]]:
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"RRF {label} items are invalid")
    return items


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _require_run_output(path: Path, root: Path) -> None:
    if not path.is_relative_to((root / "runs").resolve()):
        raise ValueError("RRF output must stay under ignored runs/")


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
