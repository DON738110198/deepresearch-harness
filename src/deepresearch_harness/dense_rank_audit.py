from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import normalized_text_file_sha256
from .evidence_span_oracle import ArtifactReference
from .retrieval_replay import (
    DenseRuntimeSnapshot,
    RankedHit,
    RetrieverCandidatesManifest,
    select_candidate,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizedArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DenseRankAcceptance(StrictContract):
    minimum_dense_top20_cases: int = Field(ge=1)
    minimum_dense_top100_cases_for_pool_diagnosis: int = Field(ge=1)


class DenseRankBudgets(StrictContract):
    maximum_provider_calls: Literal[0]
    maximum_online_search_calls: Literal[0]
    maximum_judge_calls: Literal[0]


class DenseRankRegistration(StrictContract):
    schema_version: Literal["persistent-miss-dense-rank-registration-v0"]
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    lexical_rank_result: ArtifactReference
    passage_index_result: ArtifactReference
    gold_slice: ArtifactReference
    baseline_run_root: str = Field(min_length=1)
    query_ids: tuple[str, ...] = Field(min_length=1)
    retriever_manifest: NormalizedArtifactReference
    candidate_id: str = Field(min_length=1)
    model_directory: str = Field(min_length=1)
    index_root: str = Field(min_length=1)
    query_source: Literal["trial1_recorded_successful_search_calls"]
    depths: tuple[int, ...]
    batch_size: int = Field(ge=1, le=64)
    acceptance: DenseRankAcceptance
    budgets: DenseRankBudgets
    sealed_holdout_access: Literal["forbidden"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def registration_is_consistent(self) -> "DenseRankRegistration":
        if self.depths != (20, 100, 1000):
            raise ValueError("dense-rank depths must remain exactly 20, 100, and 1000")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("dense-rank query IDs must be unique")
        for threshold in (
            self.acceptance.minimum_dense_top20_cases,
            self.acceptance.minimum_dense_top100_cases_for_pool_diagnosis,
        ):
            if threshold > len(self.query_ids):
                raise ValueError("dense-rank gate exceeds registered cases")
        return self


class FrozenDenseRankQuery(StrictContract):
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrozenDenseRankCase(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    queries: tuple[FrozenDenseRankQuery, ...] = Field(min_length=1)
    passage_gold_hit_at20: bool


class DenseRankQueryObservation(StrictContract):
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    returned_hits: int = Field(ge=1)
    ranked_hits: tuple[RankedHit, ...] = Field(min_length=1, max_length=1000)
    gold_rank: int | None = Field(default=None, ge=1, le=1000)
    matched_gold_docid: str | None = None
    gold_hit_at20: bool
    gold_hit_at100: bool
    gold_hit_at1000: bool

    @model_validator(mode="after")
    def rank_and_flags_are_consistent(self) -> "DenseRankQueryObservation":
        if self.returned_hits != len(self.ranked_hits):
            raise ValueError("dense-rank returned hit count differs from ranking")
        if (self.gold_rank is None) != (self.matched_gold_docid is None):
            raise ValueError("dense-rank gold rank and matched document must coexist")
        expected = (
            self.gold_rank is not None and self.gold_rank <= 20,
            self.gold_rank is not None and self.gold_rank <= 100,
            self.gold_rank is not None and self.gold_rank <= 1000,
        )
        if expected != (
            self.gold_hit_at20,
            self.gold_hit_at100,
            self.gold_hit_at1000,
        ):
            raise ValueError("dense-rank hit flags differ from observed rank")
        return self


class DenseRankCase(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    queries: tuple[DenseRankQueryObservation, ...] = Field(min_length=1)
    best_gold_rank: int | None = Field(default=None, ge=1, le=1000)
    best_query_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dense_gold_hit_at20: bool
    dense_gold_hit_at100: bool
    dense_gold_hit_at1000: bool
    passage_gold_hit_at20: bool

    @model_validator(mode="after")
    def best_rank_is_consistent(self) -> "DenseRankCase":
        ranked = [query for query in self.queries if query.gold_rank is not None]
        expected_rank = min((query.gold_rank for query in ranked), default=None)
        if self.best_gold_rank != expected_rank:
            raise ValueError("dense-rank case best rank differs from query observations")
        expected_query_hash = None
        if expected_rank is not None:
            expected_query_hash = next(
                query.query_sha256 for query in ranked if query.gold_rank == expected_rank
            )
        if self.best_query_sha256 != expected_query_hash:
            raise ValueError("dense-rank best query hash differs from best rank")
        expected_flags = (
            expected_rank is not None and expected_rank <= 20,
            expected_rank is not None and expected_rank <= 100,
            expected_rank is not None and expected_rank <= 1000,
        )
        if expected_flags != (
            self.dense_gold_hit_at20,
            self.dense_gold_hit_at100,
            self.dense_gold_hit_at1000,
        ):
            raise ValueError("dense-rank case flags differ from best rank")
        return self


class DenseRankGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int = Field(ge=0)
    operator: Literal["ge"] = "ge"
    threshold: int = Field(ge=1)
    passed: bool


DenseRankDecision = Literal[
    "dense_top20_candidate",
    "dense_pool_reranker_diagnosis",
    "freeze_dense_channel",
]


DenseRankNextAction = Literal[
    "preregister_fresh_dense_retrieval_comparison",
    "preregister_bounded_offline_reranker_gate",
    "freeze_dense_and_diagnose_query_entity_bridge",
]


class DenseRankResult(StrictContract):
    schema_version: Literal["persistent-miss-dense-rank-result-v0"] = (
        "persistent-miss-dense-rank-result-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    official_accuracy_status: Literal["not_run_diagnostic_only"] = (
        "not_run_diagnostic_only"
    )
    decision: DenseRankDecision
    registration: ArtifactReference
    retriever_manifest: NormalizedArtifactReference
    candidate_id: str = Field(min_length=1)
    query_source: Literal["trial1_recorded_successful_search_calls"]
    depths: tuple[int, int, int]
    query_count: int = Field(gt=0)
    generated_query_count: int = Field(gt=0)
    unique_dense_query_count: int = Field(gt=0)
    returned_hits_per_query: int = Field(gt=0)
    passage_gold_hit_cases_at20: int = Field(ge=0)
    dense_gold_hit_cases_at20: int = Field(ge=0)
    dense_gold_hit_cases_at100: int = Field(ge=0)
    dense_gold_hit_cases_at1000: int = Field(ge=0)
    dense_minus_passage_hit_cases_at20: int
    dense_wins_at20: int = Field(ge=0)
    dense_losses_at20: int = Field(ge=0)
    runtime: DenseRuntimeSnapshot
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[DenseRankCase, ...] = Field(min_length=1)
    gates: tuple[DenseRankGate, DenseRankGate]
    next_action: DenseRankNextAction
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def aggregate_counts_are_consistent(self) -> "DenseRankResult":
        if self.depths != (20, 100, 1000):
            raise ValueError("dense-rank result depths changed")
        if self.returned_hits_per_query != self.depths[-1]:
            raise ValueError("dense-rank result did not search the registered maximum depth")
        if self.query_count != len(self.items):
            raise ValueError("dense-rank query count differs from items")
        if self.generated_query_count != sum(len(item.queries) for item in self.items):
            raise ValueError("dense-rank generated query count differs from items")
        expected_counts = (
            sum(item.passage_gold_hit_at20 for item in self.items),
            sum(item.dense_gold_hit_at20 for item in self.items),
            sum(item.dense_gold_hit_at100 for item in self.items),
            sum(item.dense_gold_hit_at1000 for item in self.items),
        )
        if expected_counts != (
            self.passage_gold_hit_cases_at20,
            self.dense_gold_hit_cases_at20,
            self.dense_gold_hit_cases_at100,
            self.dense_gold_hit_cases_at1000,
        ):
            raise ValueError("dense-rank aggregate counts differ from items")
        return self


def load_dense_rank_registration(path: Path) -> DenseRankRegistration:
    registration = DenseRankRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = _repository_root(path)
    _validate_artifact(root, registration.lexical_rank_result)
    _validate_artifact(root, registration.passage_index_result)
    _validate_artifact(root, registration.gold_slice)
    _validate_directory(root, registration.baseline_run_root)

    retriever_path = _resolve_file(root, registration.retriever_manifest.path)
    if normalized_text_file_sha256(retriever_path) != (
        registration.retriever_manifest.normalized_sha256
    ):
        raise ValueError("dense-rank retriever manifest hash changed")
    manifest = RetrieverCandidatesManifest.model_validate_json(
        retriever_path.read_text(encoding="utf-8")
    )
    candidate = select_candidate(manifest, registration.candidate_id)
    model_dir = _validate_directory(root, registration.model_directory)
    index_root = _validate_directory(root, registration.index_root)
    if not (model_dir / candidate.model.model_file).is_file():
        raise ValueError("dense-rank pinned model file is missing")
    index_dir = index_root / candidate.index.subdirectory
    if not index_dir.is_dir():
        raise ValueError("dense-rank pinned index directory is missing")
    if any(not (index_dir / shard.filename).is_file() for shard in candidate.index.shards):
        raise ValueError("dense-rank pinned index shard is missing")

    lexical = _read_object(root / registration.lexical_rank_result.path)
    if lexical.get("decision") != "index_representation_diagnosis_required":
        raise ValueError("dense-rank lexical prerequisite selected another branch")
    if lexical.get("generated_query_top20_cases") != 0:
        raise ValueError("dense-rank lexical top-20 prerequisite changed")
    if _item_ids(lexical) != registration.query_ids:
        raise ValueError("dense-rank cases differ from lexical-rank result")

    passage = _read_object(root / registration.passage_index_result.path)
    if passage.get("decision") != "freeze_passage_index_branch":
        raise ValueError("dense-rank passage prerequisite selected another branch")
    if passage.get("full_document_gold_hit_cases_at20") != 0:
        raise ValueError("dense-rank full-document prerequisite changed")
    if _item_ids(passage) != registration.query_ids:
        raise ValueError("dense-rank cases differ from passage-index result")

    gold = _read_object(root / registration.gold_slice.path)
    gold_by_id = _gold_rows(gold)
    missing_gold = [query_id for query_id in registration.query_ids if query_id not in gold_by_id]
    if missing_gold:
        raise ValueError(f"dense-rank development gold is missing cases: {missing_gold}")
    return registration


def collect_dense_rank_cases(path: Path) -> tuple[FrozenDenseRankCase, ...]:
    registration = load_dense_rank_registration(path)
    root = _repository_root(path)
    passage = _read_object(root / registration.passage_index_result.path)
    gold = _read_object(root / registration.gold_slice.path)
    passage_by_id = {
        str(item["query_id"]): item for item in _items(passage, label="passage-index")
    }
    gold_by_id = _gold_rows(gold)
    frozen: list[FrozenDenseRankCase] = []
    for query_id in registration.query_ids:
        passage_case = passage_by_id[query_id]
        gold_docids = tuple(str(value) for value in gold_by_id[query_id]["gold_docids"])
        passage_gold_docids = tuple(str(value) for value in passage_case["gold_docids"])
        if passage_gold_docids != gold_docids:
            raise ValueError(f"dense-rank gold documents changed for query {query_id}")

        run_path = root / registration.baseline_run_root / query_id / "run.json"
        run = _read_object(run_path)
        if str(run.get("query_id")) != query_id:
            raise ValueError(f"dense-rank baseline run ID changed for query {query_id}")
        calls = run.get("search_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"dense-rank baseline run has no searches: {query_id}")
        passage_queries = passage_case.get("queries")
        if not isinstance(passage_queries, list) or len(passage_queries) != len(calls):
            raise ValueError(f"dense-rank passage query count changed for query {query_id}")

        queries: list[FrozenDenseRankQuery] = []
        for call, recorded in zip(calls, passage_queries, strict=True):
            if not isinstance(call, dict) or call.get("outcome") != "ok":
                raise ValueError(f"dense-rank baseline run has failed search: {query_id}")
            if not isinstance(recorded, dict):
                raise ValueError(f"dense-rank passage query is invalid: {query_id}")
            query = call.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"dense-rank baseline run has blank query: {query_id}")
            query_hash = _text_sha256(query)
            if recorded.get("query") != query or recorded.get("query_sha256") != query_hash:
                raise ValueError(f"dense-rank recorded query sequence changed for query {query_id}")
            queries.append(FrozenDenseRankQuery(query=query, query_sha256=query_hash))
        frozen.append(
            FrozenDenseRankCase(
                query_id=query_id,
                gold_docids=gold_docids,
                queries=tuple(queries),
                passage_gold_hit_at20=bool(passage_case["passage_gold_hit"]),
            )
        )
    return tuple(frozen)


def run_dense_rank_audit(
    *,
    registration_path: Path,
    candidate_results: Mapping[str, Sequence[RankedHit]],
    runtime: DenseRuntimeSnapshot,
    output_path: Path,
) -> DenseRankResult:
    registration = load_dense_rank_registration(registration_path)
    root = _repository_root(registration_path)
    output_path = output_path.resolve()
    if not output_path.is_relative_to((root / "runs").resolve()):
        raise ValueError("dense-rank output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("dense-rank result already exists")

    manifest_path = _resolve_file(root, registration.retriever_manifest.path)
    manifest = RetrieverCandidatesManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    candidate = select_candidate(manifest, registration.candidate_id)
    if runtime.device != "cpu":
        raise ValueError("dense-rank audit must run on CPU")
    if runtime.model_file_sha256 != candidate.model.model_file_sha256:
        raise ValueError("dense-rank runtime model hash differs from registration")
    expected_shards = {shard.filename: shard.sha256 for shard in candidate.index.shards}
    if runtime.index_shards_sha256 != expected_shards:
        raise ValueError("dense-rank runtime index hashes differ from registration")

    frozen = collect_dense_rank_cases(registration_path)
    ordered_queries = [query.query for item in frozen for query in item.queries]
    unique_queries = list(dict.fromkeys(ordered_queries))
    if set(candidate_results) != set(unique_queries):
        missing = sorted(set(unique_queries) - set(candidate_results))
        extras = sorted(set(candidate_results) - set(unique_queries))
        raise ValueError(
            "dense-rank results differ from frozen queries: "
            f"missing={len(missing)}, extras={len(extras)}"
        )

    maximum_depth = registration.depths[-1]
    normalized_results: dict[str, tuple[RankedHit, ...]] = {}
    for query in unique_queries:
        hits = tuple(candidate_results[query])
        if len(hits) != maximum_depth:
            raise ValueError(
                f"dense-rank query returned {len(hits)} hits instead of {maximum_depth}"
            )
        docids = [hit.docid for hit in hits]
        if len(docids) != len(set(docids)):
            raise ValueError(f"dense-rank query returned duplicate documents: {query}")
        normalized_results[query] = hits

    items: list[DenseRankCase] = []
    for frozen_case in frozen:
        observations: list[DenseRankQueryObservation] = []
        gold = set(frozen_case.gold_docids)
        for frozen_query in frozen_case.queries:
            hits = normalized_results[frozen_query.query]
            rank = None
            matched = None
            for index, hit in enumerate(hits, start=1):
                if hit.docid in gold:
                    rank = index
                    matched = hit.docid
                    break
            observations.append(
                DenseRankQueryObservation(
                    query=frozen_query.query,
                    query_sha256=frozen_query.query_sha256,
                    returned_hits=len(hits),
                    ranked_hits=hits,
                    gold_rank=rank,
                    matched_gold_docid=matched,
                    gold_hit_at20=rank is not None and rank <= 20,
                    gold_hit_at100=rank is not None and rank <= 100,
                    gold_hit_at1000=rank is not None and rank <= 1000,
                )
            )
        ranked = [item for item in observations if item.gold_rank is not None]
        best_rank = min((item.gold_rank for item in ranked), default=None)
        best_query_hash = None
        if best_rank is not None:
            best_query_hash = next(
                item.query_sha256 for item in ranked if item.gold_rank == best_rank
            )
        items.append(
            DenseRankCase(
                query_id=frozen_case.query_id,
                gold_docids=frozen_case.gold_docids,
                queries=tuple(observations),
                best_gold_rank=best_rank,
                best_query_sha256=best_query_hash,
                dense_gold_hit_at20=best_rank is not None and best_rank <= 20,
                dense_gold_hit_at100=best_rank is not None and best_rank <= 100,
                dense_gold_hit_at1000=best_rank is not None and best_rank <= 1000,
                passage_gold_hit_at20=frozen_case.passage_gold_hit_at20,
            )
        )

    dense20 = sum(item.dense_gold_hit_at20 for item in items)
    dense100 = sum(item.dense_gold_hit_at100 for item in items)
    dense1000 = sum(item.dense_gold_hit_at1000 for item in items)
    passage20 = sum(item.passage_gold_hit_at20 for item in items)
    decision, next_action = choose_dense_rank_decision(
        dense_top20_cases=dense20,
        dense_top100_cases=dense100,
        minimum_top20_cases=registration.acceptance.minimum_dense_top20_cases,
        minimum_top100_cases=(
            registration.acceptance.minimum_dense_top100_cases_for_pool_diagnosis
        ),
    )
    gates = (
        DenseRankGate(
            gate_id="dense_generated_query_gold_doc_recall_at20_cases",
            observed=dense20,
            threshold=registration.acceptance.minimum_dense_top20_cases,
            passed=dense20 >= registration.acceptance.minimum_dense_top20_cases,
        ),
        DenseRankGate(
            gate_id="dense_generated_query_gold_doc_recall_at100_cases_for_pool_diagnosis",
            observed=dense100,
            threshold=(
                registration.acceptance.minimum_dense_top100_cases_for_pool_diagnosis
            ),
            passed=(
                dense100
                >= registration.acceptance.minimum_dense_top100_cases_for_pool_diagnosis
            ),
        ),
    )
    result = DenseRankResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=_file_sha256(registration_path),
        ),
        retriever_manifest=registration.retriever_manifest,
        candidate_id=registration.candidate_id,
        query_source=registration.query_source,
        depths=registration.depths,
        query_count=len(items),
        generated_query_count=len(ordered_queries),
        unique_dense_query_count=len(unique_queries),
        returned_hits_per_query=maximum_depth,
        passage_gold_hit_cases_at20=passage20,
        dense_gold_hit_cases_at20=dense20,
        dense_gold_hit_cases_at100=dense100,
        dense_gold_hit_cases_at1000=dense1000,
        dense_minus_passage_hit_cases_at20=dense20 - passage20,
        dense_wins_at20=sum(
            item.dense_gold_hit_at20 and not item.passage_gold_hit_at20 for item in items
        ),
        dense_losses_at20=sum(
            item.passage_gold_hit_at20 and not item.dense_gold_hit_at20 for item in items
        ),
        runtime=runtime,
        items=tuple(items),
        gates=gates,
        next_action=next_action,
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


def choose_dense_rank_decision(
    *,
    dense_top20_cases: int,
    dense_top100_cases: int,
    minimum_top20_cases: int,
    minimum_top100_cases: int,
) -> tuple[DenseRankDecision, DenseRankNextAction]:
    if dense_top20_cases >= minimum_top20_cases:
        return (
            "dense_top20_candidate",
            "preregister_fresh_dense_retrieval_comparison",
        )
    if dense_top100_cases >= minimum_top100_cases:
        return (
            "dense_pool_reranker_diagnosis",
            "preregister_bounded_offline_reranker_gate",
        )
    return (
        "freeze_dense_channel",
        "freeze_dense_and_diagnose_query_entity_bridge",
    )


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate dense-rank repository root")


def _resolve_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"dense-rank registered file is missing or escapes root: {relative}")
    return path


def _validate_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(
            f"dense-rank registered directory is missing or escapes root: {relative}"
        )
    return path


def _validate_artifact(root: Path, artifact: ArtifactReference) -> None:
    path = _resolve_file(root, artifact.path)
    if _file_sha256(path) != artifact.sha256:
        raise ValueError(f"dense-rank registered artifact hash changed: {artifact.path}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"dense-rank artifact is not an object: {path}")
    return value


def _items(value: dict[str, object], *, label: str) -> list[dict[str, object]]:
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"dense-rank {label} items are invalid")
    return items


def _item_ids(value: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(item["query_id"]) for item in _items(value, label="prerequisite"))


def _gold_rows(value: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("dense-rank development gold rows are invalid")
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        query_id = str(row.get("query_id"))
        gold_docids = row.get("gold_docids")
        if (
            not query_id
            or not isinstance(gold_docids, list)
            or not gold_docids
            or not all(isinstance(docid, str) and docid for docid in gold_docids)
        ):
            raise ValueError("dense-rank development gold row is invalid")
        if query_id in output:
            raise ValueError(f"dense-rank duplicate development gold ID: {query_id}")
        output[query_id] = row
    return output
