from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch_harness.evidence_span_oracle import ArtifactReference


DocumentSearch = Callable[[str, int], Sequence[str]]
DocumentExists = Callable[[str], bool]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexFileDigest(StrictContract):
    name: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PassagePrerequisites(StrictContract):
    lexical_rank_result: ArtifactReference
    stability_audit: ArtifactReference
    gold_slice: ArtifactReference


class SourceDocumentIndex(StrictContract):
    path: str = Field(min_length=1)
    document_count: int = Field(gt=0)
    files: tuple[IndexFileDigest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def files_are_unique(self) -> "SourceDocumentIndex":
        names = [item.name for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError("source index file names must be unique")
        return self


class PassageChunking(StrictContract):
    tokenizer_id: Literal["whitespace_v0"]
    passage_tokens: int = Field(ge=32, le=2_048)
    overlap_tokens: int = Field(ge=0)
    minimum_passage_tokens: int = Field(ge=1)
    header_policy: Literal["repeat_frontmatter_v0"]
    maximum_header_tokens: int = Field(ge=0, le=256)
    passage_id_format: Literal["{docid}::p{ordinal:05d}"]

    @model_validator(mode="after")
    def overlap_is_valid(self) -> "PassageChunking":
        if self.overlap_tokens >= self.passage_tokens:
            raise ValueError("passage overlap must be smaller than passage size")
        if self.minimum_passage_tokens > self.passage_tokens:
            raise ValueError("minimum passage size exceeds passage size")
        return self


class PassageRetrieval(StrictContract):
    query_source: Literal["trial1_recorded_successful_search_calls"]
    analyzer_id: Literal["pyserini_default_english_v1.2.0"]
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
    maximum_passage_hits_per_query: int = Field(ge=20, le=10_000)
    maximum_unique_documents_per_query: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def candidate_caps_are_valid(self) -> "PassageRetrieval":
        if (
            self.maximum_passage_hits_per_query
            < self.maximum_unique_documents_per_query
        ):
            raise ValueError("passage hit cap must cover the unique-document cap")
        return self


class PassageAcceptance(StrictContract):
    minimum_passage_generated_query_gold_doc_recall_at20_cases: int = Field(ge=1)
    required_source_document_coverage_ratio: Literal[1.0]
    required_development_gold_document_coverage_ratio: Literal[1.0]


class PassageBudgets(StrictContract):
    maximum_provider_calls: Literal[0]
    maximum_online_search_calls: Literal[0]
    maximum_judge_calls: Literal[0]


class PassageIndexRegistration(StrictContract):
    schema_version: Literal["passage-index-representation-registration-v0"]
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    prerequisites: PassagePrerequisites
    baseline_run_root: str = Field(min_length=1)
    query_ids: tuple[str, ...] = Field(min_length=1)
    source_document_index: SourceDocumentIndex
    passage_corpus_path: str = Field(min_length=1)
    passage_index_path: str = Field(min_length=1)
    chunking: PassageChunking
    retrieval: PassageRetrieval
    acceptance: PassageAcceptance
    budgets: PassageBudgets
    sealed_holdout_access: Literal["forbidden"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def registration_is_consistent(self) -> "PassageIndexRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("passage-index query IDs must be unique")
        if (
            self.acceptance.minimum_passage_generated_query_gold_doc_recall_at20_cases
            > len(self.query_ids)
        ):
            raise ValueError("passage-index recall gate exceeds registered cases")
        if self.passage_corpus_path == self.passage_index_path:
            raise ValueError("passage corpus and index paths must be distinct")
        return self


class PassageRecord(StrictContract):
    id: str = Field(min_length=1)
    parent_docid: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    contents: str = Field(min_length=1)
    body_token_count: int = Field(ge=1)
    header_token_count: int = Field(ge=0)

    def collection_row(self) -> dict[str, str]:
        return {"id": self.id, "contents": self.contents}


class PassageExportStats(StrictContract):
    source_document_count: int = Field(ge=0)
    source_documents_with_passages: int = Field(ge=0)
    passage_count: int = Field(ge=0)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_docids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passage_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_latency_ms: float = Field(ge=0)


class PassageExportManifest(StrictContract):
    schema_version: Literal["passage-index-export-manifest-v0"] = (
        "passage-index-export-manifest-v0"
    )
    created_at: str = Field(min_length=1)
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_file: str = Field(min_length=1)
    chunking: PassageChunking
    source_document_count: int = Field(ge=0)
    source_documents_with_passages: int = Field(ge=0)
    passage_count: int = Field(ge=0)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_docids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passage_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_latency_ms: float = Field(ge=0)


class PassageIndexBuildManifest(StrictContract):
    schema_version: Literal["passage-index-build-manifest-v0"] = (
        "passage-index-build-manifest-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded"] = "succeeded"
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_manifest: ArtifactReference
    index_path: str = Field(min_length=1)
    index_document_count: int = Field(gt=0)
    index_files: tuple[IndexFileDigest, ...] = Field(min_length=1)
    pyserini_version: str = Field(min_length=1)
    java_version: str = Field(min_length=1)
    index_command: tuple[str, ...] = Field(min_length=1)
    build_latency_ms: float = Field(ge=0)
    completion_mode: Literal["direct", "recovered_completed_partial"] = "direct"
    recovery_artifact: ArtifactReference | None = None

    @model_validator(mode="after")
    def recovery_is_explicit(self) -> "PassageIndexBuildManifest":
        if self.completion_mode == "recovered_completed_partial":
            if self.recovery_artifact is None:
                raise ValueError("recovered passage index requires a recovery artifact")
        elif self.recovery_artifact is not None:
            raise ValueError("direct passage index build cannot carry recovery evidence")
        return self


class PassageQueryObservation(StrictContract):
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    full_document_top20: tuple[str, ...] = Field(max_length=20)
    passage_top20_parent_documents: tuple[str, ...] = Field(max_length=20)
    passage_hits_examined: int = Field(ge=0)
    full_document_gold_hit: bool
    passage_gold_hit: bool
    full_document_latency_ms: float = Field(ge=0)
    passage_latency_ms: float = Field(ge=0)


class PassageIndexCase(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    queries: tuple[PassageQueryObservation, ...] = Field(min_length=1)
    full_document_gold_hit: bool
    passage_gold_hit: bool


class PassageIndexGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: float
    operator: Literal["ge", "eq"]
    threshold: float
    passed: bool


class PassageIndexResult(StrictContract):
    schema_version: Literal["passage-index-representation-result-v0"] = (
        "passage-index-representation-result-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded"] = "succeeded"
    decision: Literal["passage_index_candidate", "freeze_passage_index_branch"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_manifest: ArtifactReference
    build_manifest: ArtifactReference
    query_count: int = Field(ge=0)
    generated_query_count: int = Field(ge=0)
    full_document_gold_hit_cases_at20: int = Field(ge=0)
    passage_gold_hit_cases_at20: int = Field(ge=0)
    passage_minus_full_document_hit_cases: int
    passage_wins: int = Field(ge=0)
    passage_losses: int = Field(ge=0)
    source_document_count: int = Field(ge=0)
    source_documents_with_passages: int = Field(ge=0)
    passage_count: int = Field(ge=0)
    passage_index_document_count: int = Field(ge=0)
    source_document_coverage_ratio: float = Field(ge=0, le=1)
    development_gold_document_count: int = Field(ge=0)
    development_gold_documents_indexed: int = Field(ge=0)
    development_gold_document_coverage_ratio: float = Field(ge=0, le=1)
    full_document_search_latency_ms: float = Field(ge=0)
    passage_search_latency_ms: float = Field(ge=0)
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[PassageIndexCase, ...]
    gates: tuple[PassageIndexGate, ...]
    next_action: Literal[
        "preregister_fresh_passage_retrieval_comparison",
        "freeze_passage_index_and_diagnose_nonlexical_retrieval",
    ]
    claim_boundary: str = Field(min_length=1)


def load_passage_index_registration(path: Path) -> PassageIndexRegistration:
    registration = PassageIndexRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            registration.prerequisites.lexical_rank_result,
            registration.prerequisites.stability_audit,
            registration.prerequisites.gold_slice,
        ),
    )
    _validate_relative_directory(root, registration.baseline_run_root)
    source_index = _validate_relative_directory(
        root, registration.source_document_index.path
    )
    _validate_index_files(source_index, registration.source_document_index.files)

    lexical = json.loads(
        (root / registration.prerequisites.lexical_rank_result.path).read_text(
            encoding="utf-8"
        )
    )
    if lexical.get("decision") != "index_representation_diagnosis_required":
        raise ValueError("passage-index prerequisite did not select index representation")
    if lexical.get("generated_query_top20_cases") != 0:
        raise ValueError("passage-index baseline top-20 prerequisite changed")
    lexical_ids = tuple(str(item["query_id"]) for item in lexical.get("items", ()))
    if lexical_ids != registration.query_ids:
        raise ValueError("passage-index query order differs from lexical-rank result")

    stability = json.loads(
        (root / registration.prerequisites.stability_audit.path).read_text(
            encoding="utf-8"
        )
    )
    expected_ids = {
        str(row["query_id"])
        for row in stability["cases"]
        if row["category"] == "persistent_retrieval_miss"
    }
    if set(registration.query_ids) != expected_ids:
        raise ValueError("passage-index cases differ from the persistent-miss cluster")
    return registration


def split_document_into_passages(
    docid: str,
    contents: str,
    chunking: PassageChunking,
) -> tuple[PassageRecord, ...]:
    if not docid:
        raise ValueError("passage source docid is blank")
    if not contents.strip():
        raise ValueError(f"passage source document is blank: {docid}")
    header, body = _split_frontmatter(contents)
    header_tokens = _tokenize(header)[: chunking.maximum_header_tokens]
    body_tokens = _tokenize(body)
    if not body_tokens:
        body_tokens = _tokenize(contents)
    step = chunking.passage_tokens - chunking.overlap_tokens
    records: list[PassageRecord] = []
    for start in range(0, len(body_tokens), step):
        body_chunk = body_tokens[start : start + chunking.passage_tokens]
        if not body_chunk:
            break
        if (
            len(body_chunk) < chunking.minimum_passage_tokens
            and records
            and start + len(body_chunk) >= len(body_tokens)
        ):
            break
        ordinal = len(records)
        passage_id = f"{docid}::p{ordinal:05d}"
        text_parts = []
        if header_tokens:
            text_parts.append(" ".join(header_tokens))
        text_parts.append(" ".join(body_chunk))
        records.append(
            PassageRecord(
                id=passage_id,
                parent_docid=docid,
                ordinal=ordinal,
                contents="\n".join(text_parts),
                body_token_count=len(body_chunk),
                header_token_count=len(header_tokens),
            )
        )
        if start + chunking.passage_tokens >= len(body_tokens):
            break
    if not records:
        raise ValueError(f"passage source document produced no passage: {docid}")
    return tuple(records)


def export_passage_corpus(
    *,
    documents: Iterable[tuple[str, str]],
    chunking: PassageChunking,
    output_path: Path,
) -> PassageExportStats:
    if output_path.exists():
        raise ValueError("passage corpus output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    corpus_digest = sha256()
    source_digest = sha256()
    passage_digest = sha256()
    seen_docids: set[str] = set()
    source_count = 0
    source_with_passages = 0
    passage_count = 0
    with output_path.open("wb") as handle:
        for docid, contents in documents:
            docid = str(docid)
            if docid in seen_docids:
                raise ValueError(f"duplicate passage source docid: {docid}")
            seen_docids.add(docid)
            source_count += 1
            source_digest.update(docid.encode("utf-8") + b"\n")
            passages = split_document_into_passages(docid, contents, chunking)
            source_with_passages += 1
            for passage in passages:
                line = (
                    json.dumps(
                        passage.collection_row(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                handle.write(line)
                corpus_digest.update(line)
                passage_digest.update(passage.id.encode("utf-8") + b"\n")
                passage_count += 1
    return PassageExportStats(
        source_document_count=source_count,
        source_documents_with_passages=source_with_passages,
        passage_count=passage_count,
        corpus_sha256=corpus_digest.hexdigest(),
        source_docids_sha256=source_digest.hexdigest(),
        passage_ids_sha256=passage_digest.hexdigest(),
        export_latency_ms=(perf_counter() - started) * 1000,
    )


def collapse_passage_hits(
    passage_ids: Sequence[str], *, maximum_documents: int
) -> tuple[str, ...]:
    parents: list[str] = []
    seen: set[str] = set()
    for passage_id in passage_ids:
        match = re.fullmatch(r"(.+)::p(\d{5})", str(passage_id))
        if match is None:
            raise ValueError(f"invalid passage id returned by index: {passage_id}")
        parent = match.group(1)
        if parent in seen:
            continue
        seen.add(parent)
        parents.append(parent)
        if len(parents) >= maximum_documents:
            break
    return tuple(parents)


def load_passage_export_manifest(
    *, registration_path: Path, manifest_path: Path
) -> PassageExportManifest:
    registration = load_passage_index_registration(registration_path)
    root = registration_path.resolve().parents[2]
    manifest = PassageExportManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.registration_sha256 != sha256_file(registration_path):
        raise ValueError("passage export registration hash changed")
    if manifest.chunking != registration.chunking:
        raise ValueError("passage export chunking differs from registration")
    expected_corpus = (
        Path(registration.passage_corpus_path) / "collection.jsonl"
    ).as_posix()
    if manifest.corpus_file != expected_corpus:
        raise ValueError("passage export corpus path differs from registration")
    corpus_path = (root / manifest.corpus_file).resolve()
    if not corpus_path.is_relative_to(root) or not corpus_path.is_file():
        raise ValueError("passage export corpus is missing or escapes root")
    if sha256_file(corpus_path) != manifest.corpus_sha256:
        raise ValueError("passage export corpus hash changed")
    if manifest.source_document_count != registration.source_document_index.document_count:
        raise ValueError("passage export source-document count differs from registration")
    if manifest.source_documents_with_passages != manifest.source_document_count:
        raise ValueError("passage export omitted a source document")
    return manifest


def load_passage_build_manifest(
    *,
    registration_path: Path,
    export_manifest_path: Path,
    build_manifest_path: Path,
) -> PassageIndexBuildManifest:
    registration = load_passage_index_registration(registration_path)
    root = registration_path.resolve().parents[2]
    export = load_passage_export_manifest(
        registration_path=registration_path, manifest_path=export_manifest_path
    )
    manifest = PassageIndexBuildManifest.model_validate_json(
        build_manifest_path.read_text(encoding="utf-8")
    )
    if manifest.registration_sha256 != sha256_file(registration_path):
        raise ValueError("passage build registration hash changed")
    expected_export = ArtifactReference(
        path=export_manifest_path.resolve().relative_to(root).as_posix(),
        sha256=sha256_file(export_manifest_path),
    )
    if manifest.export_manifest != expected_export:
        raise ValueError("passage build export manifest changed")
    if manifest.index_path != registration.passage_index_path:
        raise ValueError("passage build index path differs from registration")
    if manifest.index_document_count != export.passage_count:
        raise ValueError("passage build document count differs from export")
    index_path = _validate_relative_directory(root, manifest.index_path)
    _validate_index_files(index_path, manifest.index_files)
    return manifest


def run_passage_index_audit(
    *,
    registration_path: Path,
    export_manifest_path: Path,
    build_manifest_path: Path,
    output_path: Path,
    full_document_search: DocumentSearch,
    passage_search: DocumentSearch,
    passage_document_exists: DocumentExists,
) -> PassageIndexResult:
    registration = load_passage_index_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("passage-index output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("passage-index result already exists")
    export = load_passage_export_manifest(
        registration_path=registration_path, manifest_path=export_manifest_path
    )
    build = load_passage_build_manifest(
        registration_path=registration_path,
        export_manifest_path=export_manifest_path,
        build_manifest_path=build_manifest_path,
    )
    gold = json.loads(
        (root / registration.prerequisites.gold_slice.path).read_text(encoding="utf-8")
    )
    gold_by_id = {str(row["query_id"]): row for row in gold["rows"]}
    development_gold_docids = sorted(
        {
            str(docid)
            for row in gold["rows"]
            for docid in row.get("gold_docids", ())
        }
    )
    indexed_gold = sum(
        passage_document_exists(f"{docid}::p00000")
        for docid in development_gold_docids
    )

    items: list[PassageIndexCase] = []
    full_latency = 0.0
    passage_latency = 0.0
    generated_query_count = 0
    for query_id in registration.query_ids:
        gold_row = gold_by_id.get(query_id)
        if gold_row is None:
            raise ValueError(f"passage-index gold case is missing: {query_id}")
        gold_docids = tuple(str(value) for value in gold_row["gold_docids"])
        run_path = root / registration.baseline_run_root / query_id / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        search_calls = run.get("search_calls")
        if not isinstance(search_calls, list) or not search_calls:
            raise ValueError(f"passage-index baseline run has no searches: {query_id}")
        observations: list[PassageQueryObservation] = []
        for call in search_calls:
            if not isinstance(call, dict) or call.get("outcome") != "ok":
                raise ValueError(f"passage-index baseline run has failed search: {query_id}")
            query = call.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"passage-index baseline run has blank query: {query_id}")
            full_started = perf_counter()
            full_hits = tuple(
                str(value)
                for value in full_document_search(
                    query, registration.retrieval.maximum_unique_documents_per_query
                )
            )
            full_elapsed = (perf_counter() - full_started) * 1000
            if len(full_hits) > registration.retrieval.maximum_unique_documents_per_query:
                raise ValueError("full-document search exceeded registered candidate cap")
            if len(full_hits) != len(set(full_hits)):
                raise ValueError("full-document search returned duplicate documents")
            passage_started = perf_counter()
            passage_hits = tuple(
                str(value)
                for value in passage_search(
                    query, registration.retrieval.maximum_passage_hits_per_query
                )
            )
            passage_elapsed = (perf_counter() - passage_started) * 1000
            if len(passage_hits) > registration.retrieval.maximum_passage_hits_per_query:
                raise ValueError("passage search exceeded registered internal hit cap")
            passage_parents = collapse_passage_hits(
                passage_hits,
                maximum_documents=(
                    registration.retrieval.maximum_unique_documents_per_query
                ),
            )
            gold_set = set(gold_docids)
            observations.append(
                PassageQueryObservation(
                    query=query,
                    query_sha256=_text_sha256(query),
                    full_document_top20=full_hits,
                    passage_top20_parent_documents=passage_parents,
                    passage_hits_examined=len(passage_hits),
                    full_document_gold_hit=bool(gold_set.intersection(full_hits)),
                    passage_gold_hit=bool(gold_set.intersection(passage_parents)),
                    full_document_latency_ms=full_elapsed,
                    passage_latency_ms=passage_elapsed,
                )
            )
            full_latency += full_elapsed
            passage_latency += passage_elapsed
            generated_query_count += 1
        items.append(
            PassageIndexCase(
                query_id=query_id,
                gold_docids=gold_docids,
                queries=tuple(observations),
                full_document_gold_hit=any(
                    item.full_document_gold_hit for item in observations
                ),
                passage_gold_hit=any(item.passage_gold_hit for item in observations),
            )
        )

    full_hits = sum(item.full_document_gold_hit for item in items)
    passage_hits = sum(item.passage_gold_hit for item in items)
    source_ratio = (
        export.source_documents_with_passages / export.source_document_count
        if export.source_document_count
        else 0.0
    )
    development_ratio = (
        indexed_gold / len(development_gold_docids)
        if development_gold_docids
        else 0.0
    )
    lexical = json.loads(
        (root / registration.prerequisites.lexical_rank_result.path).read_text(
            encoding="utf-8"
        )
    )
    gates = (
        PassageIndexGate(
            gate_id="full_document_top20_reproduction",
            observed=float(full_hits),
            operator="eq",
            threshold=float(lexical["generated_query_top20_cases"]),
            passed=full_hits == lexical["generated_query_top20_cases"],
        ),
        PassageIndexGate(
            gate_id="passage_generated_query_gold_doc_recall_at20_cases",
            observed=float(passage_hits),
            operator="ge",
            threshold=float(
                registration.acceptance.minimum_passage_generated_query_gold_doc_recall_at20_cases
            ),
            passed=(
                passage_hits
                >= registration.acceptance.minimum_passage_generated_query_gold_doc_recall_at20_cases
            ),
        ),
        PassageIndexGate(
            gate_id="source_document_coverage_ratio",
            observed=source_ratio,
            operator="eq",
            threshold=registration.acceptance.required_source_document_coverage_ratio,
            passed=(
                source_ratio
                == registration.acceptance.required_source_document_coverage_ratio
            ),
        ),
        PassageIndexGate(
            gate_id="development_gold_document_coverage_ratio",
            observed=development_ratio,
            operator="eq",
            threshold=(
                registration.acceptance.required_development_gold_document_coverage_ratio
            ),
            passed=(
                development_ratio
                == registration.acceptance.required_development_gold_document_coverage_ratio
            ),
        ),
        PassageIndexGate(
            gate_id="passage_index_document_count",
            observed=float(build.index_document_count),
            operator="eq",
            threshold=float(export.passage_count),
            passed=build.index_document_count == export.passage_count,
        ),
    )
    passed = all(gate.passed for gate in gates)
    result = PassageIndexResult(
        created_at=_utc_now(),
        decision=(
            "passage_index_candidate" if passed else "freeze_passage_index_branch"
        ),
        registration_sha256=sha256_file(registration_path),
        export_manifest=ArtifactReference(
            path=export_manifest_path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(export_manifest_path),
        ),
        build_manifest=ArtifactReference(
            path=build_manifest_path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(build_manifest_path),
        ),
        query_count=len(items),
        generated_query_count=generated_query_count,
        full_document_gold_hit_cases_at20=full_hits,
        passage_gold_hit_cases_at20=passage_hits,
        passage_minus_full_document_hit_cases=passage_hits - full_hits,
        passage_wins=sum(
            item.passage_gold_hit and not item.full_document_gold_hit for item in items
        ),
        passage_losses=sum(
            item.full_document_gold_hit and not item.passage_gold_hit for item in items
        ),
        source_document_count=export.source_document_count,
        source_documents_with_passages=export.source_documents_with_passages,
        passage_count=export.passage_count,
        passage_index_document_count=build.index_document_count,
        source_document_coverage_ratio=source_ratio,
        development_gold_document_count=len(development_gold_docids),
        development_gold_documents_indexed=indexed_gold,
        development_gold_document_coverage_ratio=development_ratio,
        full_document_search_latency_ms=full_latency,
        passage_search_latency_ms=passage_latency,
        items=tuple(items),
        gates=gates,
        next_action=(
            "preregister_fresh_passage_retrieval_comparison"
            if passed
            else "freeze_passage_index_and_diagnose_nonlexical_retrieval"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def index_file_digests(index_path: Path) -> tuple[IndexFileDigest, ...]:
    files = [
        path
        for path in index_path.iterdir()
        if path.is_file() and path.name != "write.lock"
    ]
    return tuple(
        IndexFileDigest(
            name=path.name,
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(files, key=lambda item: item.name)
    )


def _split_frontmatter(contents: str) -> tuple[str, str]:
    lines = contents.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", contents
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return "", contents


def _tokenize(value: str) -> list[str]:
    return value.split()


def _validate_artifacts(root: Path, artifacts: Sequence[ArtifactReference]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")


def _validate_relative_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"registered directory is missing or escapes root: {relative}")
    return path


def _validate_index_files(
    index_path: Path, expected: Sequence[IndexFileDigest]
) -> None:
    actual_names = {
        path.name
        for path in index_path.iterdir()
        if path.is_file() and path.name != "write.lock"
    }
    expected_names = {item.name for item in expected}
    if actual_names != expected_names:
        raise ValueError("registered index file set changed")
    for item in expected:
        path = index_path / item.name
        if path.stat().st_size != item.bytes or sha256_file(path) != item.sha256:
            raise ValueError(f"registered index file changed: {item.name}")
