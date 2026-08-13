from __future__ import annotations

import pickle
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DevelopmentGoldSlice
from .browsecomp_plus import (
    load_pi_browsecomp_run,
    normalized_text_file_sha256,
)
from .pi_browsecomp import PiSmokeSummary


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PinnedModel(StrictContract):
    name: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_file: str = Field(min_length=1)
    model_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PinnedIndexShard(StrictContract):
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PinnedDenseIndex(StrictContract):
    dataset: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    subdirectory: str = Field(min_length=1)
    shards: list[PinnedIndexShard] = Field(min_length=1)

    @model_validator(mode="after")
    def shard_names_are_unique(self) -> "PinnedDenseIndex":
        names = [shard.filename for shard in self.shards]
        if len(names) != len(set(names)):
            raise ValueError("dense index shard filenames must be unique")
        return self


class DenseRetrieverCandidate(StrictContract):
    candidate_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    kind: Literal["dense_faiss"] = "dense_faiss"
    model: PinnedModel
    index: PinnedDenseIndex
    query_prefix: str = Field(min_length=1)
    pooling: Literal["eos"] = "eos"
    normalize: Literal[True] = True
    max_length: int = Field(ge=1, le=8192)
    top_k: int = Field(ge=1, le=100)


class ReplayPolicy(StrictContract):
    source_queries: Literal["frozen_agent_search_calls"]
    baseline: Literal["stored_bm25_top5"]
    candidate_depth: int = Field(ge=1, le=100)
    fusion: Literal["reciprocal_rank_fusion"]
    fusion_k: int = Field(ge=1, le=1000)
    fused_depth: int = Field(ge=1, le=100)
    official_metric_status: Literal["diagnostic_not_official"]


class RetrieverCandidatesManifest(StrictContract):
    schema_version: Literal["browsecomp-plus-retriever-candidates-v0"]
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: list[DenseRetrieverCandidate] = Field(min_length=1)
    replay: ReplayPolicy

    @model_validator(mode="after")
    def candidate_ids_are_unique(self) -> "RetrieverCandidatesManifest":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("retriever candidate IDs must be unique")
        if any(candidate.top_k != self.replay.candidate_depth for candidate in self.candidates):
            raise ValueError("candidate top_k must match replay candidate_depth")
        return self


class RankedHit(StrictContract):
    docid: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)


class DenseRuntimeSnapshot(StrictContract):
    device: Literal["cpu", "cuda"]
    torch_version: str
    transformers_version: str
    faiss_version: str
    tevatron_version: str
    model_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    local_model_files_sha256: dict[str, str]
    index_shards_sha256: dict[str, str]
    index_documents: int = Field(gt=0)
    embedding_dimensions: int = Field(gt=0)
    load_latency_ms: int = Field(ge=0)
    search_latency_ms: int = Field(ge=0)


class RetrievalMetrics(StrictContract):
    retrieved_unique: int = Field(ge=0)
    evidence_recall: float | None = Field(default=None, ge=0, le=1)
    gold_recall: float | None = Field(default=None, ge=0, le=1)


class ReplayCall(StrictContract):
    call_index: int = Field(ge=1)
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_outcome: Literal["ok", "error"]
    baseline_docids: list[str]
    candidate_docids: list[str]
    fused_docids: list[str]


class RetrievalReplayRow(StrictContract):
    query_id: str = Field(min_length=1)
    source_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_calls: int = Field(ge=0)
    unique_search_queries: int = Field(ge=0)
    baseline: RetrievalMetrics
    candidate: RetrievalMetrics
    fused: RetrievalMetrics
    calls: list[ReplayCall]


class RetrievalReplaySummary(StrictContract):
    schema_version: Literal["browsecomp-plus-retrieval-replay-v0"] = (
        "browsecomp-plus-retrieval-replay-v0"
    )
    created_at: str
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    official_accuracy_status: Literal["planned_not_run"] = "planned_not_run"
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_slice_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retriever_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    source_queries: Literal["frozen_agent_search_calls"]
    query_count: int = Field(gt=0)
    search_calls: int = Field(ge=0)
    unique_search_queries: int = Field(ge=0)
    baseline_evidence_recall_percent: float | None = Field(default=None, ge=0, le=100)
    candidate_evidence_recall_percent: float | None = Field(default=None, ge=0, le=100)
    fused_evidence_recall_percent: float | None = Field(default=None, ge=0, le=100)
    baseline_gold_recall_percent: float | None = Field(default=None, ge=0, le=100)
    candidate_gold_recall_percent: float | None = Field(default=None, ge=0, le=100)
    fused_gold_recall_percent: float | None = Field(default=None, ge=0, le=100)
    evidence_recall_delta_candidate_pp: float | None = None
    evidence_recall_delta_fused_pp: float | None = None
    runtime: DenseRuntimeSnapshot
    rows: list[RetrievalReplayRow] = Field(min_length=1)


def load_retriever_candidates(
    path: Path, *, target_manifest_path: Path
) -> RetrieverCandidatesManifest:
    manifest = RetrieverCandidatesManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.target_manifest_sha256 != normalized_text_file_sha256(target_manifest_path):
        raise ValueError("retriever candidates target a different benchmark manifest")
    return manifest


def select_candidate(
    manifest: RetrieverCandidatesManifest, candidate_id: str
) -> DenseRetrieverCandidate:
    matches = [candidate for candidate in manifest.candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"unknown retriever candidate: {candidate_id}")
    return matches[0]


def collect_frozen_search_queries(source_dir: Path) -> list[str]:
    summary = PiSmokeSummary.model_validate_json(
        (source_dir / "summary.json").read_text(encoding="utf-8")
    )
    queries: list[str] = []
    for item in summary.items:
        if item.run_sha256 is None:
            raise ValueError(f"missing frozen run for query {item.query_id}")
        run_path = source_dir / _safe_id(item.query_id) / "run.json"
        if sha256(run_path.read_bytes()).hexdigest() != item.run_sha256:
            raise ValueError(f"source run hash mismatch for query {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        queries.extend(call.query for call in run.search_calls)
    return list(dict.fromkeys(queries))


def score_retrieval_replay(
    *,
    source_dir: Path,
    gold_slice_path: Path,
    retriever_manifest_path: Path,
    target_manifest_path: Path,
    candidate_id: str,
    candidate_results: Mapping[str, Sequence[RankedHit]],
    runtime: DenseRuntimeSnapshot,
    output_path: Path,
) -> RetrievalReplaySummary:
    repository_root = _find_repository_root(source_dir.resolve())
    _require_under_runs(output_path, repository_root)
    if output_path.exists():
        raise ValueError("retrieval replay output already exists")

    summary_path = source_dir / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    gold_bytes = gold_slice_path.read_bytes()
    gold = DevelopmentGoldSlice.model_validate_json(gold_bytes)
    if gold.source_summary_sha256 != sha256(summary_bytes).hexdigest():
        raise ValueError("development gold was not opened for this prediction summary")

    manifest = load_retriever_candidates(
        retriever_manifest_path, target_manifest_path=target_manifest_path
    )
    candidate = select_candidate(manifest, candidate_id)
    if summary.target_manifest_sha256 != manifest.target_manifest_sha256:
        raise ValueError("source predictions target a different benchmark manifest")
    if runtime.model_file_sha256 != candidate.model.model_file_sha256:
        raise ValueError("dense runtime model hash does not match candidate manifest")
    expected_shards = {shard.filename: shard.sha256 for shard in candidate.index.shards}
    if runtime.index_shards_sha256 != expected_shards:
        raise ValueError("dense runtime index hashes do not match candidate manifest")

    expected_queries = collect_frozen_search_queries(source_dir)
    missing = [query for query in expected_queries if query not in candidate_results]
    extras = [query for query in candidate_results if query not in set(expected_queries)]
    if missing or extras:
        raise ValueError(
            f"candidate result keys differ from frozen queries: missing={len(missing)}, extras={len(extras)}"
        )
    for query, hits in candidate_results.items():
        if len(hits) > manifest.replay.candidate_depth:
            raise ValueError(f"candidate returned too many hits for query: {query}")
        if len({hit.docid for hit in hits}) != len(hits):
            raise ValueError(f"candidate returned duplicate docids for query: {query}")

    gold_by_id = {row.query_id: row for row in gold.rows}
    rows: list[RetrievalReplayRow] = []
    for item in summary.items:
        reference = gold_by_id.get(item.query_id)
        if reference is None or item.run_sha256 is None:
            raise ValueError(f"missing frozen run or development gold for query {item.query_id}")
        run_path = source_dir / _safe_id(item.query_id) / "run.json"
        run = load_pi_browsecomp_run(run_path)
        baseline_retrieved: set[str] = set()
        candidate_retrieved: set[str] = set()
        fused_retrieved: set[str] = set()
        calls: list[ReplayCall] = []
        for index, call in enumerate(run.search_calls, start=1):
            baseline_docids = [result.docid for result in call.results]
            candidate_docids = [hit.docid for hit in candidate_results[call.query]]
            fused_docids = reciprocal_rank_fusion(
                baseline_docids,
                candidate_docids,
                k=manifest.replay.fusion_k,
                depth=manifest.replay.fused_depth,
            )
            baseline_retrieved.update(baseline_docids)
            candidate_retrieved.update(candidate_docids)
            fused_retrieved.update(fused_docids)
            calls.append(
                ReplayCall(
                    call_index=index,
                    query=call.query,
                    query_sha256=sha256(call.query.encode("utf-8")).hexdigest(),
                    source_outcome=call.outcome,
                    baseline_docids=baseline_docids,
                    candidate_docids=candidate_docids,
                    fused_docids=fused_docids,
                )
            )
        evidence = set(reference.evidence_docids)
        gold_docids = set(reference.gold_docids)
        rows.append(
            RetrievalReplayRow(
                query_id=item.query_id,
                source_run_sha256=item.run_sha256,
                search_calls=len(run.search_calls),
                unique_search_queries=len({call.query for call in run.search_calls}),
                baseline=_metrics(baseline_retrieved, evidence, gold_docids),
                candidate=_metrics(candidate_retrieved, evidence, gold_docids),
                fused=_metrics(fused_retrieved, evidence, gold_docids),
                calls=calls,
            )
        )

    baseline_evidence = _macro_percent(rows, "baseline", "evidence_recall")
    candidate_evidence = _macro_percent(rows, "candidate", "evidence_recall")
    fused_evidence = _macro_percent(rows, "fused", "evidence_recall")
    artifact = RetrievalReplaySummary(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        gold_slice_sha256=sha256(gold_bytes).hexdigest(),
        retriever_manifest_sha256=normalized_text_file_sha256(
            retriever_manifest_path
        ),
        candidate_id=candidate_id,
        source_queries=manifest.replay.source_queries,
        query_count=len(rows),
        search_calls=sum(row.search_calls for row in rows),
        unique_search_queries=len(expected_queries),
        baseline_evidence_recall_percent=baseline_evidence,
        candidate_evidence_recall_percent=candidate_evidence,
        fused_evidence_recall_percent=fused_evidence,
        baseline_gold_recall_percent=_macro_percent(rows, "baseline", "gold_recall"),
        candidate_gold_recall_percent=_macro_percent(rows, "candidate", "gold_recall"),
        fused_gold_recall_percent=_macro_percent(rows, "fused", "gold_recall"),
        evidence_recall_delta_candidate_pp=_delta(candidate_evidence, baseline_evidence),
        evidence_recall_delta_fused_pp=_delta(fused_evidence, baseline_evidence),
        runtime=runtime,
        rows=rows,
    )
    _atomic_write(output_path, artifact.model_dump_json(indent=2))
    return artifact


class QwenDenseReplaySearcher:
    def __init__(
        self,
        *,
        candidate: DenseRetrieverCandidate,
        model_dir: Path,
        index_root: Path,
        batch_size: int = 8,
    ) -> None:
        started = time.perf_counter()
        try:
            import faiss
            import numpy as np
            import tevatron
            import torch
            import transformers
            from tevatron.retriever.modeling import DenseModel
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                'install dense replay dependencies with pip install -e ".[browsecomp-plus-dense]"'
            ) from error

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        model_dir = model_dir.resolve()
        index_dir = (index_root / candidate.index.subdirectory).resolve()
        model_file = model_dir / candidate.model.model_file
        model_hash = _file_sha256(model_file)
        if model_hash != candidate.model.model_file_sha256:
            raise ValueError("local dense model file hash does not match the pinned candidate")

        shard_hashes: dict[str, str] = {}
        index = None
        lookup: list[str] = []
        dimensions = 0
        for shard in candidate.index.shards:
            shard_path = index_dir / shard.filename
            actual_hash = _file_sha256(shard_path)
            if actual_hash != shard.sha256:
                raise ValueError(f"dense index shard hash mismatch: {shard.filename}")
            shard_hashes[shard.filename] = actual_hash
            with shard_path.open("rb") as handle:
                representations, identifiers = pickle.load(handle)
            representations = np.asarray(representations, dtype="float32")
            if representations.ndim != 2 or representations.shape[0] != len(identifiers):
                raise ValueError(f"invalid dense index shard shape: {shard.filename}")
            if index is None:
                dimensions = int(representations.shape[1])
                index = faiss.IndexFlatIP(dimensions)
            elif representations.shape[1] != dimensions:
                raise ValueError("dense index shard dimensions do not match")
            index.add(representations)
            lookup.extend(str(identifier) for identifier in identifiers)

        if index is None or not lookup:
            raise ValueError("dense index is empty")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side="left",
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = DenseModel.load(
            str(model_dir),
            pooling=candidate.pooling,
            normalize=candidate.normalize,
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = model.to(device)
        model.eval()

        self._candidate = candidate
        self._batch_size = batch_size
        self._device = device
        self._faiss_index = index
        self._lookup = lookup
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._np = np
        self._search_latency_ms = 0
        self._snapshot = DenseRuntimeSnapshot(
            device=device,
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            faiss_version=faiss.__version__,
            tevatron_version=getattr(tevatron, "__version__", "0.0.1"),
            model_file_sha256=model_hash,
            local_model_files_sha256=_model_support_file_hashes(model_dir),
            index_shards_sha256=shard_hashes,
            index_documents=len(lookup),
            embedding_dimensions=dimensions,
            load_latency_ms=round((time.perf_counter() - started) * 1000),
            search_latency_ms=0,
        )

    def search_many(self, queries: Sequence[str]) -> dict[str, list[RankedHit]]:
        unique_queries = list(dict.fromkeys(queries))
        if not unique_queries:
            return {}
        started = time.perf_counter()
        output: dict[str, list[RankedHit]] = {}
        for offset in range(0, len(unique_queries), self._batch_size):
            query_batch = unique_queries[offset : offset + self._batch_size]
            encoded = self._tokenizer(
                [self._candidate.query_prefix + query for query in query_batch],
                padding=True,
                truncation=True,
                max_length=self._candidate.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
            with self._torch.no_grad():
                representations = self._model.encode_query(encoded)
            query_vectors = representations.detach().float().cpu().numpy()
            query_vectors = self._np.asarray(query_vectors, dtype="float32")
            scores, indices = self._faiss_index.search(
                query_vectors, self._candidate.top_k
            )
            for query, query_scores, query_indices in zip(
                query_batch, scores, indices, strict=True
            ):
                output[query] = [
                    RankedHit(docid=self._lookup[int(index)], score=float(score))
                    for score, index in zip(query_scores, query_indices, strict=True)
                    if int(index) >= 0
                ]
        self._search_latency_ms = round((time.perf_counter() - started) * 1000)
        return output

    @property
    def runtime_snapshot(self) -> DenseRuntimeSnapshot:
        return self._snapshot.model_copy(
            update={"search_latency_ms": self._search_latency_ms}
        )


def reciprocal_rank_fusion(
    baseline_docids: Sequence[str],
    candidate_docids: Sequence[str],
    *,
    k: int,
    depth: int,
) -> list[str]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in (baseline_docids, candidate_docids):
        for rank, docid in enumerate(ranking, start=1):
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank)
            best_rank[docid] = min(best_rank.get(docid, rank), rank)
    ranked = sorted(scores, key=lambda docid: (-scores[docid], best_rank[docid], docid))
    return ranked[:depth]


def _metrics(
    retrieved: set[str], evidence: set[str], gold_docids: set[str]
) -> RetrievalMetrics:
    return RetrievalMetrics(
        retrieved_unique=len(retrieved),
        evidence_recall=_recall(retrieved, evidence),
        gold_recall=_recall(retrieved, gold_docids),
    )


def _recall(retrieved: set[str], relevant: set[str]) -> float | None:
    return len(retrieved & relevant) / len(relevant) if relevant else None


def _macro_percent(
    rows: Sequence[RetrievalReplayRow], variant: str, metric: str
) -> float | None:
    values = [
        getattr(getattr(row, variant), metric)
        for row in rows
        if getattr(getattr(row, variant), metric) is not None
    ]
    return round(sum(values) / len(values) * 100, 2) if values else None


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    return round(candidate - baseline, 2) if candidate is not None and baseline is not None else None


def _model_support_file_hashes(model_dir: Path) -> dict[str, str]:
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    return {name: _file_sha256(model_dir / name) for name in names}


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required replay asset is missing: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("retrieval replay artifacts must remain under ignored runs/")


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate repository root")


def _safe_id(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    if not safe or safe in {".", ".."}:
        raise ValueError("query ID cannot be represented safely")
    return safe


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
