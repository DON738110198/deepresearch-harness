from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .browsecomp_plus import PiBrowseCompRun
from .pi_browsecomp import PiSmokeSummary
from .progressive_disclosure import (
    EvidenceCandidate,
    ProgressiveDisclosurePolicy,
    ProgressiveDisclosureSession,
    format_bm25_anchor,
    format_dense_lead,
)
from .progressive_disclosure_server import HuggingFaceEvidenceTokenizer


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalibrationRow(StrictContract):
    trial_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    search_calls: int = Field(ge=0)
    baseline_evidence_recall: float = Field(ge=0, le=1)
    candidate_evidence_recall: float = Field(ge=0, le=1)
    baseline_ingress_tokens: int = Field(ge=0)
    candidate_search_ingress_tokens: int = Field(ge=0)
    candidate_search_budget_exhausted: bool
    baseline_unique_docids: int = Field(ge=0)
    candidate_unique_docids: int = Field(ge=0)


class CalibrationMetrics(StrictContract):
    query_count: int = Field(gt=0)
    search_calls: int = Field(ge=0)
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    candidate_evidence_recall_percent: float = Field(ge=0, le=100)
    evidence_recall_delta_pp: float
    baseline_mean_ingress_tokens: float = Field(ge=0)
    candidate_mean_search_ingress_tokens: float = Field(ge=0)
    aggregate_search_ingress_ratio: float = Field(ge=0)
    baseline_p90_ingress_tokens: int = Field(ge=0)
    candidate_p90_search_ingress_tokens: int = Field(ge=0)
    p90_search_ingress_ratio: float = Field(ge=0)
    search_budget_exhausted_queries: int = Field(ge=0)


class ProgressiveDisclosureCalibration(StrictContract):
    schema_version: str = "browsecomp-plus-progressive-disclosure-calibration-v0"
    created_at: str
    status: str
    provider_calls: int = 0
    sealed_holdout_accessed: bool = False
    registration_path: str
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_validation_failures: int = Field(ge=0)
    ranking_mismatches: int = Field(ge=0)
    selected_policy: ProgressiveDisclosurePolicy
    selection: dict[str, Any]
    metrics: CalibrationMetrics
    rows: list[CalibrationRow] = Field(min_length=1)
    claim_boundary: str


class PreparedRun:
    def __init__(
        self,
        *,
        trial_id: str,
        query_id: str,
        calls: list[dict[str, Any]],
        evidence_docids: set[str],
        baseline_ingress_tokens: int,
    ) -> None:
        self.trial_id = trial_id
        self.query_id = query_id
        self.calls = calls
        self.evidence_docids = evidence_docids
        self.baseline_ingress_tokens = baseline_ingress_tokens


class CachedEvidenceTokenizer:
    def __init__(self, tokenizer: HuggingFaceEvidenceTokenizer) -> None:
        self._tokenizer = tokenizer
        self._cache: dict[str, tuple[object, ...]] = {}

    def encode(self, text: str) -> Sequence[object]:
        key = sha256(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            cached = tuple(self._tokenizer.encode(text))
            self._cache[key] = cached
        return cached

    def decode(self, tokens: Sequence[object]) -> str:
        return self._tokenizer.decode(tokens)


def calibrate_progressive_disclosure(
    *,
    registration_path: Path,
    output_path: Path,
    document_index_path: Path,
    snippet_tokenizer_dir: Path,
) -> ProgressiveDisclosureCalibration:
    repository_root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("calibration output must be stored under ignored runs/")
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    if registration.get("status") != "registered_before_calibration_execution":
        raise ValueError("calibration registration is not frozen")
    _validate_registered_sources(repository_root, registration)

    try:
        from pyserini.search.lucene import LuceneSearcher
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install BrowseComp-Plus calibration dependencies") from error

    tokenizer = AutoTokenizer.from_pretrained(
        snippet_tokenizer_dir.resolve(),
        local_files_only=True,
        trust_remote_code=False,
    )
    evidence_tokenizer = CachedEvidenceTokenizer(
        HuggingFaceEvidenceTokenizer(
            tokenizer,
            maximum_tokens=max(
                registration["fixed_candidate"]["anchor_token_cap"],
                registration["fixed_candidate"]["lead_token_cap"],
                registration["fixed_candidate"]["open_token_cap"],
            ),
        )
    )
    documents = LuceneSearcher(str(document_index_path.resolve()))
    document_cache: dict[str, str] = {}

    def load_document(docid: str) -> str:
        cached = document_cache.get(docid)
        if cached is not None:
            return cached
        document = documents.doc(docid)
        if document is None:
            raise ValueError(f"calibration docid is absent from document store: {docid}")
        raw = json.loads(document.raw())
        contents = raw.get("contents")
        if not isinstance(contents, str) or not contents.strip():
            raise ValueError(f"calibration document has no contents: {docid}")
        document_cache[docid] = contents
        return contents

    rankings_path = repository_root / registration["sources"]["rankings"]["path"]
    rankings_payload = json.loads(rankings_path.read_bytes())
    rankings = {row["query_sha256"]: row for row in rankings_payload["rankings"]}
    prepared, ranking_mismatches = _prepare_runs(
        repository_root=repository_root,
        registration=registration,
        rankings=rankings,
        tokenizer=tokenizer,
    )
    fixed = registration["fixed_candidate"]
    open_budget = fixed["open_evidence_ingress_token_budget"]
    unbounded_policy = _make_policy(
        fixed=fixed,
        search_budget=100_000 - open_budget,
    )
    unbounded_rows = [
        _simulate_run(
            row,
            policy=unbounded_policy,
            tokenizer=evidence_tokenizer,
            load_document=load_document,
        )
        for row in prepared
    ]
    baseline_p90 = _nearest_rank_percentile(
        [row.baseline_ingress_tokens for row in unbounded_rows], 0.90
    )
    candidate_p90 = _nearest_rank_percentile(
        [row.candidate_search_ingress_tokens for row in unbounded_rows], 0.90
    )
    search_budget = _round_up(
        min(baseline_p90, candidate_p90),
        registration["selection_rule"]["budget_rounding_tokens"],
    )
    search_budget = max(search_budget, fixed["anchor_token_cap"])
    selected_policy = _make_policy(fixed=fixed, search_budget=search_budget)
    rows = []
    for source, unbounded in zip(prepared, unbounded_rows, strict=True):
        if unbounded.candidate_search_ingress_tokens <= search_budget:
            rows.append(unbounded)
        else:
            rows.append(
                _simulate_run(
                    source,
                    policy=selected_policy,
                    tokenizer=evidence_tokenizer,
                    load_document=load_document,
                )
            )
    metrics = _aggregate(rows)
    acceptance = registration["acceptance"]
    passed = (
        ranking_mismatches == 0
        and metrics.evidence_recall_delta_pp
        >= acceptance["minimum_evidence_recall_delta_pp"]
        and metrics.aggregate_search_ingress_ratio
        <= acceptance["maximum_aggregate_search_ingress_ratio"]
        and metrics.p90_search_ingress_ratio
        <= acceptance["maximum_p90_search_ingress_ratio"]
    )
    artifact = ProgressiveDisclosureCalibration(
        created_at=datetime.now(timezone.utc).isoformat(),
        status="calibration_passed" if passed else "calibration_rejected",
        registration_path=(
            registration_path.resolve().relative_to(repository_root).as_posix()
        ),
        registration_sha256=sha256(registration_bytes).hexdigest(),
        source_validation_failures=0,
        ranking_mismatches=ranking_mismatches,
        selected_policy=selected_policy,
        selection={
            "rule": registration["selection_rule"],
            "unbounded_candidate_p90_search_ingress_tokens": candidate_p90,
            "baseline_p90_ingress_tokens": baseline_p90,
            "selected_search_ingress_token_budget": search_budget,
            "selected_total_evidence_ingress_token_budget": (
                selected_policy.total_evidence_ingress_token_budget
            ),
        },
        metrics=metrics,
        rows=rows,
        claim_boundary=(
            "Offline development calibration over previously saved model queries and "
            "retrieval rankings. It makes no provider calls, does not evaluate answer "
            "quality, and cannot support a model-capability or leaderboard claim."
        ),
    )
    _atomic_write(output_path, artifact.model_dump_json(indent=2))
    return artifact


def _validate_registered_sources(repository_root: Path, registration: dict[str, Any]) -> None:
    sources: list[dict[str, str]] = [
        registration["sources"]["rankings"],
        registration["sources"]["target_manifest"],
        registration["sources"]["retriever_manifest"],
        *registration["sources"]["implementation"],
    ]
    for trial in registration["sources"]["trials"]:
        sources.extend([trial["summary"], trial["gold"]])
    for source in sources:
        path = (repository_root / source["path"]).resolve()
        if not path.is_relative_to(repository_root.resolve()) or not path.is_file():
            raise ValueError(f"registered calibration source is missing: {source['path']}")
        if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"registered calibration source hash changed: {source['path']}")


def _prepare_runs(
    *,
    repository_root: Path,
    registration: dict[str, Any],
    rankings: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[PreparedRun], int]:
    prepared: list[PreparedRun] = []
    mismatches = 0
    for trial in registration["sources"]["trials"]:
        summary_path = repository_root / trial["summary"]["path"]
        summary = PiSmokeSummary.model_validate_json(summary_path.read_bytes())
        source_dir = summary_path.parent
        gold = json.loads((repository_root / trial["gold"]["path"]).read_bytes())
        gold_by_id = {row["query_id"]: row for row in gold["rows"]}
        for item in summary.items:
            if item.run_sha256 is None:
                raise ValueError("calibration requires successful saved runs")
            run_path = source_dir / item.query_id / "run.json"
            run = PiBrowseCompRun.model_validate_json(run_path.read_bytes())
            gold_row = gold_by_id[item.query_id]
            calls: list[dict[str, Any]] = []
            baseline_tokens = 0
            for call in run.search_calls:
                if call.outcome != "ok":
                    continue
                query_hash = sha256(call.query.encode("utf-8")).hexdigest()
                ranking = rankings.get(query_hash)
                if ranking is None or ranking["query"] != call.query:
                    raise ValueError(f"saved ranking is missing for query hash {query_hash}")
                stored_docids = [result.docid for result in call.results]
                if ranking["bm25_docids"][: len(stored_docids)] != stored_docids:
                    mismatches += 1
                baseline_tokens += sum(
                    len(tokenizer.encode(result.snippet, add_special_tokens=False))
                    for result in call.results
                )
                calls.append(
                    {
                        "query": call.query,
                        "bm25_docids": ranking["bm25_docids"][:5],
                        "dense_docids": ranking["dense_docids"][:20],
                    }
                )
            prepared.append(
                PreparedRun(
                    trial_id=trial["trial_id"],
                    query_id=item.query_id,
                    calls=calls,
                    evidence_docids=set(gold_row["evidence_docids"]),
                    baseline_ingress_tokens=baseline_tokens,
                )
            )
    return prepared, mismatches


def _make_policy(*, fixed: dict[str, int], search_budget: int) -> ProgressiveDisclosurePolicy:
    open_budget = fixed["open_evidence_ingress_token_budget"]
    return ProgressiveDisclosurePolicy(
        anchor_count=fixed["anchor_count"],
        dense_lead_count=fixed["dense_lead_count"],
        anchor_token_cap=fixed["anchor_token_cap"],
        lead_token_cap=fixed["lead_token_cap"],
        open_token_cap=fixed["open_token_cap"],
        maximum_open_calls=fixed["maximum_open_calls"],
        total_evidence_ingress_token_budget=search_budget + open_budget,
        open_evidence_ingress_token_budget=open_budget,
    )


def _simulate_run(
    source: PreparedRun,
    *,
    policy: ProgressiveDisclosurePolicy,
    tokenizer: Any,
    load_document: Callable[[str], str],
) -> CalibrationRow:
    session = ProgressiveDisclosureSession(
        run_id=f"calibration-{source.trial_id}-{source.query_id}",
        policy=policy,
        tokenizer=tokenizer,
        document_loader=load_document,
    )
    baseline_docids: set[str] = set()
    candidate_docids: set[str] = set()
    exhausted = False
    for call in source.calls:
        baseline_docids.update(call["bm25_docids"])
        result = session.search(
            bm25_candidates=[
                EvidenceCandidate(
                    docid=docid,
                    score=float(5 - rank),
                    text=format_bm25_anchor(load_document(docid)),
                )
                for rank, docid in enumerate(call["bm25_docids"])
            ],
            dense_candidates=[
                EvidenceCandidate(
                    docid=docid,
                    score=float(20 - rank),
                    text=format_dense_lead(docid, load_document(docid)),
                )
                for rank, docid in enumerate(call["dense_docids"])
            ],
        )
        candidate_docids.update(row.docid for row in result.results)
        exhausted = exhausted or result.ingress_budget_exhausted
    snapshot = session.snapshot()
    return CalibrationRow(
        trial_id=source.trial_id,
        query_id=source.query_id,
        search_calls=len(source.calls),
        baseline_evidence_recall=_recall(baseline_docids, source.evidence_docids),
        candidate_evidence_recall=_recall(candidate_docids, source.evidence_docids),
        baseline_ingress_tokens=source.baseline_ingress_tokens,
        candidate_search_ingress_tokens=snapshot.search_ingress_tokens,
        candidate_search_budget_exhausted=exhausted,
        baseline_unique_docids=len(baseline_docids),
        candidate_unique_docids=len(candidate_docids),
    )


def _aggregate(rows: Sequence[CalibrationRow]) -> CalibrationMetrics:
    baseline_recall = mean(row.baseline_evidence_recall for row in rows) * 100
    candidate_recall = mean(row.candidate_evidence_recall for row in rows) * 100
    baseline_mean = mean(row.baseline_ingress_tokens for row in rows)
    candidate_mean = mean(row.candidate_search_ingress_tokens for row in rows)
    baseline_p90 = _nearest_rank_percentile(
        [row.baseline_ingress_tokens for row in rows], 0.90
    )
    candidate_p90 = _nearest_rank_percentile(
        [row.candidate_search_ingress_tokens for row in rows], 0.90
    )
    return CalibrationMetrics(
        query_count=len(rows),
        search_calls=sum(row.search_calls for row in rows),
        baseline_evidence_recall_percent=round(baseline_recall, 6),
        candidate_evidence_recall_percent=round(candidate_recall, 6),
        evidence_recall_delta_pp=round(candidate_recall - baseline_recall, 6),
        baseline_mean_ingress_tokens=round(baseline_mean, 6),
        candidate_mean_search_ingress_tokens=round(candidate_mean, 6),
        aggregate_search_ingress_ratio=_ratio(candidate_mean, baseline_mean),
        baseline_p90_ingress_tokens=baseline_p90,
        candidate_p90_search_ingress_tokens=candidate_p90,
        p90_search_ingress_ratio=_ratio(candidate_p90, baseline_p90),
        search_budget_exhausted_queries=sum(
            row.candidate_search_budget_exhausted for row in rows
        ),
    )


def _nearest_rank_percentile(values: Sequence[int], percentile: float) -> int:
    if not values or not 0 < percentile <= 1:
        raise ValueError("percentile requires values and must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _round_up(value: int, quantum: int) -> int:
    if value < 0 or quantum <= 0:
        raise ValueError("rounding requires a non-negative value and positive quantum")
    return math.ceil(value / quantum) * quantum


def _recall(retrieved: set[str], relevant: set[str]) -> float:
    return len(retrieved.intersection(relevant)) / len(relevant) if relevant else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0 if numerator == 0 else float("inf")
    return round(numerator / denominator, 12)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
