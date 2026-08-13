from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import (
    _decrypt_query,
    _discover_parquet_urls,
    load_browsecomp_plus_target,
    load_pi_browsecomp_run,
    load_query_partitions,
    normalized_text_file_sha256,
)
from .pi_browsecomp import PiSmokeSummary


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DevelopmentGoldRow(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    gold_docids: list[str]
    evidence_docids: list[str]


class DevelopmentGoldSlice(StrictContract):
    schema_version: Literal["browsecomp-plus-development-gold-v0"] = (
        "browsecomp-plus-development-gold-v0"
    )
    created_at: str
    partition: Literal["development"] = "development"
    gold_accessed: Literal[True] = True
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accessed_fields: tuple[
        Literal["query_id"],
        Literal["query"],
        Literal["answer"],
        Literal["gold_docs.docid"],
        Literal["evidence_docs.docid"],
    ]
    excluded_fields: tuple[
        Literal["gold_docs.text"],
        Literal["gold_docs.url"],
        Literal["evidence_docs.text"],
        Literal["evidence_docs.url"],
        Literal["negative_docs"],
    ]
    query_count: int = Field(gt=0)
    rows: list[DevelopmentGoldRow] = Field(min_length=1)

    @model_validator(mode="after")
    def rows_match_frozen_count(self) -> "DevelopmentGoldSlice":
        ids = [row.query_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("development gold query IDs must be unique")
        if self.query_count != len(self.rows):
            raise ValueError("development gold query_count does not match rows")
        return self


class DiagnosticRow(StrictContract):
    query_id: str
    status: Literal["succeeded", "failed", "budget_exhausted"]
    answer_schema_complete: bool
    exact_answer_extracted: bool
    normalized_exact_match: bool
    evidence_recall: float | None = Field(default=None, ge=0, le=1)
    gold_recall: float | None = Field(default=None, ge=0, le=1)
    search_calls: int = Field(ge=0)
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DiagnosticSummary(StrictContract):
    schema_version: Literal["browsecomp-plus-gold-diagnostic-v0"] = (
        "browsecomp-plus-gold-diagnostic-v0"
    )
    created_at: str
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    official_accuracy_status: Literal["planned_not_run"] = "planned_not_run"
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_slice_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(gt=0)
    schema_complete: int = Field(ge=0)
    exact_answer_extracted: int = Field(ge=0)
    normalized_exact_match: int = Field(ge=0)
    normalized_exact_match_percent: float = Field(ge=0, le=100)
    evidence_recall_percent: float | None = Field(default=None, ge=0, le=100)
    gold_recall_percent: float | None = Field(default=None, ge=0, le=100)
    rows: list[DiagnosticRow] = Field(min_length=1)


def freeze_development_gold_slice(
    *,
    manifest_path: Path,
    partitions_path: Path,
    source_summary_path: Path,
    output_path: Path,
) -> DevelopmentGoldSlice:
    repository_root = _find_repository_root(manifest_path.resolve())
    _require_under_runs(output_path, repository_root)
    if output_path.exists():
        raise ValueError("development gold output already exists")

    manifest = load_browsecomp_plus_target(manifest_path)
    partitions = load_query_partitions(partitions_path, manifest_path=manifest_path)
    summary_bytes = source_summary_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    if summary.target_manifest_sha256 != normalized_text_file_sha256(manifest_path):
        raise ValueError("prediction summary targets a different benchmark manifest")
    if any(item.run_sha256 is None or item.prediction_sha256 is None for item in summary.items):
        raise ValueError("all predictions must be frozen before development gold access")

    partition_by_id = {row.query_id: row.partition for row in partitions.rows}
    query_ids = sorted(item.query_id for item in summary.items)
    if any(partition_by_id.get(query_id) != "development" for query_id in query_ids):
        raise ValueError("gold slice request contains a sealed-holdout or unknown query")
    prediction_set_sha256 = _prediction_set_sha256(summary)

    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError(
            'install the optional dependency with pip install -e ".[browsecomp-plus]"'
        ) from error

    parquet_urls = _discover_parquet_urls(
        dataset_name=manifest.benchmark.query_dataset.name,
        revision=manifest.benchmark.query_dataset.revision,
    )
    connection = duckdb.connect()
    try:
        encrypted_rows = connection.execute(
            "SELECT query_id, query, answer, gold_docs, evidence_docs "
            "FROM read_parquet(?) WHERE query_id IN (SELECT unnest(?))",
            [parquet_urls, query_ids],
        ).fetchall()
    finally:
        connection.close()

    by_id = {str(row[0]): row for row in encrypted_rows}
    if set(by_id) != set(query_ids):
        raise ValueError("pinned dataset did not return the frozen prediction IDs")
    rows = []
    for query_id in query_ids:
        _, encrypted_question, encrypted_answer, gold_docs, evidence_docs = by_id[query_id]
        rows.append(
            DevelopmentGoldRow(
                query_id=query_id,
                question=_decrypt_query(encrypted_question),
                answer=_decrypt_query(encrypted_answer),
                gold_docids=sorted(
                    {_decrypt_query(document["docid"]) for document in gold_docs}
                ),
                evidence_docids=sorted(
                    {_decrypt_query(document["docid"]) for document in evidence_docs}
                ),
            )
        )

    artifact = DevelopmentGoldSlice(
        created_at=datetime.now(timezone.utc).isoformat(),
        target_manifest_sha256=normalized_text_file_sha256(manifest_path),
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        prediction_set_sha256=prediction_set_sha256,
        accessed_fields=(
            "query_id",
            "query",
            "answer",
            "gold_docs.docid",
            "evidence_docs.docid",
        ),
        excluded_fields=(
            "gold_docs.text",
            "gold_docs.url",
            "evidence_docs.text",
            "evidence_docs.url",
            "negative_docs",
        ),
        query_count=len(rows),
        rows=rows,
    )
    _atomic_write(output_path, artifact.model_dump_json(indent=2))
    return artifact


def score_gold_diagnostic(
    *, source_dir: Path, gold_slice_path: Path, output_path: Path
) -> DiagnosticSummary:
    repository_root = _find_repository_root(source_dir.resolve())
    _require_under_runs(output_path, repository_root)
    if output_path.exists():
        raise ValueError("diagnostic score output already exists")

    summary_path = source_dir / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    gold_bytes = gold_slice_path.read_bytes()
    gold = DevelopmentGoldSlice.model_validate_json(gold_bytes)
    if gold.source_summary_sha256 != sha256(summary_bytes).hexdigest():
        raise ValueError("development gold was not opened for this prediction summary")
    if gold.prediction_set_sha256 != _prediction_set_sha256(summary):
        raise ValueError("development gold prediction set hash does not match")

    gold_by_id = {row.query_id: row for row in gold.rows}
    rows = []
    for item in summary.items:
        reference = gold_by_id.get(item.query_id)
        if reference is None or item.run_sha256 is None or item.prediction_sha256 is None:
            raise ValueError(f"missing frozen run or gold for query {item.query_id}")
        run_path = source_dir / _safe_id(item.query_id) / "run.json"
        run_bytes = run_path.read_bytes()
        if sha256(run_bytes).hexdigest() != item.run_sha256:
            raise ValueError(f"source run hash mismatch for query {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        if sha256(run.answer_text.encode("utf-8")).hexdigest() != item.prediction_sha256:
            raise ValueError(f"prediction hash mismatch for query {item.query_id}")
        predicted_answer = extract_exact_answer(run.answer_text)
        retrieved = {
            result.docid for call in run.search_calls for result in call.results
        }
        rows.append(
            DiagnosticRow(
                query_id=item.query_id,
                status=run.status,
                answer_schema_complete=bool(run.answer_schema_complete),
                exact_answer_extracted=predicted_answer is not None,
                normalized_exact_match=(
                    predicted_answer is not None
                    and normalize_exact_answer(predicted_answer)
                    == normalize_exact_answer(reference.answer)
                ),
                evidence_recall=_recall(retrieved, set(reference.evidence_docids)),
                gold_recall=_recall(retrieved, set(reference.gold_docids)),
                search_calls=len(run.search_calls),
                prediction_sha256=item.prediction_sha256,
                reference_answer_sha256=sha256(
                    reference.answer.encode("utf-8")
                ).hexdigest(),
            )
        )

    evidence = [row.evidence_recall for row in rows if row.evidence_recall is not None]
    gold_recalls = [row.gold_recall for row in rows if row.gold_recall is not None]
    exact = sum(row.normalized_exact_match for row in rows)
    artifact = DiagnosticSummary(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        gold_slice_sha256=sha256(gold_bytes).hexdigest(),
        query_count=len(rows),
        schema_complete=sum(row.answer_schema_complete for row in rows),
        exact_answer_extracted=sum(row.exact_answer_extracted for row in rows),
        normalized_exact_match=exact,
        normalized_exact_match_percent=round(exact / len(rows) * 100, 2),
        evidence_recall_percent=(
            round(sum(evidence) / len(evidence) * 100, 2) if evidence else None
        ),
        gold_recall_percent=(
            round(sum(gold_recalls) / len(gold_recalls) * 100, 2)
            if gold_recalls
            else None
        ),
        rows=rows,
    )
    _atomic_write(output_path, artifact.model_dump_json(indent=2))
    return artifact


def extract_exact_answer(answer_text: str) -> str | None:
    match = re.search(
        r"(?:^|\n)\s*(?:\*\*)?Exact Answer(?:\*\*)?\s*:\s*(.+?)(?=\n|$)",
        answer_text,
        re.I,
    )
    return match.group(1).strip() if match and match.group(1).strip() else None


def normalize_exact_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(normalized.split())


def _recall(retrieved: set[str], relevant: set[str]) -> float | None:
    return len(retrieved & relevant) / len(relevant) if relevant else None


def _prediction_set_sha256(summary: PiSmokeSummary) -> str:
    canonical = "\n".join(
        f"{item.query_id}\t{item.prediction_sha256}\t{item.run_sha256}"
        for item in sorted(summary.items, key=lambda row: row.query_id)
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("gold and diagnostic artifacts must remain under ignored runs/")


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate repository root")


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not safe or safe in {".", ".."}:
        raise ValueError("query ID cannot be represented safely")
    return safe


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
