from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DevelopmentGoldSlice
from .browsecomp_plus import load_pi_browsecomp_run
from .evidence_preview import (
    build_query_aware_lead_preview,
    format_query_aware_dense_lead,
    query_term_overlap,
)
from .pi_browsecomp import PiSmokeSummary
from .progressive_disclosure import format_dense_lead


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreviewCalibrationRow(StrictContract):
    query_id: str = Field(min_length=1)
    search_index: int = Field(ge=1)
    docid: str = Field(min_length=1)
    baseline_token_cap: Literal[24] = 24
    candidate_token_cap: Literal[64] = 64
    baseline_visible_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_visible_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_visible_title: bool
    baseline_visible_date: bool
    baseline_query_term_matches: int = Field(ge=0)
    baseline_selectable: bool
    candidate_visible_title: bool
    candidate_visible_date: bool
    candidate_query_term_matches: int = Field(ge=0)
    candidate_selectable: bool


class EvidencePreviewCalibration(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-evidence-preview-calibration-v0"
    ] = "browsecomp-plus-evidence-preview-calibration-v0"
    created_at: str
    status: Literal["calibration_passed", "calibration_failed"]
    provider_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    source_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_slice_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query_count: int = Field(gt=0)
    dense_result_count: int = Field(ge=0)
    relevant_dense_hit_count: int = Field(gt=0)
    baseline_selectable_relevant_hits: int = Field(ge=0)
    candidate_selectable_relevant_hits: int = Field(ge=0)
    anchor_count: Literal[5] = 5
    dense_lead_count: Literal[15] = 15
    anchor_token_cap: Literal[512] = 512
    baseline_lead_token_cap: Literal[24] = 24
    candidate_lead_token_cap: Literal[64] = 64
    maximum_search_calls: Literal[8] = 8
    candidate_maximum_search_ingress_tokens: int = Field(ge=0)
    search_ingress_token_budget: Literal[29184] = 29184
    rows: list[PreviewCalibrationRow] = Field(min_length=1)
    selected_policy: Literal["query_window_v0_64", "none"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def counts_and_selection_match(self) -> "EvidencePreviewCalibration":
        if self.relevant_dense_hit_count != len(self.rows):
            raise ValueError("relevant dense hit count differs from rows")
        if self.baseline_selectable_relevant_hits != sum(
            row.baseline_selectable for row in self.rows
        ):
            raise ValueError("baseline selectable count differs from rows")
        if self.candidate_selectable_relevant_hits != sum(
            row.candidate_selectable for row in self.rows
        ):
            raise ValueError("candidate selectable count differs from rows")
        passed = (
            self.candidate_selectable_relevant_hits == self.relevant_dense_hit_count
            and self.candidate_maximum_search_ingress_tokens
            <= self.search_ingress_token_budget
        )
        if self.status != ("calibration_passed" if passed else "calibration_failed"):
            raise ValueError("calibration status differs from frozen gates")
        if self.selected_policy != ("query_window_v0_64" if passed else "none"):
            raise ValueError("selected preview policy differs from frozen gates")
        return self


def calibrate_evidence_preview(
    *,
    source_dir: Path,
    gold_slice_path: Path,
    document_index_path: Path,
    tokenizer_dir: Path,
    output_path: Path,
) -> EvidencePreviewCalibration:
    if output_path.exists():
        raise ValueError("evidence preview calibration output already exists")
    summary_path = source_dir / "summary.json"
    summary = PiSmokeSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    gold = DevelopmentGoldSlice.model_validate_json(
        gold_slice_path.read_text(encoding="utf-8")
    )
    if {item.query_id for item in summary.items} != {
        row.query_id for row in gold.rows
    }:
        raise ValueError("summary and gold slice query IDs differ")

    try:
        from pyserini.search.lucene import LuceneSearcher
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("preview calibration dependencies are not installed") from error
    documents = LuceneSearcher(str(document_index_path.resolve()))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir.resolve(),
        local_files_only=True,
        trust_remote_code=False,
    )

    gold_by_query = {row.query_id: row for row in gold.rows}
    dense_result_count = 0
    rows: list[PreviewCalibrationRow] = []
    for item in summary.items:
        run = load_pi_browsecomp_run(_resolve_run_path(summary_path, item.run_path))
        gold_row = gold_by_query[item.query_id]
        relevant_docids = set(gold_row.gold_docids) | set(gold_row.evidence_docids)
        for search_index, call in enumerate(run.search_calls, start=1):
            for result in call.results:
                if not result.snippet.startswith("[Dense lead"):
                    continue
                dense_result_count += 1
                if result.docid not in relevant_docids:
                    continue
                document = documents.doc(result.docid)
                if document is None:
                    raise ValueError(f"relevant document is missing: {result.docid}")
                contents = json.loads(document.raw())["contents"]
                preview = build_query_aware_lead_preview(contents, call.query)
                baseline_visible = _truncate(
                    tokenizer,
                    format_dense_lead(result.docid, contents),
                    24,
                )
                candidate_visible = _truncate(
                    tokenizer,
                    format_query_aware_dense_lead(result.docid, preview),
                    64,
                )
                baseline_title = _metadata_visible(preview.title, baseline_visible)
                baseline_date = _metadata_visible(preview.date, baseline_visible)
                candidate_title = _metadata_visible(preview.title, candidate_visible)
                candidate_date = _metadata_visible(preview.date, candidate_visible)
                baseline_matches = len(query_term_overlap(call.query, baseline_visible))
                candidate_matches = len(query_term_overlap(call.query, candidate_visible))
                rows.append(
                    PreviewCalibrationRow(
                        query_id=item.query_id,
                        search_index=search_index,
                        docid=result.docid,
                        baseline_visible_sha256=_text_sha256(baseline_visible),
                        candidate_visible_sha256=_text_sha256(candidate_visible),
                        baseline_visible_title=baseline_title,
                        baseline_visible_date=baseline_date,
                        baseline_query_term_matches=baseline_matches,
                        baseline_selectable=_selectable(
                            baseline_title, baseline_date, baseline_matches
                        ),
                        candidate_visible_title=candidate_title,
                        candidate_visible_date=candidate_date,
                        candidate_query_term_matches=candidate_matches,
                        candidate_selectable=_selectable(
                            candidate_title, candidate_date, candidate_matches
                        ),
                    )
                )

    maximum_search_ingress = 8 * ((5 * 512) + (15 * 64))
    candidate_selectable = sum(row.candidate_selectable for row in rows)
    passed = candidate_selectable == len(rows) and maximum_search_ingress <= 29184
    result = EvidencePreviewCalibration(
        created_at=datetime.now(timezone.utc).isoformat(),
        status="calibration_passed" if passed else "calibration_failed",
        source_summary_sha256=_file_sha256(summary_path),
        gold_slice_sha256=_file_sha256(gold_slice_path),
        source_query_count=summary.query_count,
        dense_result_count=dense_result_count,
        relevant_dense_hit_count=len(rows),
        baseline_selectable_relevant_hits=sum(
            row.baseline_selectable for row in rows
        ),
        candidate_selectable_relevant_hits=candidate_selectable,
        candidate_maximum_search_ingress_tokens=maximum_search_ingress,
        rows=rows,
        selected_policy="query_window_v0_64" if passed else "none",
        claim_boundary=(
            "This gold-informed development bad-case calibration only checks whether "
            "retrieved dense leads become selectable under the fixed ingress budget. "
            "It does not measure answer accuracy or official benchmark performance."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _truncate(tokenizer: Any, text: str, token_cap: int) -> str:
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=token_cap,
    )
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _metadata_visible(value: str | None, visible: str) -> bool:
    if value is None:
        return False
    return _normalize(value) in _normalize(visible)


def _selectable(title: bool, date: bool, query_matches: int) -> bool:
    return title or date or query_matches >= 2


def _normalize(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _resolve_run_path(summary_path: Path, value: str) -> Path:
    root = summary_path.resolve().parents[4]
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    resolved.relative_to(root)
    return resolved


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
