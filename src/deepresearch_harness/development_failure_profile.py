from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DiagnosticSummary
from .development_judge import DevelopmentJudgeResult
from .pi_browsecomp import PiSmokeSummary
from .research_loop import ArtifactReference


FailureCategory = Literal[
    "answer_contract_failure",
    "judge_correct",
    "reference_document_not_retrieved",
    "reference_document_retrieved_answer_wrong",
]
NextLayer = Literal[
    "answer_contract",
    "retrieval_visibility",
    "evidence_selection_opening_or_synthesis",
    "mixed_stratified_audit",
]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailureRouting(StrictContract):
    minimum_scored_wrong_cases: int = Field(ge=1)
    answer_contract_failure_share_all_percent: float = Field(ge=0, le=100)
    reference_document_not_retrieved_share_wrong_percent: float = Field(
        ge=0, le=100
    )
    reference_document_retrieved_answer_wrong_share_wrong_percent: float = Field(
        ge=0, le=100
    )


class FailureTaxonomyRegistration(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-development-failure-taxonomy-registration-v0"
    ]
    status: Literal["registered_before_gold_scoring"]
    registered_at: str = Field(min_length=1)
    profile_registration: ArtifactReference
    query_count: Literal[175]
    classification_precedence: tuple[FailureCategory, ...]
    routing: FailureRouting
    multi_agent_status: Literal["deferred"]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def rules_are_complete(self) -> "FailureTaxonomyRegistration":
        expected = (
            "answer_contract_failure",
            "judge_correct",
            "reference_document_not_retrieved",
            "reference_document_retrieved_answer_wrong",
        )
        if self.classification_precedence != expected:
            raise ValueError("failure taxonomy classification order changed")
        return self


class FailureProfileRow(StrictContract):
    query_id: str = Field(min_length=1)
    category: FailureCategory
    judge_correct: bool
    answer_schema_complete: bool
    exact_answer_extracted: bool
    normalized_exact_match: bool
    evidence_recall: float | None = Field(default=None, ge=0, le=1)
    gold_recall: float | None = Field(default=None, ge=0, le=1)
    search_calls: int = Field(ge=0)


class DevelopmentFailureProfile(StrictContract):
    schema_version: Literal["browsecomp-plus-development-failure-profile-v0"] = (
        "browsecomp-plus-development-failure-profile-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["development_diagnostic_not_official"] = (
        "development_diagnostic_not_official"
    )
    taxonomy_registration: ArtifactReference
    source_summary: ArtifactReference
    gold_diagnostic: ArtifactReference
    judge_result: ArtifactReference
    query_count: Literal[175]
    judge_correct: int = Field(ge=0, le=175)
    judge_wrong: int = Field(ge=0, le=175)
    schema_complete: int = Field(ge=0, le=175)
    category_counts: dict[FailureCategory, int]
    wrong_category_counts: dict[FailureCategory, int]
    next_layer: NextLayer
    multi_agent_status: Literal["deferred"] = "deferred"
    rows: list[FailureProfileRow]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def counts_match_rows(self) -> "DevelopmentFailureProfile":
        if len(self.rows) != self.query_count:
            raise ValueError("failure profile row count changed")
        if len({row.query_id for row in self.rows}) != self.query_count:
            raise ValueError("failure profile query IDs must be unique")
        if self.judge_correct + self.judge_wrong != self.query_count:
            raise ValueError("failure profile Judge accounting changed")
        if self.judge_correct != sum(row.judge_correct for row in self.rows):
            raise ValueError("failure profile correct count changed")
        if self.schema_complete != sum(row.answer_schema_complete for row in self.rows):
            raise ValueError("failure profile schema count changed")
        if self.category_counts != dict(Counter(row.category for row in self.rows)):
            raise ValueError("failure profile category counts changed")
        expected_wrong = dict(
            Counter(row.category for row in self.rows if not row.judge_correct)
        )
        if self.wrong_category_counts != expected_wrong:
            raise ValueError("failure profile wrong-category counts changed")
        return self


def classify_failure(
    *,
    judge_correct: bool,
    answer_schema_complete: bool,
    exact_answer_extracted: bool,
    evidence_recall: float | None,
    gold_recall: float | None,
) -> FailureCategory:
    if not answer_schema_complete or not exact_answer_extracted:
        return "answer_contract_failure"
    if judge_correct:
        return "judge_correct"
    recalls = [value for value in (evidence_recall, gold_recall) if value is not None]
    if not recalls or max(recalls) == 0:
        return "reference_document_not_retrieved"
    return "reference_document_retrieved_answer_wrong"


def select_next_layer(
    *,
    query_count: int,
    judge_wrong: int,
    category_counts: dict[str, int],
    wrong_category_counts: dict[str, int],
    routing: FailureRouting,
) -> NextLayer:
    if judge_wrong < routing.minimum_scored_wrong_cases:
        raise ValueError("not enough scored wrong cases for registered routing")
    contract_share = _percent(
        category_counts.get("answer_contract_failure", 0), query_count
    )
    if contract_share >= routing.answer_contract_failure_share_all_percent:
        return "answer_contract"
    retrieval_share = _percent(
        wrong_category_counts.get("reference_document_not_retrieved", 0),
        judge_wrong,
    )
    if (
        retrieval_share
        >= routing.reference_document_not_retrieved_share_wrong_percent
    ):
        return "retrieval_visibility"
    downstream_share = _percent(
        wrong_category_counts.get(
            "reference_document_retrieved_answer_wrong", 0
        ),
        judge_wrong,
    )
    if (
        downstream_share
        >= routing.reference_document_retrieved_answer_wrong_share_wrong_percent
    ):
        return "evidence_selection_opening_or_synthesis"
    return "mixed_stratified_audit"


def build_development_failure_profile(
    *,
    taxonomy_registration_path: Path,
    source_summary_path: Path,
    gold_diagnostic_path: Path,
    judge_result_path: Path,
    output_path: Path,
) -> DevelopmentFailureProfile:
    root = _repository_root(taxonomy_registration_path)
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("development failure profile must stay under runs/")
    if output_path.exists():
        raise ValueError("development failure profile already exists")
    registration = FailureTaxonomyRegistration.model_validate_json(
        taxonomy_registration_path.read_text(encoding="utf-8")
    )
    _validate_artifact(root, registration.profile_registration)

    summary_bytes = source_summary_path.read_bytes()
    diagnostic_bytes = gold_diagnostic_path.read_bytes()
    judge_bytes = judge_result_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    diagnostic = DiagnosticSummary.model_validate_json(diagnostic_bytes)
    judge = DevelopmentJudgeResult.model_validate_json(judge_bytes)
    summary_sha = sha256(summary_bytes).hexdigest()
    diagnostic_sha = sha256(diagnostic_bytes).hexdigest()
    judge_sha = sha256(judge_bytes).hexdigest()
    if summary.query_count != registration.query_count:
        raise ValueError("development failure profile query count changed")
    if summary.succeeded != registration.query_count or summary.failed:
        raise ValueError("development failure profile requires complete predictions")
    if diagnostic.source_summary_sha256 != summary_sha:
        raise ValueError("gold diagnostic targets another summary")
    if judge.source_summary_sha256 != summary_sha:
        raise ValueError("Judge result targets another summary")
    if judge.gold_slice_sha256 != diagnostic.gold_slice_sha256:
        raise ValueError("Judge and gold diagnostic use different gold slices")
    if judge.status != "succeeded" or judge.parse_failures or judge.request_failures:
        raise ValueError("development failure profile requires a valid Judge result")

    diagnostic_by_id = {row.query_id: row for row in diagnostic.rows}
    judge_by_id = {row.query_id: row for row in judge.observations}
    expected_ids = {item.query_id for item in summary.items}
    if set(diagnostic_by_id) != expected_ids or set(judge_by_id) != expected_ids:
        raise ValueError("development failure profile query grids differ")

    rows = []
    for query_id in sorted(expected_ids):
        diagnostic_row = diagnostic_by_id[query_id]
        judge_row = judge_by_id[query_id]
        if judge_row.correct is None:
            raise ValueError(f"Judge label is missing for query {query_id}")
        rows.append(
            FailureProfileRow(
                query_id=query_id,
                category=classify_failure(
                    judge_correct=judge_row.correct,
                    answer_schema_complete=diagnostic_row.answer_schema_complete,
                    exact_answer_extracted=diagnostic_row.exact_answer_extracted,
                    evidence_recall=diagnostic_row.evidence_recall,
                    gold_recall=diagnostic_row.gold_recall,
                ),
                judge_correct=judge_row.correct,
                answer_schema_complete=diagnostic_row.answer_schema_complete,
                exact_answer_extracted=diagnostic_row.exact_answer_extracted,
                normalized_exact_match=diagnostic_row.normalized_exact_match,
                evidence_recall=diagnostic_row.evidence_recall,
                gold_recall=diagnostic_row.gold_recall,
                search_calls=diagnostic_row.search_calls,
            )
        )
    category_counts = dict(Counter(row.category for row in rows))
    wrong_category_counts = dict(
        Counter(row.category for row in rows if not row.judge_correct)
    )
    judge_correct = sum(row.judge_correct for row in rows)
    profile = DevelopmentFailureProfile(
        created_at=datetime.now(timezone.utc).isoformat(),
        taxonomy_registration=_reference(taxonomy_registration_path, root),
        source_summary=_reference(source_summary_path, root),
        gold_diagnostic=ArtifactReference(
            path=gold_diagnostic_path.resolve().relative_to(root).as_posix(),
            sha256=diagnostic_sha,
        ),
        judge_result=ArtifactReference(
            path=judge_result_path.resolve().relative_to(root).as_posix(),
            sha256=judge_sha,
        ),
        query_count=175,
        judge_correct=judge_correct,
        judge_wrong=175 - judge_correct,
        schema_complete=sum(row.answer_schema_complete for row in rows),
        category_counts=category_counts,
        wrong_category_counts=wrong_category_counts,
        next_layer=select_next_layer(
            query_count=175,
            judge_wrong=175 - judge_correct,
            category_counts=category_counts,
            wrong_category_counts=wrong_category_counts,
            routing=registration.routing,
        ),
        rows=rows,
        claim_boundary=registration.claim_boundary,
    )
    _atomic_write(output_path, profile.model_dump(mode="json"))
    return profile


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("failure-profile percentage denominator must be positive")
    return numerator / denominator * 100


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("could not locate development-failure-profile repository root")


def _validate_artifact(root: Path, reference: ArtifactReference) -> None:
    path = (root / reference.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"failure-taxonomy artifact is missing: {reference.path}")
    if sha256(path.read_bytes()).hexdigest() != reference.sha256:
        raise ValueError(f"failure-taxonomy artifact hash changed: {reference.path}")


def _reference(path: Path, root: Path) -> ArtifactReference:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("development failure profile source escapes repository")
    return ArtifactReference(
        path=resolved.relative_to(root).as_posix(),
        sha256=sha256(resolved.read_bytes()).hexdigest(),
    )


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
