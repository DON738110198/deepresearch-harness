from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, model_validator

from .contracts import RunState


class FailureFocus(str, Enum):
    COVERAGE_GAP = "coverage_gap"
    CITATION_MISMATCH = "citation_mismatch"
    CONFLICT_OMISSION = "conflict_omission"
    REDUNDANT_SEARCH = "redundant_search"
    BUDGET_PRIORITIZATION = "budget_prioritization"


class CorpusRecord(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    snippet: str = Field(min_length=1)
    synthetic: bool


class AnswerObligation(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    gold_evidence_ids: list[str] = Field(min_length=1)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    required: bool = True


class PilotTaskSpec(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    target_user: str = Field(min_length=1)
    decision_context: str = Field(min_length=1)
    failure_focus: FailureFocus
    obligations: list[AnswerObligation] = Field(min_length=2)
    forbidden_shortcuts: list[str] = Field(min_length=1)
    acceptance_notes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_obligation_ids(self) -> "PilotTaskSpec":
        ids = [obligation.id for obligation in self.obligations]
        if len(ids) != len(set(ids)):
            raise ValueError(f"task {self.id} has duplicate obligation ids")
        return self


class PilotSuite(BaseModel):
    suite_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: str = Field(pattern="^(planned|pilot)$")
    corpus_path: str = Field(min_length=1)
    expected_task_count: int = Field(ge=1)
    expected_per_failure_focus: int = Field(ge=1)
    tasks: list[PilotTaskSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_task_ids(self) -> "PilotSuite":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("pilot suite has duplicate task ids")
        return self


class HumanReportAnnotation(BaseModel):
    task_id: str
    covered_obligation_ids: list[str] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    citation_supported_claim_ids: list[str] = Field(default_factory=list)
    citation_mismatched_claim_ids: list[str] = Field(default_factory=list)
    conflict_handled_obligation_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def annotation_sets_are_consistent(self) -> "HumanReportAnnotation":
        list_fields = (
            self.covered_obligation_ids,
            self.supported_claim_ids,
            self.unsupported_claim_ids,
            self.citation_supported_claim_ids,
            self.citation_mismatched_claim_ids,
            self.conflict_handled_obligation_ids,
        )
        if any(len(values) != len(set(values)) for values in list_fields):
            raise ValueError("annotation id lists must not contain duplicates")
        if set(self.supported_claim_ids) & set(self.unsupported_claim_ids):
            raise ValueError("a claim cannot be both supported and unsupported")
        if set(self.citation_supported_claim_ids) & set(self.citation_mismatched_claim_ids):
            raise ValueError("a citation cannot be both supported and mismatched")
        return self


class BenchmarkScore(BaseModel):
    task_id: str
    evidence_id_recall: float = Field(ge=0, le=1)
    evidence_obligation_recall: float = Field(ge=0, le=1)
    citation_structural_integrity: float = Field(ge=0, le=1)
    obligation_coverage: float | None = Field(default=None, ge=0, le=1)
    citation_support_rate: float | None = Field(default=None, ge=0, le=1)
    unsupported_claim_rate: float | None = Field(default=None, ge=0, le=1)
    conflict_handling_rate: float | None = Field(default=None, ge=0, le=1)
    annotation_status: str


def load_suite(path: Path) -> PilotSuite:
    return PilotSuite.model_validate_json(path.read_text(encoding="utf-8"))


def validate_suite_assets(path: Path) -> tuple[PilotSuite, list[CorpusRecord]]:
    suite = load_suite(path)
    if len(suite.tasks) != suite.expected_task_count:
        raise ValueError(f"expected {suite.expected_task_count} tasks, found {len(suite.tasks)}")
    focus_counts = {focus: 0 for focus in FailureFocus}
    for task in suite.tasks:
        focus_counts[task.failure_focus] += 1
    unbalanced = {focus.value: count for focus, count in focus_counts.items() if count != suite.expected_per_failure_focus}
    if unbalanced:
        raise ValueError(f"failure focus counts do not match expected balance: {unbalanced}")
    corpus_path = path.parent / suite.corpus_path
    corpus = [CorpusRecord.model_validate(item) for item in json.loads(corpus_path.read_text(encoding="utf-8"))]
    corpus_ids = [record.id for record in corpus]
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("corpus has duplicate evidence ids")
    known_ids = set(corpus_ids)
    for task in suite.tasks:
        for obligation in task.obligations:
            referenced = set(obligation.gold_evidence_ids + obligation.counter_evidence_ids)
            missing = sorted(referenced - known_ids)
            if missing:
                raise ValueError(f"task {task.id} obligation {obligation.id} references unknown evidence: {missing}")
    return suite, corpus


def score_run(task: PilotTaskSpec, run: RunState, annotation: HumanReportAnnotation | None = None) -> BenchmarkScore:
    selected_ids = {evidence.id for evidence in run.evidence}
    required = [obligation for obligation in task.obligations if obligation.required]
    all_gold_ids = {evidence_id for obligation in required for evidence_id in obligation.gold_evidence_ids}
    evidence_id_recall = _ratio(len(selected_ids & all_gold_ids), len(all_gold_ids))
    covered_by_retrieval = sum(bool(selected_ids & set(obligation.gold_evidence_ids)) for obligation in required)
    evidence_obligation_recall = _ratio(covered_by_retrieval, len(required))

    claim_by_id = {claim.id: claim for claim in run.claims}
    evidence_ids = {evidence.id for evidence in run.evidence}
    structurally_valid = sum(
        citation.claim_id in claim_by_id
        and citation.evidence_id in evidence_ids
        and citation.evidence_id in claim_by_id[citation.claim_id].evidence_ids
        for citation in run.citations
    )
    citation_integrity = (
        _ratio(structurally_valid, len(run.citations))
        if run.citations
        else (1.0 if not run.claims else 0.0)
    )

    if annotation is None:
        return BenchmarkScore(
            task_id=task.id,
            evidence_id_recall=evidence_id_recall,
            evidence_obligation_recall=evidence_obligation_recall,
            citation_structural_integrity=citation_integrity,
            annotation_status="not_annotated",
        )
    if annotation.task_id != task.id:
        raise ValueError("annotation task_id does not match task")
    known_obligations = {obligation.id for obligation in task.obligations}
    annotated_obligations = set(annotation.covered_obligation_ids + annotation.conflict_handled_obligation_ids)
    if not annotated_obligations.issubset(known_obligations):
        raise ValueError("annotation references an unknown obligation")
    annotated_claims = set(
        annotation.supported_claim_ids
        + annotation.unsupported_claim_ids
        + annotation.citation_supported_claim_ids
        + annotation.citation_mismatched_claim_ids
    )
    if not annotated_claims.issubset(claim_by_id):
        raise ValueError("annotation references an unknown claim")

    total_claims = len(annotation.supported_claim_ids) + len(annotation.unsupported_claim_ids)
    total_cited_claims = len(annotation.citation_supported_claim_ids) + len(annotation.citation_mismatched_claim_ids)
    conflict_obligations = [obligation for obligation in required if obligation.counter_evidence_ids]
    return BenchmarkScore(
        task_id=task.id,
        evidence_id_recall=evidence_id_recall,
        evidence_obligation_recall=evidence_obligation_recall,
        citation_structural_integrity=citation_integrity,
        obligation_coverage=_ratio(len(set(annotation.covered_obligation_ids) & {item.id for item in required}), len(required)),
        citation_support_rate=(
            _ratio(len(annotation.citation_supported_claim_ids), total_cited_claims)
            if total_cited_claims
            else None
        ),
        unsupported_claim_rate=(
            _ratio(len(annotation.unsupported_claim_ids), total_claims)
            if total_claims
            else None
        ),
        conflict_handling_rate=(
            _ratio(len(set(annotation.conflict_handled_obligation_ids)), len(conflict_obligations))
            if conflict_obligations
            else None
        ),
        annotation_status="human_annotated",
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0
