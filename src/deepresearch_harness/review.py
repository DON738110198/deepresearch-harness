from __future__ import annotations

import json
import random
from enum import Enum
from hashlib import sha256
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel, Field, HttpUrl, model_validator

from .batch import BatchSummary
from .benchmark import AnswerObligation, BenchmarkScore, HumanReportAnnotation, PilotTaskSpec, load_suite, score_run
from .contracts import Citation, Claim, Evidence, RunState, RunStatus, Task


class ReviewEvidence(BaseModel):
    id: str
    title: str
    url: HttpUrl
    excerpt: str


class BlindCandidate(BaseModel):
    candidate_id: str
    report: str
    evidence: list[ReviewEvidence]
    claims: list[Claim]
    citations: list[Citation]


class BlindTask(BaseModel):
    task_id: str
    question: str
    decision_context: str
    obligations: list[AnswerObligation]
    forbidden_shortcuts: list[str]
    acceptance_notes: list[str]
    candidates: list[BlindCandidate]


class BlindReviewPacket(BaseModel):
    experiment_id: str
    rubric: list[str]
    tasks: list[BlindTask]


class ReviewerType(str, Enum):
    HUMAN = "human"
    AI_ASSISTED = "ai_assisted"


class CandidateAnnotation(BaseModel):
    task_id: str
    candidate_id: str
    covered_obligation_ids: list[str] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    citation_supported_claim_ids: list[str] = Field(default_factory=list)
    citation_mismatched_claim_ids: list[str] = Field(default_factory=list)
    irrelevant_claim_ids: list[str] = Field(default_factory=list)
    conflict_handled_obligation_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def annotation_sets_are_consistent(self) -> "CandidateAnnotation":
        list_fields = (
            self.covered_obligation_ids,
            self.supported_claim_ids,
            self.unsupported_claim_ids,
            self.citation_supported_claim_ids,
            self.citation_mismatched_claim_ids,
            self.irrelevant_claim_ids,
            self.conflict_handled_obligation_ids,
        )
        if any(len(values) != len(set(values)) for values in list_fields):
            raise ValueError("annotation id lists must not contain duplicates")
        if set(self.supported_claim_ids) & set(self.unsupported_claim_ids):
            raise ValueError("a claim cannot be both supported and unsupported")
        if set(self.citation_supported_claim_ids) & set(self.citation_mismatched_claim_ids):
            raise ValueError("a cited claim cannot be both supported and mismatched")
        return self


class BlindReviewSubmission(BaseModel):
    experiment_id: str
    reviewer_type: ReviewerType
    annotations: list[CandidateAnnotation]

    @model_validator(mode="after")
    def candidate_keys_are_unique(self) -> "BlindReviewSubmission":
        keys = [(item.task_id, item.candidate_id) for item in self.annotations]
        if len(keys) != len(set(keys)):
            raise ValueError("review submission has duplicate task/candidate pairs")
        return self


class CandidateReviewScore(BaseModel):
    task_id: str
    candidate_id: str
    variant: str
    score: BenchmarkScore


class ReviewVariantAggregate(BaseModel):
    candidate_count: int = Field(ge=0)
    mean_evidence_id_recall: float | None = None
    mean_evidence_id_precision: float | None = None
    mean_evidence_obligation_recall: float | None = None
    mean_citation_structural_integrity: float | None = None
    mean_obligation_coverage: float | None = None
    mean_citation_support_rate: float | None = None
    mean_unsupported_claim_rate: float | None = None
    mean_irrelevant_claim_rate: float | None = None
    mean_conflict_handling_rate: float | None = None


class BlindReviewResult(BaseModel):
    experiment_id: str
    reviewer_type: ReviewerType
    result_status: str
    packet_sha256: str
    annotations_sha256: str
    answer_key_sha256: str
    candidates: list[CandidateReviewScore]
    aggregates: dict[str, ReviewVariantAggregate]


def prepare_blind_review(
    *,
    summary_path: Path,
    suite_path: Path,
    output_dir: Path,
    seed: int = 20260812,
) -> tuple[Path, Path, Path]:
    summary = BatchSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    suite = load_suite(suite_path)
    tasks_by_id = {task.id: task for task in suite.tasks}
    records_by_task: dict[str, list] = {}
    for record in summary.records:
        if record.status == "succeeded":
            records_by_task.setdefault(record.task_id, []).append(record)

    blind_tasks: list[BlindTask] = []
    templates: list[CandidateAnnotation] = []
    answer_key: dict[str, dict[str, str]] = {}
    for task_id in sorted(records_by_task):
        records = records_by_task[task_id]
        if len(records) != 2:
            continue
        task = tasks_by_id[task_id]
        shuffled = list(records)
        random.Random(f"{seed}:{task_id}").shuffle(shuffled)
        candidates: list[BlindCandidate] = []
        answer_key[task_id] = {}
        for index, record in enumerate(shuffled):
            candidate_id = chr(ord("A") + index)
            state = RunState.model_validate_json(Path(record.state_path).read_text(encoding="utf-8"))
            report = Path(state.report_path).read_text(encoding="utf-8")
            candidates.append(
                BlindCandidate(
                    candidate_id=candidate_id,
                    report=report,
                    evidence=[
                        ReviewEvidence(id=item.id, title=item.title, url=item.url, excerpt=item.excerpt)
                        for item in state.evidence
                    ],
                    claims=state.claims,
                    citations=state.citations,
                )
            )
            answer_key[task_id][candidate_id] = record.variant
            templates.append(CandidateAnnotation(task_id=task_id, candidate_id=candidate_id))
        blind_tasks.append(_blind_task(task, candidates))

    packet = BlindReviewPacket(
        experiment_id=summary.experiment_id,
        rubric=[
            "Mark each required obligation covered only when the report addresses it substantively.",
            "Judge whether each cited evidence item supports the exact strength of its claim.",
            "Mark factual claims without eligible support as unsupported.",
            "Mark supported but decision-irrelevant claims as irrelevant.",
            "For obligations with counter-evidence, require acknowledgement and reconciliation of the conflict.",
        ],
        tasks=blind_tasks,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    packet_path = output_dir / "review_packet.json"
    template_path = output_dir / "annotation_template.json"
    key_path = output_dir / "answer_key.json"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    template = BlindReviewSubmission(
        experiment_id=summary.experiment_id,
        reviewer_type=ReviewerType.HUMAN,
        annotations=templates,
    )
    template_path.write_text(template.model_dump_json(indent=2), encoding="utf-8")
    key_path.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")
    return packet_path, template_path, key_path


def score_blind_review(
    *,
    packet_path: Path,
    annotations_path: Path,
    answer_key_path: Path,
    output_path: Path,
) -> BlindReviewResult:
    packet = BlindReviewPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    submission = BlindReviewSubmission.model_validate_json(annotations_path.read_text(encoding="utf-8"))
    annotations = validate_blind_submission(packet, submission)

    # The key is intentionally read only after the blinded submission is complete and valid.
    answer_key_text = answer_key_path.read_text(encoding="utf-8")
    answer_key = json.loads(answer_key_text)
    _validate_answer_key(packet, answer_key)

    candidate_scores: list[CandidateReviewScore] = []
    scores_by_variant: dict[str, list[BenchmarkScore]] = {}
    for task in packet.tasks:
        task_spec = _task_spec_from_blind_task(task)
        for candidate in task.candidates:
            annotation = annotations[(task.task_id, candidate.candidate_id)]
            score = score_run(
                task_spec,
                _run_from_blind_candidate(task, candidate),
                HumanReportAnnotation.model_validate(annotation.model_dump(exclude={"candidate_id"})),
            ).model_copy(
                update={
                    "annotation_status": (
                        "human_annotated"
                        if submission.reviewer_type == ReviewerType.HUMAN
                        else "ai_assisted_calibration"
                    )
                }
            )
            variant = answer_key[task.task_id][candidate.candidate_id]
            candidate_scores.append(
                CandidateReviewScore(
                    task_id=task.task_id,
                    candidate_id=candidate.candidate_id,
                    variant=variant,
                    score=score,
                )
            )
            scores_by_variant.setdefault(variant, []).append(score)

    result = BlindReviewResult(
        experiment_id=packet.experiment_id,
        reviewer_type=submission.reviewer_type,
        result_status="reviewed" if submission.reviewer_type == ReviewerType.HUMAN else "calibration_only",
        packet_sha256=file_sha256(packet_path),
        annotations_sha256=file_sha256(annotations_path),
        answer_key_sha256=sha256(answer_key_text.encode("utf-8")).hexdigest(),
        candidates=candidate_scores,
        aggregates={variant: _aggregate_review_scores(scores) for variant, scores in scores_by_variant.items()},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def validate_blind_submission(
    packet: BlindReviewPacket,
    submission: BlindReviewSubmission,
) -> dict[tuple[str, str], CandidateAnnotation]:
    if submission.experiment_id != packet.experiment_id:
        raise ValueError("review submission experiment_id does not match packet")
    expected = {
        (task.task_id, candidate.candidate_id): (task, candidate)
        for task in packet.tasks
        for candidate in task.candidates
    }
    actual = {(item.task_id, item.candidate_id): item for item in submission.annotations}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"review submission candidate set mismatch; missing={missing}, extra={extra}")

    for key, annotation in actual.items():
        task, candidate = expected[key]
        obligation_ids = {item.id for item in task.obligations}
        conflict_ids = {item.id for item in task.obligations if item.counter_evidence_ids}
        claim_ids = {item.id for item in candidate.claims}
        cited_claim_ids = {item.claim_id for item in candidate.citations}
        if not cited_claim_ids.issubset(claim_ids):
            raise ValueError(f"{key} packet citations reference unknown claims")
        _require_subset(annotation.covered_obligation_ids, obligation_ids, key, "covered obligation")
        _require_subset(annotation.conflict_handled_obligation_ids, conflict_ids, key, "conflict obligation")
        _require_subset(annotation.irrelevant_claim_ids, claim_ids, key, "irrelevant claim")
        support_partition = set(annotation.supported_claim_ids) | set(annotation.unsupported_claim_ids)
        if support_partition != claim_ids:
            raise ValueError(f"{key} must classify every claim as supported or unsupported")
        citation_partition = set(annotation.citation_supported_claim_ids) | set(annotation.citation_mismatched_claim_ids)
        if citation_partition != cited_claim_ids:
            raise ValueError(f"{key} must classify every cited claim as citation-supported or mismatched")
    return actual


def _require_subset(values: list[str], allowed: set[str], key: tuple[str, str], label: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{key} references unknown {label} ids: {unknown}")


def _validate_answer_key(packet: BlindReviewPacket, answer_key: dict[str, dict[str, str]]) -> None:
    expected = {
        task.task_id: {candidate.candidate_id for candidate in task.candidates}
        for task in packet.tasks
    }
    actual = {task_id: set(mapping) for task_id, mapping in answer_key.items()}
    if actual != expected:
        raise ValueError("answer key does not match review packet candidates")
    if any(not variant for mapping in answer_key.values() for variant in mapping.values()):
        raise ValueError("answer key contains an empty variant")
    if any(len(set(mapping.values())) != len(mapping) for mapping in answer_key.values()):
        raise ValueError("answer key maps multiple candidates to the same variant")


def _task_spec_from_blind_task(task: BlindTask) -> PilotTaskSpec:
    return PilotTaskSpec(
        id=task.task_id,
        question=task.question,
        target_user="blinded reviewer",
        decision_context=task.decision_context,
        failure_focus="coverage_gap",
        obligations=task.obligations,
        forbidden_shortcuts=task.forbidden_shortcuts,
        acceptance_notes=task.acceptance_notes,
    )


def _run_from_blind_candidate(task: BlindTask, candidate: BlindCandidate) -> RunState:
    return RunState(
        run_id=f"blind-{task.task_id}-{candidate.candidate_id}",
        task=Task(id=task.task_id, question=task.question),
        status=RunStatus.SUCCEEDED,
        evidence=[
            Evidence(
                id=item.id,
                title=item.title,
                url=item.url,
                excerpt=item.excerpt,
                query="hidden-for-blind-review",
            )
            for item in candidate.evidence
        ],
        claims=candidate.claims,
        citations=candidate.citations,
    )


def _aggregate_review_scores(scores: list[BenchmarkScore]) -> ReviewVariantAggregate:
    return ReviewVariantAggregate(
        candidate_count=len(scores),
        mean_evidence_id_recall=_mean(scores, "evidence_id_recall"),
        mean_evidence_id_precision=_mean(scores, "evidence_id_precision"),
        mean_evidence_obligation_recall=_mean(scores, "evidence_obligation_recall"),
        mean_citation_structural_integrity=_mean(scores, "citation_structural_integrity"),
        mean_obligation_coverage=_mean(scores, "obligation_coverage"),
        mean_citation_support_rate=_mean(scores, "citation_support_rate"),
        mean_unsupported_claim_rate=_mean(scores, "unsupported_claim_rate"),
        mean_irrelevant_claim_rate=_mean(scores, "irrelevant_claim_rate"),
        mean_conflict_handling_rate=_mean(scores, "conflict_handling_rate"),
    )


def _mean(scores: list[BenchmarkScore], field: str) -> float | None:
    values = [getattr(score, field) for score in scores if getattr(score, field) is not None]
    return fmean(values) if values else None


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _blind_task(task: PilotTaskSpec, candidates: list[BlindCandidate]) -> BlindTask:
    return BlindTask(
        task_id=task.id,
        question=task.question,
        decision_context=task.decision_context,
        obligations=task.obligations,
        forbidden_shortcuts=task.forbidden_shortcuts,
        acceptance_notes=task.acceptance_notes,
        candidates=candidates,
    )
