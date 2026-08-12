from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl

from .batch import BatchSummary
from .benchmark import AnswerObligation, PilotTaskSpec, load_suite
from .contracts import Citation, Claim, RunState


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


class CandidateAnnotationTemplate(BaseModel):
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
    templates: list[CandidateAnnotationTemplate] = []
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
            templates.append(CandidateAnnotationTemplate(task_id=task_id, candidate_id=candidate_id))
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
    template_path.write_text(json.dumps([item.model_dump() for item in templates], indent=2), encoding="utf-8")
    key_path.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")
    return packet_path, template_path, key_path


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
