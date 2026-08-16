from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_FRONT_MATTER = re.compile(r"\A---\s*\r?\n.*?\r?\n---\s*", re.DOTALL)
_ANSWER_SEPARATOR = re.compile(r"\s*(?:,|;|\||\band\b)\s*", re.IGNORECASE)
_TARGET_INTENT_MARKERS = (
    "i am seeking",
    "we are seeking",
    "what is",
    "what was",
    "what does",
    "which ",
    "who ",
    "where ",
    "when ",
    "give the name",
    "provide the name",
    "identify the",
    "name of",
)
_SECTION_ANCHOR_GROUPS = (
    ("abstract",),
    ("introduction",),
    ("method", "methods", "methodology"),
    ("result", "results"),
    ("discussion",),
    ("conclusion", "conclusions"),
    (
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
    ),
    ("reference", "references", "bibliography"),
    ("appendix",),
)
_STOPWORDS = {
    "about",
    "after",
    "also",
    "been",
    "before",
    "between",
    "does",
    "from",
    "given",
    "have",
    "into",
    "mentioned",
    "section",
    "that",
    "their",
    "there",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _normalise(value: str) -> str:
    return " ".join(_TOKEN.findall(value.casefold()))


def _terms(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in _TOKEN.findall(value.casefold())
            if (len(token) >= 3 or token.isdigit()) and token not in _STOPWORDS
        )
    )


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


SelectorId = Literal[
    "answer_obligation_window_v0",
    "answer_obligation_window_v1",
    "answer_obligation_window_v2",
]


class ArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceSpanOracleAcceptance(StrictContract):
    minimum_selected_span_hit_cases: int = Field(ge=1)
    minimum_uninspected_selected_span_hit_cases: int = Field(ge=1)
    minimum_selected_over_head_delta: int = Field(ge=1)
    missing_documents_must_equal: Literal[0] = 0


class EvidenceSpanOracleRegistration(StrictContract):
    schema_version: Literal["evidence-span-availability-oracle-registration-v0"] = (
        "evidence-span-availability-oracle-registration-v0"
    )
    status: Literal[
        "posthoc_registered_after_single_case_spotcheck",
        "posthoc_registered_for_new_failure_cluster",
    ]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    candidate_run_root: str = Field(min_length=1)
    diagnostic: ArtifactReference
    judge_summary: ArtifactReference
    gold_slice: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    previously_spotchecked_query_ids: tuple[str, ...] = Field(max_length=1)
    document_index_path: str = Field(min_length=1)
    selector_id: SelectorId = "answer_obligation_window_v0"
    maximum_span_characters: int = Field(ge=200, le=4_000)
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: EvidenceSpanOracleAcceptance
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def cases_and_gate_are_valid(self) -> "EvidenceSpanOracleRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("oracle query IDs must be unique")
        if self.acceptance.minimum_selected_span_hit_cases > len(self.query_ids):
            raise ValueError("oracle hit gate exceeds the registered case count")
        if not set(self.previously_spotchecked_query_ids).issubset(self.query_ids):
            raise ValueError("spotchecked oracle IDs must be registered cases")
        uninspected = len(self.query_ids) - len(self.previously_spotchecked_query_ids)
        if self.acceptance.minimum_uninspected_selected_span_hit_cases > uninspected:
            raise ValueError("uninspected oracle hit gate exceeds the eligible cases")
        return self


class SelectedEvidenceSpan(StrictContract):
    selector_id: SelectorId = "answer_obligation_window_v0"
    obligation_query: str = Field(min_length=1)
    content: str = Field(min_length=1)
    start_character: int = Field(ge=0)
    end_character: int = Field(gt=0)
    matched_section_anchors: tuple[str, ...] = ()
    matched_obligation_terms: tuple[str, ...]
    matched_question_terms: tuple[str, ...]
    score: float = Field(allow_inf_nan=False)


class AnswerCoverage(StrictContract):
    atoms: tuple[str, ...] = Field(min_length=1)
    matched_atoms: tuple[str, ...]
    coverage: float = Field(ge=0.0, le=1.0)
    all_atoms_present: bool


class DocumentSpanResult(StrictContract):
    docid: str = Field(min_length=1)
    document_characters: int = Field(ge=0)
    head_span: str | None
    selected_span: SelectedEvidenceSpan | None
    missing: bool


class EvidenceSpanOracleCaseResult(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    retrieved_gold_docids: tuple[str, ...] = Field(min_length=1)
    documents: tuple[DocumentSpanResult, ...] = Field(min_length=1)
    full_document_coverage: AnswerCoverage
    head_span_coverage: AnswerCoverage
    selected_span_coverage: AnswerCoverage


class DecisionGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int | float
    operator: Literal["eq", "ge"]
    threshold: int | float
    passed: bool


class EvidenceSpanOracleResult(StrictContract):
    schema_version: Literal["evidence-span-availability-oracle-v0"] = (
        "evidence-span-availability-oracle-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded", "failed"]
    decision: Literal["pass", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    document_open_count: int = Field(ge=0)
    missing_documents: int = Field(ge=0)
    full_document_hit_cases: int = Field(ge=0)
    head_span_hit_cases: int = Field(ge=0)
    selected_span_hit_cases: int = Field(ge=0)
    uninspected_selected_span_hit_cases: int = Field(ge=0)
    selected_over_head_delta: int
    provider_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    items: tuple[EvidenceSpanOracleCaseResult, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "preregister_obligation_conditioned_span_opening",
        "reject_span_opening_and_rediagnose",
    ]
    claim_boundary: str = Field(min_length=1)


def derive_answer_obligation_query(question: str) -> str:
    compact = " ".join(question.split())
    if not compact:
        raise ValueError("question must not be blank")
    sentences = [row.strip() for row in _SENTENCE.split(compact) if row.strip()]
    interrogatives = [row for row in sentences if "?" in row]
    return (interrogatives[-1] if interrogatives else sentences[-1]).rstrip("?").strip()


def derive_answer_obligation_query_v1(question: str) -> str:
    compact = " ".join(question.split())
    if not compact:
        raise ValueError("question must not be blank")
    sentences = [row.strip() for row in _SENTENCE.split(compact) if row.strip()]
    interrogatives = [row for row in sentences if "?" in row]
    if interrogatives:
        return interrogatives[-1].rstrip("?").strip()
    scored = [
        (
            sum(marker in sentence.casefold() for marker in _TARGET_INTENT_MARKERS),
            -index,
            sentence,
        )
        for index, sentence in enumerate(sentences)
    ]
    score, _negative_index, selected = max(scored)
    return (selected if score else sentences[-1]).rstrip("?").strip()


def select_answer_obligation_span(
    contents: str,
    question: str,
    *,
    maximum_span_characters: int = 2_000,
    selector_id: SelectorId = "answer_obligation_window_v0",
) -> SelectedEvidenceSpan:
    if not contents.strip():
        raise ValueError("document contents must not be blank")
    if maximum_span_characters < 200 or maximum_span_characters > 4_000:
        raise ValueError("maximum span characters must be between 200 and 4000")
    body_start = 0
    front_matter = _FRONT_MATTER.match(contents)
    if front_matter is not None:
        body_start = front_matter.end()
    obligation = (
        derive_answer_obligation_query_v1(question)
        if selector_id in {
            "answer_obligation_window_v1",
            "answer_obligation_window_v2",
        }
        else derive_answer_obligation_query(question)
    )
    obligation_terms = _terms(obligation)
    question_terms = _terms(question)
    section_anchors = (
        _section_anchor_aliases(obligation)
        if selector_id == "answer_obligation_window_v2"
        else ()
    )
    if not obligation_terms:
        obligation_terms = question_terms

    candidates = _window_candidates(contents, body_start, maximum_span_characters)
    start, end, _unused_score = max(
        candidates,
        key=lambda row: (
            len(_matched_phrases(contents[row[0] : row[1]], section_anchors)),
            _span_score(contents[row[0] : row[1]], obligation_terms, question_terms),
            -row[0],
        ),
    )
    content = " ".join(contents[start:end].split())
    if not content:
        raise ValueError("selected evidence span is blank")
    lowered = content.casefold()
    matched_section_anchors = _matched_phrases(content, section_anchors)
    matched_obligation = tuple(term for term in obligation_terms if term in lowered)
    matched_question = tuple(term for term in question_terms if term in lowered)
    score = _span_score(content, obligation_terms, question_terms)
    return SelectedEvidenceSpan(
        selector_id=selector_id,
        obligation_query=obligation,
        content=content,
        start_character=start,
        end_character=end,
        matched_section_anchors=matched_section_anchors,
        matched_obligation_terms=matched_obligation,
        matched_question_terms=matched_question,
        score=score,
    )


def _section_anchor_aliases(value: str) -> tuple[str, ...]:
    normalised = f" {_normalise(value)} "
    return tuple(
        alias
        for group in _SECTION_ANCHOR_GROUPS
        if any(f" {_normalise(candidate)} " in normalised for candidate in group)
        for alias in group
    )


def _matched_phrases(value: str, phrases: Sequence[str]) -> tuple[str, ...]:
    normalised = f" {_normalise(value)} "
    return tuple(
        phrase
        for phrase in phrases
        if f" {_normalise(phrase)} " in normalised
    )


def answer_coverage(correct_answer: str, text: str) -> AnswerCoverage:
    raw_atoms = [row.strip() for row in _ANSWER_SEPARATOR.split(correct_answer) if row.strip()]
    atoms = tuple(raw_atoms if len(raw_atoms) > 1 else [correct_answer.strip()])
    normalised_text = _normalise(text)
    matched = tuple(atom for atom in atoms if _normalise(atom) in normalised_text)
    coverage = len(matched) / len(atoms)
    return AnswerCoverage(
        atoms=atoms,
        matched_atoms=matched,
        coverage=coverage,
        all_atoms_present=len(matched) == len(atoms),
    )


def load_evidence_span_oracle_registration(
    path: Path,
) -> EvidenceSpanOracleRegistration:
    registration = EvidenceSpanOracleRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            registration.diagnostic,
            registration.judge_summary,
            registration.gold_slice,
            *registration.frozen_artifacts,
        ),
    )
    run_root = (root / registration.candidate_run_root).resolve()
    if not run_root.is_relative_to(root) or not run_root.is_dir():
        raise ValueError("candidate run root is missing or escapes the repository")
    index_path = (root / registration.document_index_path).resolve()
    if not index_path.is_relative_to(root) or not index_path.is_dir():
        raise ValueError("document index is missing or escapes the repository")
    return registration


def run_evidence_span_oracle(
    *,
    registration_path: Path,
    output_path: Path,
    document_loader: Callable[[str], str | None],
) -> EvidenceSpanOracleResult:
    registration = load_evidence_span_oracle_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("oracle output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("oracle output already exists")

    diagnostic = json.loads((root / registration.diagnostic.path).read_text(encoding="utf-8"))
    judge = json.loads((root / registration.judge_summary.path).read_text(encoding="utf-8"))
    gold = json.loads((root / registration.gold_slice.path).read_text(encoding="utf-8"))
    diagnostic_by_id = {str(row["query_id"]): row for row in diagnostic["rows"]}
    judge_by_id = {str(row["query_id"]): row for row in judge["observations"]}
    gold_by_id = {str(row["query_id"]): row for row in gold["rows"]}

    items: list[EvidenceSpanOracleCaseResult] = []
    missing_documents = 0
    document_open_count = 0
    for query_id in registration.query_ids:
        if query_id not in diagnostic_by_id or query_id not in judge_by_id or query_id not in gold_by_id:
            raise ValueError(f"registered oracle case is absent from a source: {query_id}")
        diagnostic_row = diagnostic_by_id[query_id]
        judge_row = judge_by_id[query_id]
        gold_row = gold_by_id[query_id]
        if judge_row["correct"]:
            raise ValueError(f"oracle case was not a Judge failure: {query_id}")
        if float(diagnostic_row["gold_recall"]) <= 0:
            raise ValueError(f"oracle case has no retrieved gold document: {query_id}")

        run_path = root / registration.candidate_run_root / query_id / "run.json"
        if not run_path.is_file():
            raise ValueError(f"candidate run is missing: {run_path}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        question = _extract_question(run)
        result_path = (
            root
            / registration.judge_summary.path
        ).parent / "results" / f"{query_id}_eval.json"
        evaluation = json.loads(result_path.read_text(encoding="utf-8"))
        correct_answer = str(evaluation["correct_answer"])
        retrieved_docids = {
            str(result["docid"])
            for call in run["search_calls"]
            for result in call["results"]
        }
        retrieved_gold_docids = tuple(
            str(docid)
            for docid in gold_row["gold_docids"]
            if str(docid) in retrieved_docids
        )
        if not retrieved_gold_docids:
            raise ValueError(f"oracle run trace has no retrieved gold document: {query_id}")

        documents: list[DocumentSpanResult] = []
        full_texts: list[str] = []
        head_spans: list[str] = []
        selected_spans: list[str] = []
        for docid in retrieved_gold_docids:
            document_open_count += 1
            contents = document_loader(docid)
            if contents is None or not contents.strip():
                missing_documents += 1
                documents.append(
                    DocumentSpanResult(
                        docid=docid,
                        document_characters=0,
                        head_span=None,
                        selected_span=None,
                        missing=True,
                    )
                )
                continue
            head = " ".join(contents[: registration.maximum_span_characters].split())
            selected = select_answer_obligation_span(
                contents,
                question,
                maximum_span_characters=registration.maximum_span_characters,
                selector_id=registration.selector_id,
            )
            full_texts.append(contents)
            head_spans.append(head)
            selected_spans.append(selected.content)
            documents.append(
                DocumentSpanResult(
                    docid=docid,
                    document_characters=len(contents),
                    head_span=head,
                    selected_span=selected,
                    missing=False,
                )
            )
        items.append(
            EvidenceSpanOracleCaseResult(
                query_id=query_id,
                question=question,
                correct_answer=correct_answer,
                retrieved_gold_docids=retrieved_gold_docids,
                documents=tuple(documents),
                full_document_coverage=answer_coverage(correct_answer, "\n".join(full_texts)),
                head_span_coverage=answer_coverage(correct_answer, "\n".join(head_spans)),
                selected_span_coverage=answer_coverage(correct_answer, "\n".join(selected_spans)),
            )
        )

    full_hits = sum(item.full_document_coverage.all_atoms_present for item in items)
    head_hits = sum(item.head_span_coverage.all_atoms_present for item in items)
    selected_hits = sum(item.selected_span_coverage.all_atoms_present for item in items)
    spotchecked = set(registration.previously_spotchecked_query_ids)
    uninspected_selected_hits = sum(
        item.selected_span_coverage.all_atoms_present
        for item in items
        if item.query_id not in spotchecked
    )
    delta = selected_hits - head_hits
    acceptance = registration.acceptance
    gates = (
        DecisionGate(
            gate_id="missing_documents",
            observed=missing_documents,
            operator="eq",
            threshold=acceptance.missing_documents_must_equal,
            passed=missing_documents == acceptance.missing_documents_must_equal,
        ),
        DecisionGate(
            gate_id="selected_span_hit_cases",
            observed=selected_hits,
            operator="ge",
            threshold=acceptance.minimum_selected_span_hit_cases,
            passed=selected_hits >= acceptance.minimum_selected_span_hit_cases,
        ),
        DecisionGate(
            gate_id="uninspected_selected_span_hit_cases",
            observed=uninspected_selected_hits,
            operator="ge",
            threshold=acceptance.minimum_uninspected_selected_span_hit_cases,
            passed=(
                uninspected_selected_hits
                >= acceptance.minimum_uninspected_selected_span_hit_cases
            ),
        ),
        DecisionGate(
            gate_id="selected_over_head_delta",
            observed=delta,
            operator="ge",
            threshold=acceptance.minimum_selected_over_head_delta,
            passed=delta >= acceptance.minimum_selected_over_head_delta,
        ),
    )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = EvidenceSpanOracleResult(
        created_at=_utc_now(),
        status="succeeded" if missing_documents == 0 else "failed",
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=len(items),
        document_open_count=document_open_count,
        missing_documents=missing_documents,
        full_document_hit_cases=full_hits,
        head_span_hit_cases=head_hits,
        selected_span_hit_cases=selected_hits,
        uninspected_selected_span_hit_cases=uninspected_selected_hits,
        selected_over_head_delta=delta,
        items=tuple(items),
        gates=gates,
        next_action=(
            "preregister_obligation_conditioned_span_opening"
            if decision == "pass"
            else "reject_span_opening_and_rediagnose"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _window_candidates(
    contents: str, body_start: int, maximum_span_characters: int
) -> tuple[tuple[int, int, float], ...]:
    body_length = len(contents) - body_start
    if body_length <= maximum_span_characters:
        return ((body_start, len(contents), 0.0),)
    stride = max(maximum_span_characters // 2, 1)
    starts = list(range(body_start, len(contents), stride))
    final_start = max(len(contents) - maximum_span_characters, body_start)
    if final_start not in starts:
        starts.append(final_start)
    return tuple(
        (start, min(start + maximum_span_characters, len(contents)), 0.0)
        for start in starts
        if contents[start : min(start + maximum_span_characters, len(contents))].strip()
    )


def _span_score(
    span: str,
    obligation_terms: Sequence[str],
    question_terms: Sequence[str],
) -> float:
    lowered = span.casefold()
    obligation_matches = tuple(term for term in obligation_terms if term in lowered)
    question_matches = tuple(term for term in question_terms if term in lowered)
    obligation_bigrams = sum(
        f"{left} {right}" in _normalise(span)
        for left, right in zip(obligation_terms, obligation_terms[1:])
    )
    rare_term_weight = sum(min(len(term), 12) for term in obligation_matches)
    digit_matches = sum(term.isdigit() for term in obligation_matches)
    return (
        len(set(obligation_matches)) * 100
        + obligation_bigrams * 30
        + rare_term_weight
        + digit_matches * 20
        + len(set(question_matches)) * 3
    )


def _extract_question(run: dict[str, object]) -> str:
    messages = run.get("messages")
    if not isinstance(messages, list):
        raise ValueError("run trace has no messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = str(part.get("text", ""))
            marker = "Question:"
            if marker in text:
                question = text.split(marker, 1)[1].split(
                    "Your response should be", 1
                )[0].strip()
                if question:
                    return question
    raise ValueError("could not extract question from run trace")


def _validate_artifacts(root: Path, artifacts: Sequence[ArtifactReference]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")
