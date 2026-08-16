from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import PiBrowseCompRun, load_pi_browsecomp_run


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


DebtReason = Literal[
    "answer_schema_missing",
    "missing_citations",
    "cited_evidence_does_not_support_exact_answer",
    "explicit_unresolved_evidence",
]


class EvidenceDebtAudit(StrictContract):
    schema_version: Literal["saved-trace-evidence-debt-audit-v0"] = (
        "saved-trace-evidence-debt-audit-v0"
    )
    query_id: str = Field(min_length=1)
    source_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_answer: str | None
    subject_hypothesis: str | None
    confidence_percent: int | None = Field(default=None, ge=0, le=100)
    cited_docids: tuple[str, ...]
    supporting_cited_docids: tuple[str, ...]
    supporting_all_docids: tuple[str, ...]
    explicit_uncertainty_phrases: tuple[str, ...]
    status: Literal["supported", "no_repair_trigger", "open", "unscorable"]
    reasons: tuple[DebtReason, ...]
    repair_queries: tuple[str, ...] = Field(max_length=2)
    provider_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    claim_boundary: Literal[
        "Deterministic saved-trace diagnostic; not benchmark effectiveness evidence."
    ] = "Deterministic saved-trace diagnostic; not benchmark effectiveness evidence."

    @model_validator(mode="after")
    def status_matches_debt(self) -> "EvidenceDebtAudit":
        if self.status in {"supported", "no_repair_trigger"} and (
            self.reasons or self.repair_queries
        ):
            raise ValueError("closed audit status cannot carry reasons or repair queries")
        if self.status == "supported" and not self.supporting_cited_docids:
            raise ValueError("supported status requires directly supporting cited evidence")
        if self.status == "no_repair_trigger" and self.supporting_cited_docids:
            raise ValueError("no-repair status is reserved for conservative abstention")
        if self.status == "open" and (not self.reasons or not self.repair_queries):
            raise ValueError("open debt requires reasons and at least one repair query")
        if self.status == "unscorable" and self.reasons != ("answer_schema_missing",):
            raise ValueError("unscorable debt must be caused by a missing answer schema")
        if not set(self.supporting_cited_docids).issubset(self.cited_docids):
            raise ValueError("supporting cited docids must be cited")
        if not set(self.supporting_cited_docids).issubset(self.supporting_all_docids):
            raise ValueError("cited support must also be present in all support")
        return self


_EXACT_ANSWER = re.compile(r"(?im)^\s*Exact Answer\s*:\s*(?P<answer>.+?)\s*$")
_CONFIDENCE = re.compile(r"(?im)^\s*Confidence\s*:\s*(?P<value>\d{1,3})\s*%")
_CITATION = re.compile(r"\[(?P<docid>\d+)\]")
_QUESTION = re.compile(
    r"(?s)\bQuestion:\s*(?P<question>.+?)\n\nYour response should be",
)
_SUBJECT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:the\s+)?clues?\s+point(?:s)?\s+to\s+(?P<subject>[^,.\n]+)",
        r"\bmost fitting match is\s+(?P<subject>[^,.\n]+)",
        r"\b(?:the\s+)?(?:author|person|athlete|runner|kingdom|band)\s+is\s+(?P<subject>[^,.\n]+)",
    )
)
_UNCERTAINTY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"not (?:fully|directly|definitively) confirm(?:ed)?",
        r"does not directly confirm",
        r"did not yield a definitive",
        r"incomplete evidence",
        r"confidence is limited",
        r"cannot (?:be )?confirm(?:ed)?",
        r"unverified",
        r"unsupported by the available evidence",
    )
)
_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "his",
        "in",
        "of",
        "on",
        "or",
        "sir",
        "the",
        "their",
        "to",
        "with",
    }
)
_RELATION_KEYWORDS = (
    "foundation",
    "founded",
    "established",
    "adopted",
    "partner",
    "dismissed",
    "practice",
    "knighted",
    "knighthood",
    "died",
    "death",
    "honor",
    "honour",
    "award",
    "race",
    "title",
    "subtitle",
    "year",
)


def audit_saved_trace(path: Path) -> EvidenceDebtAudit:
    raw = path.read_bytes()
    run = load_pi_browsecomp_run(path)
    return audit_pi_run(run, source_run_sha256=sha256(raw).hexdigest())


def audit_pi_run(
    run: PiBrowseCompRun,
    *,
    source_run_sha256: str,
) -> EvidenceDebtAudit:
    question = _extract_question(run.messages)
    exact_match = _EXACT_ANSWER.search(run.answer_text)
    confidence_match = _CONFIDENCE.search(run.answer_text)
    cited_docids = tuple(
        sorted({match.group("docid") for match in _CITATION.finditer(run.answer_text)})
    )
    evidence = _collect_evidence(run)

    if exact_match is None:
        return EvidenceDebtAudit(
            query_id=run.query_id,
            source_run_sha256=source_run_sha256,
            question_sha256=_text_sha256(question),
            answer_sha256=_text_sha256(run.answer_text),
            exact_answer=None,
            subject_hypothesis=None,
            confidence_percent=(
                int(confidence_match.group("value")) if confidence_match else None
            ),
            cited_docids=cited_docids,
            supporting_cited_docids=(),
            supporting_all_docids=(),
            explicit_uncertainty_phrases=(),
            status="unscorable",
            reasons=("answer_schema_missing",),
            repair_queries=(),
        )

    exact_answer = exact_match.group("answer").strip()
    confidence = int(confidence_match.group("value")) if confidence_match else None
    subject = _extract_subject(run.answer_text)
    supporting_all = tuple(
        sorted(
            docid
            for docid, text in evidence.items()
            if _supports_answer_claim(
                text,
                exact_answer=exact_answer,
                subject=subject,
                question=question,
            )
        )
    )
    supporting_cited = tuple(
        docid for docid in supporting_all if docid in set(cited_docids)
    )
    uncertainty = tuple(
        dict.fromkeys(
            match.group(0).strip()
            for pattern in _UNCERTAINTY_PATTERNS
            for match in pattern.finditer(run.answer_text)
        )
    )

    reasons: list[DebtReason] = []
    if not cited_docids:
        reasons.append("missing_citations")
    elif not supporting_cited and (uncertainty or confidence is None or confidence <= 50):
        reasons.append("cited_evidence_does_not_support_exact_answer")
    if uncertainty:
        reasons.append("explicit_unresolved_evidence")

    status: Literal["supported", "no_repair_trigger", "open"]
    if reasons:
        status = "open"
    elif supporting_cited:
        status = "supported"
    else:
        status = "no_repair_trigger"
    repair_queries = (
        _build_repair_queries(
            question=question,
            answer_text=run.answer_text,
            exact_answer=exact_answer,
            subject=subject,
        )
        if status == "open"
        else ()
    )
    return EvidenceDebtAudit(
        query_id=run.query_id,
        source_run_sha256=source_run_sha256,
        question_sha256=_text_sha256(question),
        answer_sha256=_text_sha256(run.answer_text),
        exact_answer=exact_answer,
        subject_hypothesis=subject,
        confidence_percent=confidence,
        cited_docids=cited_docids,
        supporting_cited_docids=supporting_cited,
        supporting_all_docids=supporting_all,
        explicit_uncertainty_phrases=uncertainty,
        status=status,
        reasons=tuple(reasons),
        repair_queries=repair_queries,
    )


def _extract_question(messages: list[dict[str, object]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if not isinstance(text, str):
                continue
            match = _QUESTION.search(text)
            if match:
                return match.group("question").strip()
    raise ValueError("saved Pi trace does not contain the benchmark question")


def _extract_subject(answer_text: str) -> str | None:
    for pattern in _SUBJECT_PATTERNS:
        match = pattern.search(answer_text)
        if match:
            subject = re.sub(r"\s+", " ", match.group("subject")).strip(" *_'\"")
            return subject or None
    return None


def _collect_evidence(run: PiBrowseCompRun) -> dict[str, str]:
    chunks: dict[str, list[str]] = {}
    for call in run.search_calls:
        for result in call.results:
            chunks.setdefault(result.docid, []).append(result.snippet)
    for call in run.evidence_open_calls:
        if call.result is not None and call.result.content:
            chunks.setdefault(call.docid, []).append(call.result.content)
    return {docid: "\n".join(values) for docid, values in chunks.items()}


def _supports_answer_claim(
    text: str,
    *,
    exact_answer: str,
    subject: str | None,
    question: str,
) -> bool:
    normalized_text = _normalize(text)
    numeric_answer = re.fullmatch(r"\d{4}", exact_answer.strip())
    if numeric_answer:
        if not re.search(rf"(?<!\d){re.escape(exact_answer.strip())}(?!\d)", text):
            return False
        if subject and not _alias_supported(normalized_text, subject, minimum_coverage=0.5):
            return False
        relation_terms = _present_relation_terms(question)
        return not relation_terms or any(term in normalized_text for term in relation_terms)
    return _alias_supported(normalized_text, exact_answer, minimum_coverage=0.8)


def _alias_supported(normalized_text: str, value: str, *, minimum_coverage: float) -> bool:
    aliases = [value]
    parenthetical = re.search(r"\(([^)]+)\)", value)
    if parenthetical:
        aliases.extend((value[: parenthetical.start()].strip(), parenthetical.group(1).strip()))
    for alias in aliases:
        normalized_alias = _normalize(alias)
        if normalized_alias and normalized_alias in normalized_text:
            return True
        tokens = _content_tokens(alias)
        if len(tokens) < 2:
            continue
        present = sum(token in normalized_text.split() for token in tokens)
        if present / len(tokens) >= minimum_coverage:
            return True
    return False


def _build_repair_queries(
    *,
    question: str,
    answer_text: str,
    exact_answer: str,
    subject: str | None,
) -> tuple[str, ...]:
    anchor = subject or exact_answer
    relation_terms = _present_relation_terms(f"{question}\n{answer_text}")[:5]
    if re.fullmatch(r"\d{4}", exact_answer.strip()):
        first = " ".join((f'"{anchor}"', *relation_terms))
        second = " ".join((f'"{anchor}"', exact_answer.strip(), *relation_terms[:3]))
    else:
        first = f'"{exact_answer}" biography'
        second = " ".join((f'"{exact_answer}"', *relation_terms))
    return tuple(dict.fromkeys(query.strip() for query in (first, second) if query.strip()))[:2]


def _present_relation_terms(text: str) -> tuple[str, ...]:
    normalized = _normalize(text)
    return tuple(term for term in _RELATION_KEYWORDS if term in normalized)


def _content_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in _WORD.findall(value.casefold())
            if len(token) > 1 and token not in _STOP_WORDS
        )
    )


def _normalize(value: str) -> str:
    return " ".join(_WORD.findall(value.casefold()))


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
