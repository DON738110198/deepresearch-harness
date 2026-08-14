from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_FRONT_MATTER = re.compile(
    r"\A---\s*\r?\n(?P<metadata>.*?)\r?\n---\s*(?:\r?\n|\Z)",
    re.DOTALL,
)
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")
_PASSAGE_BOUNDARY = re.compile(r"\r?\n\s*\r?\n")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "before",
    "between",
    "from",
    "into",
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


class QueryAwareLeadPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["query_window_v0"] = "query_window_v0"
    title: str | None = None
    date: str | None = None
    passage: str = Field(min_length=1, max_length=480)
    matched_query_terms: tuple[str, ...]


def build_query_aware_lead_preview(
    contents: str,
    query: str,
    *,
    maximum_passage_characters: int = 480,
) -> QueryAwareLeadPreview:
    if not contents.strip():
        raise ValueError("lead contents must not be blank")
    if not query.strip():
        raise ValueError("lead query must not be blank")
    if maximum_passage_characters < 40 or maximum_passage_characters > 2_000:
        raise ValueError("maximum passage characters must be between 40 and 2000")

    metadata, body = _split_front_matter(contents)
    title = _metadata_value(metadata, "title")
    date = _metadata_value(metadata, "date")
    terms = _distinct_query_terms(query)
    passages = _candidate_passages(body, maximum_passage_characters)
    best = max(
        enumerate(passages),
        key=lambda row: (_passage_score(row[1], terms), -row[0]),
    )[1]
    matched = tuple(term for term in terms if term in best.casefold())
    passage = _compact(best)[:maximum_passage_characters].rstrip()
    if not passage:
        passage = _compact(body)[:maximum_passage_characters].rstrip()
    if not passage:
        passage = _compact(contents)[:maximum_passage_characters].rstrip()
    return QueryAwareLeadPreview(
        title=title,
        date=date,
        passage=passage,
        matched_query_terms=matched,
    )


def format_query_aware_dense_lead(
    docid: str,
    preview: QueryAwareLeadPreview,
) -> str:
    if not docid.strip():
        raise ValueError("lead docid must not be blank")
    rows = [
        f"[Dense lead preview; docid={docid}; verify with open_evidence]",
    ]
    if preview.title:
        rows.append(f"Title: {preview.title}")
    if preview.date:
        rows.append(f"Date: {preview.date}")
    rows.append(f"Passage: {preview.passage}")
    return "\n".join(rows)


def query_term_overlap(query: str, text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(term for term in _distinct_query_terms(query) if term in lowered)


def _split_front_matter(contents: str) -> tuple[str, str]:
    match = _FRONT_MATTER.match(contents)
    if match is None:
        return "", contents.strip()
    return match.group("metadata"), contents[match.end() :].strip()


def _metadata_value(metadata: str, name: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*:\s*(.+?)\s*$", metadata)
    if match is None:
        return None
    value = _compact(match.group(1))
    return value or None


def _distinct_query_terms(query: str) -> tuple[str, ...]:
    seen: set[str] = set()
    terms: list[str] = []
    for match in _QUERY_TOKEN.finditer(query):
        term = match.group(0).casefold()
        if (len(term) < 3 and not term.isdigit()) or term in _STOPWORDS:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return tuple(terms)


def _candidate_passages(body: str, maximum_characters: int) -> list[str]:
    passages: list[str] = []
    for row in _PASSAGE_BOUNDARY.split(body):
        compact = _compact(row)
        if len(compact) < 20:
            continue
        if len(compact) <= maximum_characters:
            passages.append(compact)
            continue
        stride = max(maximum_characters // 2, 1)
        passages.extend(
            compact[start : start + maximum_characters]
            for start in range(0, len(compact), stride)
            if len(compact[start : start + maximum_characters].strip()) >= 20
        )
    if passages:
        return passages
    compact = _compact(body)
    return [compact] if compact else ["No non-empty passage was available."]


def _passage_score(passage: str, terms: tuple[str, ...]) -> tuple[int, int, int]:
    lowered = passage.casefold()
    matched = [term for term in terms if term in lowered]
    digit_matches = sum(term.isdigit() for term in matched)
    return len(set(matched)), digit_matches, min(len(passage), 480)


def _compact(value: str) -> str:
    return " ".join(value.split())
