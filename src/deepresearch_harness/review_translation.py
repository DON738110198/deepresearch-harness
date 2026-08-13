from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .contracts import TraceEvent, Usage, utc_now
from .providers import LLMProvider
from .review import BlindReviewPacket, file_sha256


class ReviewTranslationEntry(BaseModel):
    id: str = Field(pattern=r"^text-\d{4}$")
    source: str = Field(min_length=1)
    translated: str = Field(min_length=1)


class ReviewTranslationBundle(BaseModel):
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locale: Literal["zh-CN"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    entries: list[ReviewTranslationEntry] = Field(min_length=1)
    trace: list[TraceEvent] = Field(default_factory=list)
    total_usage: Usage = Field(default_factory=Usage)

    @model_validator(mode="after")
    def entries_are_unique(self) -> "ReviewTranslationBundle":
        ids = [item.id for item in self.entries]
        sources = [item.source for item in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("translation entry ids must be unique")
        if len(sources) != len(set(sources)):
            raise ValueError("translation entry sources must be unique")
        return self


class _TranslationResult(BaseModel):
    id: str
    text: str = Field(min_length=1)


class _TranslationResponse(BaseModel):
    translations: list[_TranslationResult]


def translate_review_packet(
    *,
    packet_path: Path,
    output_path: Path,
    provider: LLMProvider,
    max_chunk_characters: int = 6_000,
    max_output_tokens: int = 8_192,
) -> ReviewTranslationBundle:
    if max_chunk_characters < 500:
        raise ValueError("max_chunk_characters must be at least 500")
    packet = BlindReviewPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    sources = review_source_texts(packet)
    indexed = [
        {"id": f"text-{index:04d}", "source": source}
        for index, source in enumerate(sources, start=1)
    ]
    translated_by_id: dict[str, str] = {}
    trace: list[TraceEvent] = []
    total_usage = Usage()
    for chunk_index, chunk in enumerate(_chunks(indexed, max_chunk_characters), start=1):
        prompt = json.dumps(
            {
                "instruction": (
                    "Translate every source string into clear Simplified Chinese for a blinded human reviewer. "
                    "Preserve meaning and evidentiary strength exactly. Preserve Markdown structure, citation "
                    "markers such as [1], claim and evidence IDs, URLs, numbers, and proper nouns. Do not add, "
                    "remove, weaken, strengthen, explain, or summarize content. Return JSON only."
                ),
                "locale": "zh-CN",
                "items": chunk,
                "response_contract": {
                    "translations": [
                        {"id": "same id as input", "text": "translated Simplified Chinese text"}
                    ]
                },
            },
            ensure_ascii=False,
        )
        completion = provider.complete(
            stage="review_translate",
            prompt=prompt,
            json_output=True,
            max_output_tokens=max_output_tokens,
        )
        response = _TranslationResponse.model_validate(_load_json_object(completion.text))
        expected_ids = {item["id"] for item in chunk}
        actual_ids = [item.id for item in response.translations]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
            raise ValueError(
                f"translation response ids do not match chunk {chunk_index}; "
                f"expected={sorted(expected_ids)}, actual={sorted(actual_ids)}"
            )
        source_by_id = {item["id"]: item["source"] for item in chunk}
        for item in response.translations:
            _validate_protected_content(source_by_id[item.id], item.text, item.id)
        translated_by_id.update({item.id: item.text for item in response.translations})
        trace.append(
            TraceEvent(
                stage="review_translate",
                provider=provider.name,
                model=provider.model,
                latency_ms=completion.latency_ms,
                usage=completion.usage,
                outcome="ok",
                detail=f"chunk={chunk_index},entries={len(chunk)}",
            )
        )
        _add_usage(total_usage, completion.usage)

    entries = [
        ReviewTranslationEntry(
            id=item["id"],
            source=item["source"],
            translated=translated_by_id[item["id"]],
        )
        for item in indexed
    ]
    bundle = ReviewTranslationBundle(
        packet_sha256=file_sha256(packet_path),
        locale="zh-CN",
        provider=provider.name,
        model=provider.model,
        entries=entries,
        trace=trace,
        total_usage=total_usage,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    return bundle


def load_review_translation_bundle(
    *,
    packet_path: Path,
    translations_path: Path,
) -> ReviewTranslationBundle:
    packet = BlindReviewPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    bundle = ReviewTranslationBundle.model_validate_json(translations_path.read_text(encoding="utf-8"))
    packet_hash = file_sha256(packet_path)
    if bundle.packet_sha256 != packet_hash:
        raise ValueError("translation bundle packet_sha256 does not match review packet")
    expected_sources = set(review_source_texts(packet))
    actual_sources = {item.source for item in bundle.entries}
    missing = expected_sources - actual_sources
    extra = actual_sources - expected_sources
    if missing or extra:
        raise ValueError(
            f"translation bundle source set mismatch; missing={len(missing)}, extra={len(extra)}"
        )
    for item in bundle.entries:
        _validate_protected_content(item.source, item.translated, item.id)
    return bundle


def review_source_texts(packet: BlindReviewPacket) -> list[str]:
    texts: list[str] = []

    def add(value: str) -> None:
        if value and value not in texts:
            texts.append(value)

    for item in packet.rubric:
        add(item)
    for task in packet.tasks:
        add(task.question)
        add(task.decision_context)
        for obligation in task.obligations:
            add(obligation.description)
        for item in task.forbidden_shortcuts:
            add(item)
        for item in task.acceptance_notes:
            add(item)
        for candidate in task.candidates:
            add(candidate.report)
            for evidence in candidate.evidence:
                add(evidence.title)
                add(evidence.excerpt)
            for claim in candidate.claims:
                add(claim.text)
    return texts


def _chunks(items: list[dict[str, str]], character_limit: int) -> list[list[dict[str, str]]]:
    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_characters = 0
    for item in items:
        item_characters = len(item["source"])
        if current and current_characters + item_characters > character_limit:
            chunks.append(current)
            current = []
            current_characters = 0
        current.append(item)
        current_characters += item_characters
    if current:
        chunks.append(current)
    return chunks


def _load_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("translation response must be a JSON object")
    return value


def _add_usage(total: Usage, value: Usage) -> None:
    total.input_tokens += value.input_tokens
    total.output_tokens += value.output_tokens
    total.input_cache_hit_tokens += value.input_cache_hit_tokens
    total.input_cache_miss_tokens += value.input_cache_miss_tokens
    total.estimated_cost_usd += value.estimated_cost_usd


def _validate_protected_content(source: str, translated: str, entry_id: str) -> None:
    patterns = (
        r"\[[0-9]+\]",
        r"https?://[^\s)]+",
        r"\b(?:claim-|t\d{2}-)[A-Za-z0-9._:-]+",
    )
    for pattern in patterns:
        if re.findall(pattern, source) != re.findall(pattern, translated):
            raise ValueError(f"translation {entry_id} changed a protected citation, URL, or identifier")
