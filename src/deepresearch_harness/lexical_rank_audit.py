from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch_harness.evidence_span_oracle import ArtifactReference


SearchFunction = Callable[[str, int], Sequence[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LexicalRankAcceptance(StrictContract):
    minimum_raw_question_top5_cases: int = Field(ge=1)
    minimum_generated_query_top100_cases_for_pool_diagnosis: int = Field(ge=1)


class LexicalRankRegistration(StrictContract):
    schema_version: Literal["persistent-miss-lexical-rank-registration-v0"] = (
        "persistent-miss-lexical-rank-registration-v0"
    )
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    stability_audit: ArtifactReference
    corpus_answerability: ArtifactReference
    gold_slice: ArtifactReference
    baseline_run_root: str = Field(min_length=1)
    query_ids: tuple[str, ...] = Field(min_length=1)
    document_index_path: str = Field(min_length=1)
    maximum_rank: int = Field(ge=100, le=10_000)
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: LexicalRankAcceptance
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def gate_is_valid(self) -> "LexicalRankRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("lexical-rank query IDs must be unique")
        for value in (
            self.acceptance.minimum_raw_question_top5_cases,
            self.acceptance.minimum_generated_query_top100_cases_for_pool_diagnosis,
        ):
            if value > len(self.query_ids):
                raise ValueError("lexical-rank gate exceeds registered cases")
        return self


class QueryRankObservation(StrictContract):
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_rank: int | None = Field(default=None, ge=1)
    matched_gold_docid: str | None = None
    top5_docids: tuple[str, ...] = Field(max_length=5)


class LexicalRankCase(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    raw_question: QueryRankObservation
    generated_queries: tuple[QueryRankObservation, ...] = Field(min_length=1)
    best_generated_gold_rank: int | None = Field(default=None, ge=1)
    raw_question_better_than_generated: bool


class LexicalRankGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int
    operator: Literal["ge"] = "ge"
    threshold: int
    passed: bool


class LexicalRankResult(StrictContract):
    schema_version: Literal["persistent-miss-lexical-rank-v0"] = (
        "persistent-miss-lexical-rank-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded"] = "succeeded"
    decision: Literal[
        "raw_question_anchor_candidate",
        "bm25_pool_or_rerank_candidate",
        "index_representation_diagnosis_required",
    ]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    raw_question_top5_cases: int = Field(ge=0)
    raw_question_top20_cases: int = Field(ge=0)
    raw_question_top100_cases: int = Field(ge=0)
    raw_question_top1000_cases: int = Field(ge=0)
    generated_query_top5_cases: int = Field(ge=0)
    generated_query_top20_cases: int = Field(ge=0)
    generated_query_top100_cases: int = Field(ge=0)
    generated_query_top1000_cases: int = Field(ge=0)
    raw_question_better_cases: int = Field(ge=0)
    offline_bm25_queries: int = Field(ge=0)
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[LexicalRankCase, ...]
    gates: tuple[LexicalRankGate, ...]
    next_action: Literal[
        "preregister_raw_question_anchor_retrieval_gate",
        "diagnose_bm25_candidate_pool_and_reranking",
        "diagnose_index_representation",
    ]
    claim_boundary: str = Field(min_length=1)


def load_lexical_rank_registration(path: Path) -> LexicalRankRegistration:
    registration = LexicalRankRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            registration.stability_audit,
            registration.corpus_answerability,
            registration.gold_slice,
            *registration.frozen_artifacts,
        ),
    )
    for relative, label in (
        (registration.baseline_run_root, "baseline run root"),
        (registration.document_index_path, "document index"),
    ):
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            raise ValueError(f"lexical-rank {label} is missing or escapes root")
    stability = json.loads(
        (root / registration.stability_audit.path).read_text(encoding="utf-8")
    )
    expected_ids = {
        str(row["query_id"])
        for row in stability["cases"]
        if row["category"] == "persistent_retrieval_miss"
    }
    if set(registration.query_ids) != expected_ids:
        raise ValueError("registered cases differ from the persistent-miss cluster")
    answerability = json.loads(
        (root / registration.corpus_answerability.path).read_text(encoding="utf-8")
    )
    if (
        answerability.get("decision") != "retrieval_layer_confirmed"
        or answerability.get("answerable_cases") != len(registration.query_ids)
    ):
        raise ValueError("lexical-rank prerequisite did not confirm retrieval layer")
    return registration


def run_lexical_rank_audit(
    *,
    registration_path: Path,
    output_path: Path,
    search: SearchFunction,
) -> LexicalRankResult:
    registration = load_lexical_rank_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("lexical-rank output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("lexical-rank output already exists")
    gold = json.loads((root / registration.gold_slice.path).read_text(encoding="utf-8"))
    gold_by_id = {str(row["query_id"]): row for row in gold["rows"]}

    items: list[LexicalRankCase] = []
    offline_queries = 0
    for query_id in registration.query_ids:
        gold_row = gold_by_id.get(query_id)
        if gold_row is None:
            raise ValueError(f"lexical-rank gold case is missing: {query_id}")
        run_path = root / registration.baseline_run_root / query_id / "run.json"
        if not run_path.is_file():
            raise ValueError(f"lexical-rank baseline run is missing: {query_id}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        question = str(gold_row["question"])
        if _compact(_extract_question(run)) != _compact(question):
            raise ValueError(f"lexical-rank run question differs from gold: {query_id}")
        gold_docids = tuple(str(value) for value in gold_row["gold_docids"])
        raw = _rank_query(
            question,
            gold_docids,
            maximum_rank=registration.maximum_rank,
            search=search,
        )
        offline_queries += 1
        generated: list[QueryRankObservation] = []
        search_calls = run.get("search_calls")
        if not isinstance(search_calls, list) or not search_calls:
            raise ValueError(f"lexical-rank run has no searches: {query_id}")
        for call in search_calls:
            if not isinstance(call, dict) or call.get("outcome") != "ok":
                raise ValueError(f"lexical-rank run has a failed search: {query_id}")
            query = call.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"lexical-rank run has a blank query: {query_id}")
            generated.append(
                _rank_query(
                    query,
                    gold_docids,
                    maximum_rank=registration.maximum_rank,
                    search=search,
                )
            )
            offline_queries += 1
        generated_ranks = [row.gold_rank for row in generated if row.gold_rank is not None]
        best_generated = min(generated_ranks) if generated_ranks else None
        raw_rank_value = raw.gold_rank or registration.maximum_rank + 1
        generated_rank_value = best_generated or registration.maximum_rank + 1
        items.append(
            LexicalRankCase(
                query_id=query_id,
                gold_docids=gold_docids,
                raw_question=raw,
                generated_queries=tuple(generated),
                best_generated_gold_rank=best_generated,
                raw_question_better_than_generated=raw_rank_value
                < generated_rank_value,
            )
        )

    raw_top5 = _count_at(items, source="raw", cutoff=5)
    raw_top20 = _count_at(items, source="raw", cutoff=20)
    raw_top100 = _count_at(items, source="raw", cutoff=100)
    raw_top1000 = _count_at(items, source="raw", cutoff=registration.maximum_rank)
    generated_top5 = _count_at(items, source="generated", cutoff=5)
    generated_top20 = _count_at(items, source="generated", cutoff=20)
    generated_top100 = _count_at(items, source="generated", cutoff=100)
    generated_top1000 = _count_at(
        items, source="generated", cutoff=registration.maximum_rank
    )
    raw_better = sum(item.raw_question_better_than_generated for item in items)
    raw_gate = LexicalRankGate(
        gate_id="raw_question_top5_cases",
        observed=raw_top5,
        threshold=registration.acceptance.minimum_raw_question_top5_cases,
        passed=(
            raw_top5 >= registration.acceptance.minimum_raw_question_top5_cases
        ),
    )
    pool_gate = LexicalRankGate(
        gate_id="generated_query_top100_cases",
        observed=generated_top100,
        threshold=(
            registration.acceptance.minimum_generated_query_top100_cases_for_pool_diagnosis
        ),
        passed=(
            generated_top100
            >= registration.acceptance.minimum_generated_query_top100_cases_for_pool_diagnosis
        ),
    )
    if raw_gate.passed:
        decision = "raw_question_anchor_candidate"
        next_action = "preregister_raw_question_anchor_retrieval_gate"
    elif pool_gate.passed:
        decision = "bm25_pool_or_rerank_candidate"
        next_action = "diagnose_bm25_candidate_pool_and_reranking"
    else:
        decision = "index_representation_diagnosis_required"
        next_action = "diagnose_index_representation"
    result = LexicalRankResult(
        created_at=_utc_now(),
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=len(items),
        raw_question_top5_cases=raw_top5,
        raw_question_top20_cases=raw_top20,
        raw_question_top100_cases=raw_top100,
        raw_question_top1000_cases=raw_top1000,
        generated_query_top5_cases=generated_top5,
        generated_query_top20_cases=generated_top20,
        generated_query_top100_cases=generated_top100,
        generated_query_top1000_cases=generated_top1000,
        raw_question_better_cases=raw_better,
        offline_bm25_queries=offline_queries,
        items=tuple(items),
        gates=(raw_gate, pool_gate),
        next_action=next_action,
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _rank_query(
    query: str,
    gold_docids: tuple[str, ...],
    *,
    maximum_rank: int,
    search: SearchFunction,
) -> QueryRankObservation:
    docids = tuple(str(value) for value in search(query, maximum_rank))
    if len(docids) > maximum_rank:
        raise ValueError("lexical-rank search exceeded maximum rank")
    if len(docids) != len(set(docids)):
        raise ValueError("lexical-rank search returned duplicate docids")
    gold_set = set(gold_docids)
    match = next(
        ((index + 1, docid) for index, docid in enumerate(docids) if docid in gold_set),
        None,
    )
    return QueryRankObservation(
        query=query,
        query_sha256=_text_sha256(query),
        gold_rank=match[0] if match else None,
        matched_gold_docid=match[1] if match else None,
        top5_docids=docids[:5],
    )


def _count_at(
    items: Sequence[LexicalRankCase],
    *,
    source: Literal["raw", "generated"],
    cutoff: int,
) -> int:
    if source == "raw":
        return sum(
            item.raw_question.gold_rank is not None
            and item.raw_question.gold_rank <= cutoff
            for item in items
        )
    return sum(
        item.best_generated_gold_rank is not None
        and item.best_generated_gold_rank <= cutoff
        for item in items
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
            if "Question:" in text:
                question = text.split("Question:", 1)[1].split(
                    "Your response should be", 1
                )[0].strip()
                if question:
                    return question
    raise ValueError("could not extract question from run trace")


def _compact(value: str) -> str:
    return " ".join(value.split())


def _validate_artifacts(root: Path, artifacts: Sequence[ArtifactReference]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")
