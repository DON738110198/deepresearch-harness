from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import DevelopmentQueryArtifact, normalized_text_file_sha256
from .evidence_span_oracle import ArtifactReference
from .retrieval_replay import (
    DenseRuntimeSnapshot,
    RankedHit,
    RetrieverCandidatesManifest,
    select_candidate,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizedArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RawQuestionPrerequisites(StrictContract):
    dense_document_visibility_result: ArtifactReference
    generated_query_dense_rank_result: ArtifactReference
    gold_slice: ArtifactReference


class RawQuestionComparison(StrictContract):
    baseline: str = Field(min_length=1)
    rank_win_policy: str = Field(min_length=1)
    tie_policy: Literal["equal ranks are ties"]


class RawQuestionAcceptance(StrictContract):
    minimum_raw_question_top20_cases: int = Field(ge=1)
    minimum_raw_question_top100_cases_for_pool_diagnosis: int = Field(ge=1)
    minimum_raw_question_rank_wins_for_alignment_signal: int = Field(ge=1)


class RawQuestionBudgets(StrictContract):
    expected_dense_query_encodes: int = Field(gt=0)
    maximum_provider_calls: Literal[0] = 0
    maximum_online_search_calls: Literal[0] = 0
    maximum_judge_calls: Literal[0] = 0
    gpu_allowed: Literal[False] = False


class RawQuestionDenseRankRegistration(StrictContract):
    schema_version: Literal["raw-question-dense-rank-registration-v0"] = (
        "raw-question-dense-rank-registration-v0"
    )
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    prerequisites: RawQuestionPrerequisites
    question_artifact: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    retriever_manifest: NormalizedArtifactReference
    candidate_id: str = Field(min_length=1)
    model_directory: str = Field(min_length=1)
    index_root: str = Field(min_length=1)
    query_source: Literal["frozen_pre_generation_question_artifact"]
    depths: tuple[int, int, int]
    batch_size: int = Field(gt=0)
    comparison: RawQuestionComparison
    acceptance: RawQuestionAcceptance
    budgets: RawQuestionBudgets
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def registration_is_consistent(self) -> "RawQuestionDenseRankRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("raw-question query IDs must be unique")
        if self.depths != (20, 100, 1000):
            raise ValueError("raw-question dense depths must remain 20, 100, and 1000")
        if self.budgets.expected_dense_query_encodes != len(self.query_ids):
            raise ValueError("raw-question encode budget differs from registered cases")
        if self.batch_size > len(self.query_ids):
            raise ValueError("raw-question batch size exceeds registered cases")
        thresholds = (
            self.acceptance.minimum_raw_question_top20_cases,
            self.acceptance.minimum_raw_question_top100_cases_for_pool_diagnosis,
            self.acceptance.minimum_raw_question_rank_wins_for_alignment_signal,
        )
        if any(threshold > len(self.query_ids) for threshold in thresholds):
            raise ValueError("raw-question acceptance exceeds registered cases")
        return self


class RawQuestionDenseSlateCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    returned_hits: int = Field(gt=0)
    ranked_hits: tuple[RankedHit, ...] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def ranking_is_consistent(self) -> "RawQuestionDenseSlateCase":
        if self.returned_hits != len(self.ranked_hits):
            raise ValueError("raw-question returned-hit count differs from ranking")
        docids = [hit.docid for hit in self.ranked_hits]
        if len(docids) != len(set(docids)):
            raise ValueError("raw-question ranking contains duplicate documents")
        return self


class RawQuestionDenseSlate(StrictContract):
    schema_version: Literal["raw-question-dense-slate-v0"] = (
        "raw-question-dense-slate-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["built_gold_blind"] = "built_gold_blind"
    registration: ArtifactReference
    question_artifact: ArtifactReference
    candidate_id: str = Field(min_length=1)
    query_count: int = Field(gt=0)
    dense_query_encodes: int = Field(gt=0)
    model_loads: Literal[1] = 1
    returned_hits_per_query: Literal[1000] = 1000
    runtime: DenseRuntimeSnapshot
    gold_inputs_opened: Literal[False] = False
    generated_rank_result_opened: Literal[False] = False
    visibility_result_opened: Literal[False] = False
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    gpu_used: Literal[False] = False
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[RawQuestionDenseSlateCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> "RawQuestionDenseSlate":
        if self.query_count != len(self.items) or self.dense_query_encodes != len(
            self.items
        ):
            raise ValueError("raw-question slate count differs from items")
        if any(item.returned_hits != self.returned_hits_per_query for item in self.items):
            raise ValueError("raw-question slate depth differs from registered maximum")
        if self.runtime.device != "cpu":
            raise ValueError("raw-question dense slate must be CPU-only")
        return self


RankRelation = Literal["win", "loss", "tie"]


class RawQuestionDenseScoreCase(StrictContract):
    query_id: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    raw_question_gold_rank: int | None = Field(default=None, ge=1, le=1000)
    matched_raw_gold_docid: str | None = None
    generated_query_best_gold_rank: int | None = Field(default=None, ge=1, le=1000)
    raw_rank_relation: RankRelation
    raw_gold_hit_at20: bool
    raw_gold_hit_at100: bool
    raw_gold_hit_at1000: bool

    @model_validator(mode="after")
    def rank_fields_are_consistent(self) -> "RawQuestionDenseScoreCase":
        rank = self.raw_question_gold_rank
        expected_flags = (
            rank is not None and rank <= 20,
            rank is not None and rank <= 100,
            rank is not None,
        )
        if expected_flags != (
            self.raw_gold_hit_at20,
            self.raw_gold_hit_at100,
            self.raw_gold_hit_at1000,
        ):
            raise ValueError("raw-question hit flags differ from observed rank")
        if (rank is None) != (self.matched_raw_gold_docid is None):
            raise ValueError("raw-question matched document differs from observed rank")
        if self.raw_rank_relation != compare_ranks(
            rank, self.generated_query_best_gold_rank
        ):
            raise ValueError("raw-question rank relation differs from observed ranks")
        return self


class RawQuestionDenseGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int = Field(ge=0)
    operator: Literal["ge"] = "ge"
    threshold: int = Field(ge=1)
    passed: bool


RawQuestionDecision = Literal[
    "raw_question_top20_candidate",
    "raw_question_pool_candidate",
    "raw_question_alignment_signal_only",
    "freeze_raw_question_dense",
]


RawQuestionNextAction = Literal[
    "preregister_fresh_raw_question_anchor_comparison",
    "preregister_raw_question_pool_reranker_gate",
    "diagnose_constraint_preserving_query_portfolio",
    "freeze_raw_question_dense_and_preregister_typed_bridge",
]


class RawQuestionDenseRankResult(StrictContract):
    schema_version: Literal["raw-question-dense-rank-result-v0"] = (
        "raw-question-dense-rank-result-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    decision: RawQuestionDecision
    registration: ArtifactReference
    slate: ArtifactReference
    query_count: int = Field(gt=0)
    dense_query_encodes: int = Field(gt=0)
    generated_query_gold_hit_cases_at20: int = Field(ge=0)
    generated_query_gold_hit_cases_at100: int = Field(ge=0)
    generated_query_gold_hit_cases_at1000: int = Field(ge=0)
    raw_question_gold_hit_cases_at20: int = Field(ge=0)
    raw_question_gold_hit_cases_at100: int = Field(ge=0)
    raw_question_gold_hit_cases_at1000: int = Field(ge=0)
    raw_minus_generated_cases_at20: int
    raw_minus_generated_cases_at100: int
    raw_minus_generated_cases_at1000: int
    raw_rank_wins: int = Field(ge=0)
    raw_rank_losses: int = Field(ge=0)
    raw_rank_ties: int = Field(ge=0)
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    gpu_used: Literal[False] = False
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[RawQuestionDenseScoreCase, ...] = Field(min_length=1)
    gates: tuple[RawQuestionDenseGate, RawQuestionDenseGate, RawQuestionDenseGate]
    next_action: RawQuestionNextAction
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def aggregates_are_consistent(self) -> "RawQuestionDenseRankResult":
        raw_counts = (
            sum(item.raw_gold_hit_at20 for item in self.items),
            sum(item.raw_gold_hit_at100 for item in self.items),
            sum(item.raw_gold_hit_at1000 for item in self.items),
        )
        relations = (
            sum(item.raw_rank_relation == "win" for item in self.items),
            sum(item.raw_rank_relation == "loss" for item in self.items),
            sum(item.raw_rank_relation == "tie" for item in self.items),
        )
        if self.query_count != len(self.items) or self.dense_query_encodes != len(
            self.items
        ):
            raise ValueError("raw-question result count differs from items")
        if raw_counts != (
            self.raw_question_gold_hit_cases_at20,
            self.raw_question_gold_hit_cases_at100,
            self.raw_question_gold_hit_cases_at1000,
        ):
            raise ValueError("raw-question aggregate recall differs from items")
        if relations != (self.raw_rank_wins, self.raw_rank_losses, self.raw_rank_ties):
            raise ValueError("raw-question rank relations differ from items")
        return self


def compare_ranks(raw_rank: int | None, generated_rank: int | None) -> RankRelation:
    if raw_rank is None and generated_rank is None:
        return "tie"
    if generated_rank is None:
        return "win"
    if raw_rank is None:
        return "loss"
    if raw_rank < generated_rank:
        return "win"
    if raw_rank > generated_rank:
        return "loss"
    return "tie"


def choose_raw_question_dense_decision(
    *,
    raw_top20_cases: int,
    raw_top100_cases: int,
    raw_rank_wins: int,
    minimum_top20_cases: int,
    minimum_top100_cases: int,
    minimum_rank_wins: int,
) -> tuple[RawQuestionDecision, RawQuestionNextAction]:
    if raw_top20_cases >= minimum_top20_cases:
        return (
            "raw_question_top20_candidate",
            "preregister_fresh_raw_question_anchor_comparison",
        )
    if raw_top100_cases >= minimum_top100_cases:
        return (
            "raw_question_pool_candidate",
            "preregister_raw_question_pool_reranker_gate",
        )
    if raw_rank_wins >= minimum_rank_wins:
        return (
            "raw_question_alignment_signal_only",
            "diagnose_constraint_preserving_query_portfolio",
        )
    return (
        "freeze_raw_question_dense",
        "freeze_raw_question_dense_and_preregister_typed_bridge",
    )


def load_raw_question_builder_registration(
    path: Path,
) -> RawQuestionDenseRankRegistration:
    registration = RawQuestionDenseRankRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifact(root, registration.question_artifact)
    questions = _load_questions(root, registration)
    if tuple(query_id for query_id, _ in questions) != registration.query_ids:
        raise ValueError("raw-question artifact cases differ from registration")

    manifest_path = _require_file(root, registration.retriever_manifest.path)
    if (
        normalized_text_file_sha256(manifest_path)
        != registration.retriever_manifest.normalized_sha256
    ):
        raise ValueError("raw-question retriever manifest hash changed")
    manifest = RetrieverCandidatesManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    select_candidate(manifest, registration.candidate_id)
    _require_directory(root, registration.model_directory)
    _require_directory(root, registration.index_root)
    return registration


def load_raw_question_dense_registration(
    path: Path,
) -> RawQuestionDenseRankRegistration:
    registration = load_raw_question_builder_registration(path)
    root = path.resolve().parents[2]
    for artifact in registration.prerequisites.model_dump().values():
        _validate_artifact(root, ArtifactReference.model_validate(artifact))

    visibility = _read_object(
        _require_file(
            root, registration.prerequisites.dense_document_visibility_result.path
        )
    )
    if visibility.get("decision") != "reject_head_truncation_hypothesis":
        raise ValueError("raw-question visibility prerequisite changed")
    generated = _read_object(
        _require_file(
            root, registration.prerequisites.generated_query_dense_rank_result.path
        )
    )
    if generated.get("decision") != "freeze_dense_channel":
        raise ValueError("raw-question generated-rank prerequisite changed")
    for label, artifact in (("visibility", visibility), ("generated-rank", generated)):
        if _item_ids(artifact, label=label) != registration.query_ids:
            raise ValueError(f"raw-question {label} cases differ from registration")
    return registration


def collect_raw_questions(registration_path: Path) -> tuple[str, ...]:
    registration = load_raw_question_builder_registration(registration_path)
    root = registration_path.resolve().parents[2]
    return tuple(question for _, question in _load_questions(root, registration))


def build_raw_question_dense_slate(
    *,
    registration_path: Path,
    output_path: Path,
    candidate_results: Mapping[str, Sequence[RankedHit]],
    runtime: DenseRuntimeSnapshot,
) -> RawQuestionDenseSlate:
    registration = load_raw_question_builder_registration(registration_path)
    root = registration_path.resolve().parents[2]
    output = output_path.resolve()
    _require_run_output(output, root)
    if output.exists():
        raise ValueError("raw-question dense slate already exists")
    if runtime.device != "cpu":
        raise ValueError("raw-question dense audit must run on CPU")

    manifest_path = _require_file(root, registration.retriever_manifest.path)
    manifest = RetrieverCandidatesManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    candidate = select_candidate(manifest, registration.candidate_id)
    if runtime.model_file_sha256 != candidate.model.model_file_sha256:
        raise ValueError("raw-question runtime model hash differs from registration")
    expected_shards = {shard.filename: shard.sha256 for shard in candidate.index.shards}
    if runtime.index_shards_sha256 != expected_shards:
        raise ValueError("raw-question runtime index hashes differ from registration")

    questions = _load_questions(root, registration)
    question_texts = [question for _, question in questions]
    if len(question_texts) != len(set(question_texts)):
        raise ValueError("raw-question selected questions must be unique")
    missing = [question for question in question_texts if question not in candidate_results]
    extras = [question for question in candidate_results if question not in set(question_texts)]
    if missing or extras:
        raise ValueError(
            "raw-question result keys differ from frozen questions: "
            f"missing={len(missing)}, extras={len(extras)}"
        )

    maximum_depth = registration.depths[-1]
    items: list[RawQuestionDenseSlateCase] = []
    for query_id, question in questions:
        hits = tuple(candidate_results[question])
        if len(hits) != maximum_depth:
            raise ValueError(
                f"raw-question query returned {len(hits)} hits instead of {maximum_depth}"
            )
        items.append(
            RawQuestionDenseSlateCase(
                query_id=query_id,
                question=question,
                question_sha256=_text_sha256(question),
                returned_hits=len(hits),
                ranked_hits=hits,
            )
        )

    slate = RawQuestionDenseSlate(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=_sha256_file(registration_path),
        ),
        question_artifact=registration.question_artifact,
        candidate_id=registration.candidate_id,
        query_count=len(items),
        dense_query_encodes=len(items),
        runtime=runtime,
        items=tuple(items),
    )
    _atomic_write(output, slate.model_dump(mode="json"))
    return slate


def score_raw_question_dense_slate(
    *, registration_path: Path, slate_path: Path, output_path: Path
) -> RawQuestionDenseRankResult:
    registration = load_raw_question_dense_registration(registration_path)
    root = registration_path.resolve().parents[2]
    slate_file = slate_path.resolve()
    output = output_path.resolve()
    _require_run_output(slate_file, root)
    _require_run_output(output, root)
    if output.exists():
        raise ValueError("raw-question dense result already exists")
    if not slate_file.is_file():
        raise ValueError("raw-question dense slate is missing")
    slate = RawQuestionDenseSlate.model_validate_json(
        slate_file.read_text(encoding="utf-8")
    )
    expected_registration = registration_path.resolve().relative_to(root).as_posix()
    if slate.registration != ArtifactReference(
        path=expected_registration, sha256=_sha256_file(registration_path)
    ):
        raise ValueError("raw-question slate registration binding changed")
    if slate.question_artifact != registration.question_artifact:
        raise ValueError("raw-question slate question binding changed")
    if tuple(item.query_id for item in slate.items) != registration.query_ids:
        raise ValueError("raw-question slate cases differ from registration")

    questions = dict(_load_questions(root, registration))
    for item in slate.items:
        question = questions[item.query_id]
        if item.question != question or item.question_sha256 != _text_sha256(question):
            raise ValueError(f"raw-question slate question changed: {item.query_id}")

    generated = _read_object(
        _require_file(
            root, registration.prerequisites.generated_query_dense_rank_result.path
        )
    )
    generated_items = _items(generated, label="generated-rank")
    generated_by_id = {str(item["query_id"]): item for item in generated_items}
    gold = _read_object(_require_file(root, registration.prerequisites.gold_slice.path))
    gold_by_id = {str(row["query_id"]): row for row in _rows(gold)}

    scored: list[RawQuestionDenseScoreCase] = []
    for slate_item in slate.items:
        gold_row = gold_by_id.get(slate_item.query_id)
        if gold_row is None or not isinstance(gold_row.get("gold_docids"), list):
            raise ValueError(f"raw-question gold case is missing: {slate_item.query_id}")
        gold_docids = tuple(str(value) for value in gold_row["gold_docids"])
        generated_item = generated_by_id[slate_item.query_id]
        if tuple(str(value) for value in generated_item["gold_docids"]) != gold_docids:
            raise ValueError(f"raw-question gold documents changed: {slate_item.query_id}")
        match = next(
            (
                (rank, hit.docid)
                for rank, hit in enumerate(slate_item.ranked_hits, start=1)
                if hit.docid in set(gold_docids)
            ),
            None,
        )
        raw_rank = match[0] if match else None
        matched_docid = match[1] if match else None
        raw_generated_rank = generated_item.get("best_gold_rank")
        generated_rank = int(raw_generated_rank) if raw_generated_rank is not None else None
        scored.append(
            RawQuestionDenseScoreCase(
                query_id=slate_item.query_id,
                gold_docids=gold_docids,
                raw_question_gold_rank=raw_rank,
                matched_raw_gold_docid=matched_docid,
                generated_query_best_gold_rank=generated_rank,
                raw_rank_relation=compare_ranks(raw_rank, generated_rank),
                raw_gold_hit_at20=raw_rank is not None and raw_rank <= 20,
                raw_gold_hit_at100=raw_rank is not None and raw_rank <= 100,
                raw_gold_hit_at1000=raw_rank is not None,
            )
        )

    generated20 = int(generated["dense_gold_hit_cases_at20"])
    generated100 = int(generated["dense_gold_hit_cases_at100"])
    generated1000 = int(generated["dense_gold_hit_cases_at1000"])
    raw20 = sum(item.raw_gold_hit_at20 for item in scored)
    raw100 = sum(item.raw_gold_hit_at100 for item in scored)
    raw1000 = sum(item.raw_gold_hit_at1000 for item in scored)
    wins = sum(item.raw_rank_relation == "win" for item in scored)
    acceptance = registration.acceptance
    decision, next_action = choose_raw_question_dense_decision(
        raw_top20_cases=raw20,
        raw_top100_cases=raw100,
        raw_rank_wins=wins,
        minimum_top20_cases=acceptance.minimum_raw_question_top20_cases,
        minimum_top100_cases=(
            acceptance.minimum_raw_question_top100_cases_for_pool_diagnosis
        ),
        minimum_rank_wins=(
            acceptance.minimum_raw_question_rank_wins_for_alignment_signal
        ),
    )
    result = RawQuestionDenseRankResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        registration=slate.registration,
        slate=ArtifactReference(
            path=slate_file.relative_to(root).as_posix(),
            sha256=_sha256_file(slate_file),
        ),
        query_count=len(scored),
        dense_query_encodes=slate.dense_query_encodes,
        generated_query_gold_hit_cases_at20=generated20,
        generated_query_gold_hit_cases_at100=generated100,
        generated_query_gold_hit_cases_at1000=generated1000,
        raw_question_gold_hit_cases_at20=raw20,
        raw_question_gold_hit_cases_at100=raw100,
        raw_question_gold_hit_cases_at1000=raw1000,
        raw_minus_generated_cases_at20=raw20 - generated20,
        raw_minus_generated_cases_at100=raw100 - generated100,
        raw_minus_generated_cases_at1000=raw1000 - generated1000,
        raw_rank_wins=wins,
        raw_rank_losses=sum(item.raw_rank_relation == "loss" for item in scored),
        raw_rank_ties=sum(item.raw_rank_relation == "tie" for item in scored),
        items=tuple(scored),
        gates=(
            _gate(
                "raw_question_gold_doc_recall_at20_cases",
                raw20,
                acceptance.minimum_raw_question_top20_cases,
            ),
            _gate(
                "raw_question_gold_doc_recall_at100_cases_for_pool_diagnosis",
                raw100,
                acceptance.minimum_raw_question_top100_cases_for_pool_diagnosis,
            ),
            _gate(
                "raw_question_rank_wins_for_alignment_signal",
                wins,
                acceptance.minimum_raw_question_rank_wins_for_alignment_signal,
            ),
        ),
        next_action=next_action,
        claim_boundary=registration.claim_boundary,
    )
    _atomic_write(output, result.model_dump(mode="json"))
    return result


def _gate(gate_id: str, observed: int, threshold: int) -> RawQuestionDenseGate:
    return RawQuestionDenseGate(
        gate_id=gate_id,
        observed=observed,
        threshold=threshold,
        passed=observed >= threshold,
    )


def _load_questions(
    root: Path, registration: RawQuestionDenseRankRegistration
) -> tuple[tuple[str, str], ...]:
    artifact = DevelopmentQueryArtifact.model_validate_json(
        _require_file(root, registration.question_artifact.path).read_text(
            encoding="utf-8"
        )
    )
    by_id = {item.query_id: item.question for item in artifact.queries}
    missing = [query_id for query_id in registration.query_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"raw-question artifact is missing cases: {missing}")
    return tuple((query_id, by_id[query_id]) for query_id in registration.query_ids)


def _validate_artifact(root: Path, artifact: ArtifactReference) -> None:
    path = _require_file(root, artifact.path)
    if _sha256_file(path) != artifact.sha256:
        raise ValueError(f"raw-question artifact hash changed: {artifact.path}")


def _item_ids(value: Mapping[str, object], *, label: str) -> tuple[str, ...]:
    return tuple(str(item["query_id"]) for item in _items(value, label=label))


def _items(value: Mapping[str, object], *, label: str) -> list[dict[str, object]]:
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"raw-question {label} items are invalid")
    return items


def _rows(value: Mapping[str, object]) -> list[dict[str, object]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("raw-question gold rows are invalid")
    return rows


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"raw-question artifact is not an object: {path}")
    return value


def _require_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"raw-question required file is missing or escapes root: {relative}")
    return path


def _require_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(
            f"raw-question required directory is missing or escapes root: {relative}"
        )
    return path


def _require_run_output(path: Path, root: Path) -> None:
    if not path.is_relative_to((root / "runs").resolve()):
        raise ValueError("raw-question artifacts must stay under ignored runs/")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise ValueError("raw-question partial artifact already exists")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(path)
