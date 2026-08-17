from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DevelopmentGoldSlice
from .browsecomp_plus import load_pi_browsecomp_run
from .development_failure_profile import DevelopmentFailureProfile
from .development_judge import DevelopmentJudgeResult
from .evidence_reachability_funnel import (
    _atomic_write,
    _combined_content,
    _query_ids_sha256,
    _reference,
    _repository_root,
    _validate_source,
    extract_cited_docids,
)
from .evidence_span_oracle import ArtifactReference, answer_coverage
from .pi_browsecomp import PiSmokeSummary, validate_pi_attempt_archives


CausalCategory = Literal[
    "answer_contract_failure",
    "no_reference_arrival",
    "supporting_evidence_only_arrived",
    "gold_arrived_gold_span_incomplete",
    "gold_arrived_answer_visible_uncited",
    "gold_arrived_answer_visible_cited_wrong",
]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunnelSources(StrictContract):
    failure_profile: ArtifactReference
    source_summary: ArtifactReference
    gold_slice: ArtifactReference
    judge_result: ArtifactReference


class CausalFunnelAnalysis(StrictContract):
    provider_calls: Literal[0]
    online_search_calls: Literal[0]
    judge_calls: Literal[0]
    gpu_calls: Literal[0]
    sealed_holdout_access: Literal["forbidden"]
    arrival_policy: Literal["gold_docids_distinguished_from_evidence_docids"]
    visible_content_policy: Literal[
        "saved_gold_and_supporting_snippets_plus_successful_opens"
    ]
    literal_answer_policy: Literal["answer_coverage_all_atoms_v0"]
    citation_policy: Literal["bracketed_known_docid_tokens_v0"]
    classification_precedence: tuple[CausalCategory, ...]

    @model_validator(mode="after")
    def classification_order_is_frozen(self) -> "CausalFunnelAnalysis":
        expected = (
            "answer_contract_failure",
            "no_reference_arrival",
            "supporting_evidence_only_arrived",
            "gold_arrived_gold_span_incomplete",
            "gold_arrived_answer_visible_uncited",
            "gold_arrived_answer_visible_cited_wrong",
        )
        if self.classification_precedence != expected:
            raise ValueError("gold/evidence causal funnel classification order changed")
        return self


class CausalFunnelAcceptance(StrictContract):
    case_count_must_equal: Literal[110]
    provider_calls_must_equal: Literal[0]
    online_search_calls_must_equal: Literal[0]
    judge_calls_must_equal: Literal[0]
    gpu_calls_must_equal: Literal[0]
    sealed_holdout_access: Literal["forbidden"]


class GoldEvidenceCausalFunnelRegistration(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-gold-evidence-causal-funnel-registration-v1"
    ]
    status: Literal["selected_after_v0_before_correction_execution"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    simplest_baseline: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    sources: FunnelSources
    case_count: Literal[110]
    query_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_ids_hash_encoding: Literal["utf8_lf_delimited_with_final_lf"]
    analysis: CausalFunnelAnalysis
    acceptance: CausalFunnelAcceptance
    claim_boundary: str = Field(min_length=1)


class CausalFunnelRow(StrictContract):
    query_id: str = Field(min_length=1)
    profile_category: str = Field(min_length=1)
    category: CausalCategory
    gold_docids: tuple[str, ...]
    supporting_docids: tuple[str, ...]
    retrieved_gold_docids: tuple[str, ...]
    retrieved_supporting_docids: tuple[str, ...]
    cited_reference_docids: tuple[str, ...]
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visible_reference_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_visible_answer_coverage: float = Field(ge=0, le=1)
    supporting_visible_answer_coverage: float = Field(ge=0, le=1)
    any_reference_answer_coverage: float = Field(ge=0, le=1)
    cited_answer_coverage: float = Field(ge=0, le=1)
    answer_visible_any_reference: bool
    answer_visible_supporting_only: bool
    gold_span_complete: bool
    cited_answer_complete: bool
    search_calls: int = Field(ge=0)
    evidence_open_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def category_matches_evidence(self) -> "CausalFunnelRow":
        expected = classify_causal_failure(
            answer_contract_failure=self.profile_category == "answer_contract_failure",
            gold_arrived=bool(self.retrieved_gold_docids),
            supporting_arrived=bool(self.retrieved_supporting_docids),
            gold_span_complete=self.gold_span_complete,
            cited_answer_complete=self.cited_answer_complete,
        )
        if self.category != expected:
            raise ValueError("causal funnel category differs from row evidence")
        if self.cited_answer_complete and not self.answer_visible_any_reference:
            raise ValueError("cited answer evidence must also be visible")
        if self.gold_span_complete and not self.answer_visible_any_reference:
            raise ValueError("gold-visible answer must be visible in all references")
        return self


class CausalQueues(StrictContract):
    retrieval_or_evidence_frontier: int = Field(ge=0)
    evidence_exposure: int = Field(ge=0)
    evidence_selection: int = Field(ge=0)
    synthesis_verification: int = Field(ge=0)
    answer_contract: int = Field(ge=0)


class GoldEvidenceCausalFunnel(StrictContract):
    schema_version: Literal["browsecomp-plus-gold-evidence-causal-funnel-v1"] = (
        "browsecomp-plus-gold-evidence-causal-funnel-v1"
    )
    created_at: str = Field(min_length=1)
    status: Literal["selected_after_v0_development_diagnostic_not_effectiveness"] = (
        "selected_after_v0_development_diagnostic_not_effectiveness"
    )
    registration: ArtifactReference
    sources: FunnelSources
    case_count: Literal[110]
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    gpu_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    promotion_allowed: Literal[False] = False
    category_counts: dict[CausalCategory, int]
    queues: CausalQueues
    rows: tuple[CausalFunnelRow, ...] = Field(min_length=110, max_length=110)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def counts_match_rows(self) -> "GoldEvidenceCausalFunnel":
        if len({row.query_id for row in self.rows}) != self.case_count:
            raise ValueError("causal funnel query IDs must be unique")
        counts = dict(Counter(row.category for row in self.rows))
        if counts != self.category_counts:
            raise ValueError("causal funnel category counts differ from rows")
        if self.queues != queue_counts(counts):
            raise ValueError("causal funnel queues differ from category counts")
        return self


def classify_causal_failure(
    *,
    answer_contract_failure: bool,
    gold_arrived: bool,
    supporting_arrived: bool,
    gold_span_complete: bool,
    cited_answer_complete: bool,
) -> CausalCategory:
    if answer_contract_failure:
        return "answer_contract_failure"
    if not gold_arrived and not supporting_arrived:
        return "no_reference_arrival"
    if not gold_arrived:
        return "supporting_evidence_only_arrived"
    if not gold_span_complete:
        return "gold_arrived_gold_span_incomplete"
    if not cited_answer_complete:
        return "gold_arrived_answer_visible_uncited"
    return "gold_arrived_answer_visible_cited_wrong"


def queue_counts(counts: dict[CausalCategory, int]) -> CausalQueues:
    return CausalQueues(
        retrieval_or_evidence_frontier=(
            counts.get("no_reference_arrival", 0)
            + counts.get("supporting_evidence_only_arrived", 0)
        ),
        evidence_exposure=counts.get("gold_arrived_gold_span_incomplete", 0),
        evidence_selection=counts.get("gold_arrived_answer_visible_uncited", 0),
        synthesis_verification=counts.get(
            "gold_arrived_answer_visible_cited_wrong", 0
        ),
        answer_contract=counts.get("answer_contract_failure", 0),
    )


def run_gold_evidence_causal_funnel(
    *, registration_path: Path, output_path: Path
) -> GoldEvidenceCausalFunnel:
    root = _repository_root(registration_path)
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("causal funnel output must stay under runs/")
    if output_path.exists():
        raise ValueError("causal funnel output already exists")
    registration = GoldEvidenceCausalFunnelRegistration.model_validate_json(
        registration_path.read_text(encoding="utf-8")
    )
    source_paths = {
        name: _validate_source(root, reference)
        for name, reference in registration.sources.model_dump().items()
    }
    profile = DevelopmentFailureProfile.model_validate_json(
        source_paths["failure_profile"].read_text(encoding="utf-8")
    )
    summary_bytes = source_paths["source_summary"].read_bytes()
    gold_bytes = source_paths["gold_slice"].read_bytes()
    judge_bytes = source_paths["judge_result"].read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    gold = DevelopmentGoldSlice.model_validate_json(gold_bytes)
    judge = DevelopmentJudgeResult.model_validate_json(judge_bytes)
    _validate_source_relationships(
        profile=profile,
        summary_bytes=summary_bytes,
        gold=gold,
        gold_bytes=gold_bytes,
        judge=judge,
        judge_bytes=judge_bytes,
    )
    selected_ids = sorted(
        row.query_id for row in judge.observations if row.correct is False
    )
    if len(selected_ids) != registration.case_count:
        raise ValueError("registered causal funnel case count changed")
    if _query_ids_sha256(selected_ids) != registration.query_ids_sha256:
        raise ValueError("registered causal funnel query IDs changed")

    items = {item.query_id: item for item in summary.items}
    profile_by_id = {row.query_id: row for row in profile.rows}
    gold_by_id = {row.query_id: row for row in gold.rows}
    rows = tuple(
        _build_row(
            root=root,
            query_id=query_id,
            item=items.get(query_id),
            profile_row=profile_by_id.get(query_id),
            reference=gold_by_id.get(query_id),
        )
        for query_id in selected_ids
    )
    counts = dict(Counter(row.category for row in rows))
    result = GoldEvidenceCausalFunnel(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=_reference(registration_path, root),
        sources=registration.sources,
        case_count=registration.case_count,
        category_counts=counts,
        queues=queue_counts(counts),
        rows=rows,
        claim_boundary=registration.claim_boundary,
    )
    _atomic_write(output_path, result.model_dump(mode="json"))
    return result


def _validate_source_relationships(
    *,
    profile: DevelopmentFailureProfile,
    summary_bytes: bytes,
    gold: DevelopmentGoldSlice,
    gold_bytes: bytes,
    judge: DevelopmentJudgeResult,
    judge_bytes: bytes,
) -> None:
    summary_sha = sha256(summary_bytes).hexdigest()
    gold_sha = sha256(gold_bytes).hexdigest()
    if profile.source_summary.sha256 != summary_sha:
        raise ValueError("failure profile targets another summary")
    if profile.judge_result.sha256 != sha256(judge_bytes).hexdigest():
        raise ValueError("failure profile targets another Judge result")
    if gold.source_summary_sha256 != summary_sha:
        raise ValueError("gold slice targets another summary")
    if judge.source_summary_sha256 != summary_sha or judge.gold_slice_sha256 != gold_sha:
        raise ValueError("Judge result targets another prediction or gold slice")
    if judge.status != "succeeded" or judge.parse_failures or judge.request_failures:
        raise ValueError("causal funnel requires a complete calibrated Judge result")


def _build_row(
    *, root: Path, query_id: str, item: object, profile_row: object, reference: object
) -> CausalFunnelRow:
    if item is None or profile_row is None or reference is None:
        raise ValueError(f"causal funnel source row is missing: {query_id}")
    if profile_row.judge_correct:
        raise ValueError(f"causal funnel case is no longer Judge-wrong: {query_id}")
    if item.run_path is None or item.run_sha256 is None or item.prediction_sha256 is None:
        raise ValueError(f"causal funnel run is not frozen: {query_id}")
    run_path = (root / item.run_path).resolve()
    if not run_path.is_relative_to(root):
        raise ValueError(f"causal funnel run escapes root: {query_id}")
    if sha256(run_path.read_bytes()).hexdigest() != item.run_sha256:
        raise ValueError(f"causal funnel run hash changed: {query_id}")
    validate_pi_attempt_archives(query_root=run_path.parent, item=item)
    run = load_pi_browsecomp_run(run_path)
    if sha256(run.answer_text.encode("utf-8")).hexdigest() != item.prediction_sha256:
        raise ValueError(f"causal funnel prediction hash changed: {query_id}")

    gold_docids = set(reference.gold_docids)
    supporting_docids = set(reference.evidence_docids) - gold_docids
    reference_docids = gold_docids | supporting_docids
    contents: dict[str, list[str]] = defaultdict(list)
    retrieved: set[str] = set()
    for call in run.search_calls:
        for result in call.results:
            if result.docid in reference_docids:
                retrieved.add(result.docid)
                if result.snippet not in contents[result.docid]:
                    contents[result.docid].append(result.snippet)
    for call in run.evidence_open_calls:
        if (
            call.docid in reference_docids
            and call.outcome == "ok"
            and call.result is not None
            and call.result.outcome == "opened"
            and call.result.content is not None
            and call.result.content not in contents[call.docid]
        ):
            contents[call.docid].append(call.result.content)

    cited = set(extract_cited_docids(run.answer_text, reference_docids))
    gold_text = _combined_content(contents, retrieved & gold_docids)
    supporting_text = _combined_content(contents, retrieved & supporting_docids)
    visible_text = _combined_content(contents, retrieved)
    cited_text = _combined_content(contents, cited)
    gold_coverage = answer_coverage(reference.answer, gold_text)
    supporting_coverage = answer_coverage(reference.answer, supporting_text)
    visible_coverage = answer_coverage(reference.answer, visible_text)
    cited_coverage = answer_coverage(reference.answer, cited_text)
    return CausalFunnelRow(
        query_id=query_id,
        profile_category=profile_row.category,
        category=classify_causal_failure(
            answer_contract_failure=profile_row.category == "answer_contract_failure",
            gold_arrived=bool(retrieved & gold_docids),
            supporting_arrived=bool(retrieved & supporting_docids),
            gold_span_complete=gold_coverage.all_atoms_present,
            cited_answer_complete=cited_coverage.all_atoms_present,
        ),
        gold_docids=tuple(sorted(gold_docids)),
        supporting_docids=tuple(sorted(supporting_docids)),
        retrieved_gold_docids=tuple(sorted(retrieved & gold_docids)),
        retrieved_supporting_docids=tuple(sorted(retrieved & supporting_docids)),
        cited_reference_docids=tuple(sorted(cited)),
        run_sha256=item.run_sha256,
        prediction_sha256=item.prediction_sha256,
        answer_sha256=sha256(reference.answer.encode("utf-8")).hexdigest(),
        visible_reference_content_sha256=sha256(visible_text.encode("utf-8")).hexdigest(),
        gold_visible_answer_coverage=gold_coverage.coverage,
        supporting_visible_answer_coverage=supporting_coverage.coverage,
        any_reference_answer_coverage=visible_coverage.coverage,
        cited_answer_coverage=cited_coverage.coverage,
        answer_visible_any_reference=visible_coverage.all_atoms_present,
        answer_visible_supporting_only=supporting_coverage.all_atoms_present,
        gold_span_complete=gold_coverage.all_atoms_present,
        cited_answer_complete=cited_coverage.all_atoms_present,
        search_calls=len(run.search_calls),
        evidence_open_calls=len(run.evidence_open_calls),
    )
