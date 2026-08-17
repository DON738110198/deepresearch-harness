from __future__ import annotations

import json
import re
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
from .evidence_span_oracle import ArtifactReference, answer_coverage
from .pi_browsecomp import PiSmokeSummary, validate_pi_attempt_archives


FunnelCategory = Literal[
    "answer_hidden_after_reference_arrival",
    "answer_visible_reference_uncited",
    "answer_visible_reference_cited_wrong",
]
NextLayer = Literal[
    "evidence_exposure_or_opening",
    "evidence_selection",
    "synthesis_verification",
    "mixed_stratified_audit",
]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunnelSources(StrictContract):
    failure_profile: ArtifactReference
    source_summary: ArtifactReference
    gold_slice: ArtifactReference
    judge_result: ArtifactReference


class FunnelAnalysisContract(StrictContract):
    provider_calls: Literal[0]
    online_search_calls: Literal[0]
    judge_calls: Literal[0]
    sealed_holdout_access: Literal["forbidden"]
    reference_docid_policy: Literal["union_of_gold_and_evidence_docids"]
    visible_content_policy: Literal[
        "saved_reference_snippets_plus_successful_reference_opens"
    ]
    literal_answer_policy: Literal["answer_coverage_all_atoms_v0"]
    citation_policy: Literal["bracketed_known_docid_tokens_v0"]
    classification_precedence: tuple[FunnelCategory, ...]

    @model_validator(mode="after")
    def classification_order_is_frozen(self) -> "FunnelAnalysisContract":
        expected = (
            "answer_hidden_after_reference_arrival",
            "answer_visible_reference_uncited",
            "answer_visible_reference_cited_wrong",
        )
        if self.classification_precedence != expected:
            raise ValueError("evidence reachability classification order changed")
        return self


class FunnelRouting(StrictContract):
    dominance_percent: float = Field(ge=50, le=100)
    denominator_policy: Literal[
        "hidden_over_all_cases_then_uncited_or_cited_over_answer_visible_cases"
    ]


class FunnelAcceptance(StrictContract):
    case_count_must_equal: Literal[67]
    reference_arrived_must_equal: Literal[67]
    provider_calls_must_equal: Literal[0]
    online_search_calls_must_equal: Literal[0]
    judge_calls_must_equal: Literal[0]
    sealed_holdout_access: Literal["forbidden"]


class EvidenceReachabilityRegistration(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-evidence-reachability-funnel-registration-v0"
    ]
    status: Literal["registered_before_trace_execution"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    simplest_baseline: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    sources: FunnelSources
    failure_category: Literal["reference_document_retrieved_answer_wrong"]
    case_count: Literal[67]
    query_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_ids_hash_encoding: Literal["utf8_lf_delimited_with_final_lf"]
    analysis: FunnelAnalysisContract
    routing: FunnelRouting
    acceptance: FunnelAcceptance
    claim_boundary: str = Field(min_length=1)


class ReachabilityRow(StrictContract):
    query_id: str = Field(min_length=1)
    category: FunnelCategory
    reference_docids: tuple[str, ...] = Field(min_length=1)
    retrieved_reference_docids: tuple[str, ...] = Field(min_length=1)
    open_capable_reference_docids: tuple[str, ...]
    explicitly_opened_reference_docids: tuple[str, ...]
    assumed_full_reference_docids: tuple[str, ...]
    cited_reference_docids: tuple[str, ...]
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visible_reference_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visible_answer_coverage: float = Field(ge=0, le=1)
    cited_answer_coverage: float = Field(ge=0, le=1)
    literal_answer_visible: bool
    literal_answer_cited: bool
    search_calls: int = Field(ge=0)
    evidence_open_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def category_matches_reachability(self) -> "ReachabilityRow":
        expected = classify_reachability(
            literal_answer_visible=self.literal_answer_visible,
            literal_answer_cited=self.literal_answer_cited,
        )
        if self.category != expected:
            raise ValueError("reachability category differs from row evidence")
        if self.literal_answer_cited and not self.literal_answer_visible:
            raise ValueError("cited answer evidence must also be visible")
        if not set(self.retrieved_reference_docids).issubset(self.reference_docids):
            raise ValueError("retrieved reference IDs escape the reference set")
        return self


class EvidenceReachabilityFunnel(StrictContract):
    schema_version: Literal["browsecomp-plus-evidence-reachability-funnel-v0"] = (
        "browsecomp-plus-evidence-reachability-funnel-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["posthoc_development_diagnostic_not_effectiveness"] = (
        "posthoc_development_diagnostic_not_effectiveness"
    )
    registration: ArtifactReference
    sources: FunnelSources
    case_count: Literal[67]
    provider_calls: Literal[0] = 0
    online_search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    reference_arrived: int = Field(ge=0, le=67)
    reference_open_capable: int = Field(ge=0, le=67)
    reference_explicitly_opened: int = Field(ge=0, le=67)
    answer_visible: int = Field(ge=0, le=67)
    answer_hidden: int = Field(ge=0, le=67)
    answer_visible_reference_uncited: int = Field(ge=0, le=67)
    answer_visible_reference_cited_wrong: int = Field(ge=0, le=67)
    category_counts: dict[FunnelCategory, int]
    next_layer: NextLayer
    rows: tuple[ReachabilityRow, ...] = Field(min_length=67, max_length=67)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def counts_match_rows(self) -> "EvidenceReachabilityFunnel":
        if len({row.query_id for row in self.rows}) != self.case_count:
            raise ValueError("reachability query IDs must be unique")
        expected = Counter(row.category for row in self.rows)
        if dict(expected) != self.category_counts:
            raise ValueError("reachability category counts differ from rows")
        checks = {
            "reference_arrived": sum(bool(row.retrieved_reference_docids) for row in self.rows),
            "reference_open_capable": sum(bool(row.open_capable_reference_docids) for row in self.rows),
            "reference_explicitly_opened": sum(
                bool(row.explicitly_opened_reference_docids) for row in self.rows
            ),
            "answer_visible": sum(row.literal_answer_visible for row in self.rows),
            "answer_hidden": sum(not row.literal_answer_visible for row in self.rows),
            "answer_visible_reference_uncited": expected.get(
                "answer_visible_reference_uncited", 0
            ),
            "answer_visible_reference_cited_wrong": expected.get(
                "answer_visible_reference_cited_wrong", 0
            ),
        }
        for name, value in checks.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} differs from reachability rows")
        if self.answer_visible + self.answer_hidden != self.case_count:
            raise ValueError("answer visibility does not cover every case")
        return self


def classify_reachability(
    *, literal_answer_visible: bool, literal_answer_cited: bool
) -> FunnelCategory:
    if not literal_answer_visible:
        return "answer_hidden_after_reference_arrival"
    if not literal_answer_cited:
        return "answer_visible_reference_uncited"
    return "answer_visible_reference_cited_wrong"


def route_reachability(
    *, answer_hidden: int, answer_visible_uncited: int, answer_visible_cited: int,
    dominance_percent: float,
) -> NextLayer:
    case_count = answer_hidden + answer_visible_uncited + answer_visible_cited
    if case_count <= 0:
        raise ValueError("reachability routing requires at least one case")
    if _percent(answer_hidden, case_count) >= dominance_percent:
        return "evidence_exposure_or_opening"
    visible = answer_visible_uncited + answer_visible_cited
    if visible <= 0:
        return "evidence_exposure_or_opening"
    if _percent(answer_visible_uncited, visible) >= dominance_percent:
        return "evidence_selection"
    if _percent(answer_visible_cited, visible) >= dominance_percent:
        return "synthesis_verification"
    return "mixed_stratified_audit"


def extract_cited_docids(answer_text: str, known_docids: set[str]) -> tuple[str, ...]:
    cited: set[str] = set()
    for body in re.findall(r"\[([^\[\]]+)\]", answer_text):
        for token in re.split(r"[\s,;]+", body.strip()):
            candidate = token.strip("(){}")
            if candidate in known_docids:
                cited.add(candidate)
    return tuple(sorted(cited))


def run_evidence_reachability_funnel(
    *, registration_path: Path, output_path: Path
) -> EvidenceReachabilityFunnel:
    root = _repository_root(registration_path)
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("evidence reachability output must stay under runs/")
    if output_path.exists():
        raise ValueError("evidence reachability output already exists")

    registration = EvidenceReachabilityRegistration.model_validate_json(
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
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    gold_bytes = source_paths["gold_slice"].read_bytes()
    gold = DevelopmentGoldSlice.model_validate_json(gold_bytes)
    judge_bytes = source_paths["judge_result"].read_bytes()
    judge = DevelopmentJudgeResult.model_validate_json(judge_bytes)

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

    selected_ids = sorted(
        row.query_id
        for row in profile.rows
        if row.category == registration.failure_category
    )
    if len(selected_ids) != registration.case_count:
        raise ValueError("registered reachability case count changed")
    if _query_ids_sha256(selected_ids) != registration.query_ids_sha256:
        raise ValueError("registered reachability query IDs changed")

    items = {item.query_id: item for item in summary.items}
    gold_by_id = {row.query_id: row for row in gold.rows}
    judge_by_id = {row.query_id: row for row in judge.observations}
    rows: list[ReachabilityRow] = []
    for query_id in selected_ids:
        item = items.get(query_id)
        reference = gold_by_id.get(query_id)
        judge_row = judge_by_id.get(query_id)
        if item is None or reference is None or judge_row is None:
            raise ValueError(f"reachability source row is missing: {query_id}")
        if judge_row.correct is not False:
            raise ValueError(f"reachability case is no longer Judge-wrong: {query_id}")
        if item.run_path is None or item.run_sha256 is None or item.prediction_sha256 is None:
            raise ValueError(f"reachability run is not frozen: {query_id}")
        run_path = (root / item.run_path).resolve()
        if not run_path.is_relative_to(root) or sha256(run_path.read_bytes()).hexdigest() != item.run_sha256:
            raise ValueError(f"reachability run hash changed: {query_id}")
        validate_pi_attempt_archives(query_root=run_path.parent, item=item)
        run = load_pi_browsecomp_run(run_path)
        if sha256(run.answer_text.encode("utf-8")).hexdigest() != item.prediction_sha256:
            raise ValueError(f"reachability prediction hash changed: {query_id}")

        reference_docids = set(reference.gold_docids) | set(reference.evidence_docids)
        contents: dict[str, list[str]] = defaultdict(list)
        retrieved: set[str] = set()
        for call in run.search_calls:
            for result in call.results:
                if result.docid in reference_docids:
                    retrieved.add(result.docid)
                    if result.snippet not in contents[result.docid]:
                        contents[result.docid].append(result.snippet)

        explicitly_opened: set[str] = set()
        for call in run.evidence_open_calls:
            if (
                call.docid in reference_docids
                and call.outcome == "ok"
                and call.result is not None
                and call.result.outcome == "opened"
                and call.result.content is not None
            ):
                explicitly_opened.add(call.docid)
                if call.result.content not in contents[call.docid]:
                    contents[call.docid].append(call.result.content)

        final_state = run.disclosure_state
        eligible = set(final_state.eligible_open_docids) if final_state is not None else set()
        state_opened = set(final_state.opened_docids) if final_state is not None else set()
        open_capable = (eligible | explicitly_opened) & reference_docids
        assumed_full = (state_opened - explicitly_opened) & reference_docids
        cited = set(extract_cited_docids(run.answer_text, reference_docids))
        visible_text = _combined_content(contents, retrieved)
        cited_text = _combined_content(contents, cited)
        visible_coverage = answer_coverage(reference.answer, visible_text)
        cited_coverage = answer_coverage(reference.answer, cited_text)
        category = classify_reachability(
            literal_answer_visible=visible_coverage.all_atoms_present,
            literal_answer_cited=cited_coverage.all_atoms_present,
        )
        rows.append(
            ReachabilityRow(
                query_id=query_id,
                category=category,
                reference_docids=tuple(sorted(reference_docids)),
                retrieved_reference_docids=tuple(sorted(retrieved)),
                open_capable_reference_docids=tuple(sorted(open_capable)),
                explicitly_opened_reference_docids=tuple(sorted(explicitly_opened)),
                assumed_full_reference_docids=tuple(sorted(assumed_full)),
                cited_reference_docids=tuple(sorted(cited)),
                answer_sha256=sha256(reference.answer.encode("utf-8")).hexdigest(),
                visible_reference_content_sha256=sha256(
                    visible_text.encode("utf-8")
                ).hexdigest(),
                visible_answer_coverage=visible_coverage.coverage,
                cited_answer_coverage=cited_coverage.coverage,
                literal_answer_visible=visible_coverage.all_atoms_present,
                literal_answer_cited=cited_coverage.all_atoms_present,
                search_calls=len(run.search_calls),
                evidence_open_calls=len(run.evidence_open_calls),
            )
        )

    counts = Counter(row.category for row in rows)
    reference_arrived = sum(bool(row.retrieved_reference_docids) for row in rows)
    if reference_arrived != registration.acceptance.reference_arrived_must_equal:
        raise ValueError("reachability cases no longer have reference-document arrival")
    next_layer = route_reachability(
        answer_hidden=counts.get("answer_hidden_after_reference_arrival", 0),
        answer_visible_uncited=counts.get("answer_visible_reference_uncited", 0),
        answer_visible_cited=counts.get("answer_visible_reference_cited_wrong", 0),
        dominance_percent=registration.routing.dominance_percent,
    )
    result = EvidenceReachabilityFunnel(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=_reference(registration_path, root),
        sources=registration.sources,
        case_count=67,
        reference_arrived=reference_arrived,
        reference_open_capable=sum(bool(row.open_capable_reference_docids) for row in rows),
        reference_explicitly_opened=sum(
            bool(row.explicitly_opened_reference_docids) for row in rows
        ),
        answer_visible=sum(row.literal_answer_visible for row in rows),
        answer_hidden=sum(not row.literal_answer_visible for row in rows),
        answer_visible_reference_uncited=counts.get(
            "answer_visible_reference_uncited", 0
        ),
        answer_visible_reference_cited_wrong=counts.get(
            "answer_visible_reference_cited_wrong", 0
        ),
        category_counts=dict(counts),
        next_layer=next_layer,
        rows=tuple(rows),
        claim_boundary=registration.claim_boundary,
    )
    _atomic_write(output_path, result.model_dump(mode="json"))
    return result


def _combined_content(contents: dict[str, list[str]], docids: set[str]) -> str:
    return "\n".join(
        text
        for docid in sorted(docids)
        for text in contents.get(docid, ())
    )


def _query_ids_sha256(query_ids: list[str]) -> str:
    return sha256(("\n".join(query_ids) + "\n").encode("utf-8")).hexdigest()


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("percentage denominator must be positive")
    return numerator / denominator * 100


def _repository_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("could not locate evidence reachability repository root")


def _validate_source(root: Path, payload: object) -> Path:
    reference = ArtifactReference.model_validate(payload)
    path = (root / reference.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"reachability source is missing: {reference.path}")
    if sha256(path.read_bytes()).hexdigest() != reference.sha256:
        raise ValueError(f"reachability source hash changed: {reference.path}")
    return path


def _reference(path: Path, root: Path) -> ArtifactReference:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("reachability artifact escapes repository root")
    return ArtifactReference(
        path=resolved.relative_to(root).as_posix(),
        sha256=sha256(resolved.read_bytes()).hexdigest(),
    )


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
