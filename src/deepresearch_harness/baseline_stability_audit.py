from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DevelopmentGoldSlice, DiagnosticSummary
from .browsecomp_plus import load_pi_browsecomp_run
from .development_judge import DevelopmentJudgeResult
from .evidence_span_oracle import ArtifactReference
from .pi_browsecomp import PiSmokeSummary


FailureCategory = Literal[
    "stable_correct",
    "persistent_retrieval_miss",
    "gold_doc_present_span_missing",
    "answer_span_present_persistent_wrong",
    "unstable_answer",
    "persistent_wrong_other",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaselineStabilityCase(StrictContract):
    query_id: str = Field(min_length=1)
    category: FailureCategory
    judge_correct_trials: int = Field(ge=0, le=3)
    normalized_exact_trials: int = Field(ge=0, le=3)
    gold_doc_retrieved_trials: int = Field(ge=0, le=3)
    gold_answer_span_in_gold_preview_trials: int = Field(ge=0, le=3)


class BaselineStabilityAudit(StrictContract):
    schema_version: Literal["v10-baseline-cross-trial-stability-audit-v0"] = (
        "v10-baseline-cross-trial-stability-audit-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["development_diagnostic_not_official"] = (
        "development_diagnostic_not_official"
    )
    baseline_adapter: Literal["pi-browsecomp-v10"] = "pi-browsecomp-v10"
    trial_count: Literal[3] = 3
    query_count: int = Field(gt=0)
    total_observations: int = Field(gt=0)
    category_counts: dict[FailureCategory, int]
    cases: tuple[BaselineStabilityCase, ...] = Field(min_length=1)
    actionable_case_ids: tuple[str, ...]
    retrieval_query_generation_status: Literal["frozen_by_prior_registered_rejections"] = (
        "frozen_by_prior_registered_rejections"
    )
    evidence_recall_warning: str = Field(min_length=1)
    audit_provider_calls: Literal[0] = 0
    audit_judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    sources: dict[str, ArtifactReference]
    next_action: Literal["calibrate_monotonic_post_run_evidence_overlay"] = (
        "calibrate_monotonic_post_run_evidence_overlay"
    )
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def counts_match_cases(self) -> "BaselineStabilityAudit":
        if self.total_observations != self.trial_count * self.query_count:
            raise ValueError("stability observation count is inconsistent")
        observed = Counter(case.category for case in self.cases)
        if dict(observed) != self.category_counts:
            raise ValueError("stability category counts differ from cases")
        expected_actionable = tuple(
            case.query_id
            for case in self.cases
            if case.category == "gold_doc_present_span_missing"
        )
        if self.actionable_case_ids != expected_actionable:
            raise ValueError("actionable case IDs differ from the selected failure cluster")
        return self


def classify_baseline_stability(
    *,
    judge_correct_trials: int,
    gold_doc_retrieved_trials: int,
    gold_answer_span_trials: int,
) -> FailureCategory:
    if judge_correct_trials == 3:
        return "stable_correct"
    if judge_correct_trials == 0 and gold_doc_retrieved_trials == 0:
        return "persistent_retrieval_miss"
    if (
        judge_correct_trials == 0
        and gold_doc_retrieved_trials >= 2
        and gold_answer_span_trials == 0
    ):
        return "gold_doc_present_span_missing"
    if judge_correct_trials == 0 and gold_answer_span_trials > 0:
        return "answer_span_present_persistent_wrong"
    if 0 < judge_correct_trials < 3:
        return "unstable_answer"
    return "persistent_wrong_other"


def run_baseline_stability_audit(
    *,
    summary_paths: tuple[Path, Path, Path],
    diagnostic_paths: tuple[Path, Path, Path],
    judge_paths: tuple[Path, Path, Path],
    gold_slice_path: Path,
    output_path: Path,
) -> BaselineStabilityAudit:
    all_input_paths = tuple(
        path.resolve()
        for path in (*summary_paths, *diagnostic_paths, *judge_paths, gold_slice_path)
    )
    root = _common_repository_root(all_input_paths)
    output_path = output_path.resolve()
    if not output_path.is_relative_to((root / "runs").resolve()):
        raise ValueError("baseline stability output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("baseline stability output already exists")

    summaries = tuple(
        PiSmokeSummary.model_validate_json(path.read_text(encoding="utf-8"))
        for path in all_input_paths[:3]
    )
    diagnostics = tuple(
        DiagnosticSummary.model_validate_json(path.read_text(encoding="utf-8"))
        for path in all_input_paths[3:6]
    )
    judges = tuple(
        DevelopmentJudgeResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in all_input_paths[6:9]
    )
    gold = DevelopmentGoldSlice.model_validate_json(
        all_input_paths[9].read_text(encoding="utf-8")
    )
    query_ids = tuple(item.query_id for item in summaries[0].items)
    expected = set(query_ids)
    if len(expected) != len(query_ids):
        raise ValueError("baseline stability query IDs are not unique")
    for summary in summaries:
        if summary.failed or summary.budget_exhausted:
            raise ValueError("baseline stability requires complete generation trials")
        if {item.query_id for item in summary.items} != expected:
            raise ValueError("baseline summary query IDs differ")
    for diagnostic in diagnostics:
        if {row.query_id for row in diagnostic.rows} != expected:
            raise ValueError("baseline diagnostic query IDs differ")
    for judge in judges:
        if judge.parse_failures or judge.request_failures:
            raise ValueError("baseline Judge result is incomplete")
        if {row.query_id for row in judge.observations} != expected:
            raise ValueError("baseline Judge query IDs differ")
    gold_by_id = {row.query_id: row for row in gold.rows}
    if set(gold_by_id) != expected:
        raise ValueError("baseline gold query IDs differ")

    summary_by_trial = [
        {item.query_id: item for item in summary.items} for summary in summaries
    ]
    diagnostic_by_trial = [
        {row.query_id: row for row in diagnostic.rows} for diagnostic in diagnostics
    ]
    judge_by_trial = [
        {row.query_id: row for row in judge.observations} for judge in judges
    ]
    cases: list[BaselineStabilityCase] = []
    for query_id in query_ids:
        gold_row = gold_by_id[query_id]
        gold_docids = set(gold_row.gold_docids)
        judge_correct = sum(
            judge_by_trial[index][query_id].correct for index in range(3)
        )
        normalized_exact = sum(
            diagnostic_by_trial[index][query_id].normalized_exact_match
            for index in range(3)
        )
        gold_retrieved = sum(
            diagnostic_by_trial[index][query_id].gold_recall > 0
            for index in range(3)
        )
        span_trials = 0
        for index in range(3):
            item = summary_by_trial[index][query_id]
            if item.run_path is None or item.run_sha256 is None:
                raise ValueError(f"baseline run artifact is missing: {query_id}")
            run_path = (root / item.run_path).resolve()
            if _sha256_file(run_path) != item.run_sha256:
                raise ValueError(f"baseline run hash changed: {query_id}")
            run = load_pi_browsecomp_run(run_path)
            answer = gold_row.answer.casefold()
            if any(
                result.docid in gold_docids and answer in result.snippet.casefold()
                for call in run.search_calls
                for result in call.results
            ):
                span_trials += 1
        category = classify_baseline_stability(
            judge_correct_trials=judge_correct,
            gold_doc_retrieved_trials=gold_retrieved,
            gold_answer_span_trials=span_trials,
        )
        cases.append(
            BaselineStabilityCase(
                query_id=query_id,
                category=category,
                judge_correct_trials=judge_correct,
                normalized_exact_trials=normalized_exact,
                gold_doc_retrieved_trials=gold_retrieved,
                gold_answer_span_in_gold_preview_trials=span_trials,
            )
        )
    counts = Counter(case.category for case in cases)
    source_names = (
        "trial1_summary",
        "trial2_summary",
        "trial3_summary",
        "trial1_diagnostic",
        "trial2_diagnostic",
        "trial3_diagnostic",
        "trial1_judge",
        "trial2_judge",
        "trial3_judge",
        "gold_slice",
    )
    result = BaselineStabilityAudit(
        created_at=_utc_now(),
        query_count=len(cases),
        total_observations=3 * len(cases),
        category_counts=dict(counts),
        cases=tuple(cases),
        actionable_case_ids=tuple(
            case.query_id
            for case in cases
            if case.category == "gold_doc_present_span_missing"
        ),
        evidence_recall_warning=(
            "The existing answer-atom evidence_recall diagnostic can count unrelated common "
            "tokens. This audit therefore requires the literal gold answer to occur inside a "
            "retrieved gold-document preview before calling an answer span present."
        ),
        sources={
            name: ArtifactReference(
                path=path.relative_to(root).as_posix(), sha256=_sha256_file(path)
            )
            for name, path in zip(source_names, all_input_paths, strict=True)
        },
        claim_boundary=(
            "This is a development-only, posthoc failure-localization audit over three valid "
            "v10 runs on the same 25 questions. It uses already-exported development gold and "
            "Judge artifacts, makes no new provider or Judge call, excludes the invalid candidate "
            "arm, and does not establish official accuracy or model capability improvement."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _common_repository_root(paths: tuple[Path, ...]) -> Path:
    for parent in paths[0].parents:
        if (parent / "pyproject.toml").is_file() and (parent / "runs").is_dir():
            if all(path.is_relative_to(parent) for path in paths):
                return parent
    raise ValueError("audit inputs do not share a repository root")
