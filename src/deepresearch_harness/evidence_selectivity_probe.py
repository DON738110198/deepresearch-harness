from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_decision import DecisionSource
from .browsecomp_evaluation import DevelopmentGoldSlice, DiagnosticSummary
from .browsecomp_plus import load_pi_browsecomp_run
from .browsecomp_repeats import RepeatComparisonSummary, aggregate_repeat_experiment
from .repeat_development_judge import RepeatDevelopmentJudgeComparison


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSelectivityMetrics(StrictContract):
    search_calls: int = Field(ge=0)
    unique_search_queries: int = Field(ge=0)
    repeated_query_calls: int = Field(ge=0)
    repeated_query_rate_percent: float = Field(ge=0, le=100)
    result_slots: int = Field(ge=0)
    unique_docids: int = Field(ge=0)
    duplicate_result_slots: int = Field(ge=0)
    duplicate_result_rate_percent: float = Field(ge=0, le=100)
    relevant_unique_docids_seen: int = Field(ge=0)
    gold_unique_docids_seen: int = Field(ge=0)
    first_relevant_call: int | None = Field(default=None, ge=1)
    first_gold_call: int | None = Field(default=None, ge=1)
    minimum_relevant_rank: int | None = Field(default=None, ge=1)
    minimum_gold_rank: int | None = Field(default=None, ge=1)
    snippet_characters: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> "RunSelectivityMetrics":
        if self.repeated_query_calls != self.search_calls - self.unique_search_queries:
            raise ValueError("repeated query count is inconsistent")
        if self.duplicate_result_slots != self.result_slots - self.unique_docids:
            raise ValueError("duplicate result count is inconsistent")
        return self


class EvidenceSelectivityObservation(StrictContract):
    trial_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    paired_outcome: Literal[
        "candidate_improvement",
        "candidate_regression",
        "both_correct",
        "both_incorrect",
    ]
    baseline_correct: bool
    candidate_correct: bool
    baseline_evidence_recall: float = Field(ge=0, le=1)
    candidate_evidence_recall: float = Field(ge=0, le=1)
    evidence_recall_delta: float = Field(ge=-1, le=1)
    baseline: RunSelectivityMetrics
    candidate: RunSelectivityMetrics


class EvidenceSelectivityGroup(StrictContract):
    paired_outcome: Literal[
        "candidate_improvement",
        "candidate_regression",
        "both_correct",
        "both_incorrect",
    ]
    observations: int = Field(ge=0)
    mean_evidence_recall_delta: float
    baseline_relevant_found_percent: float = Field(ge=0, le=100)
    candidate_relevant_found_percent: float = Field(ge=0, le=100)
    baseline_mean_search_calls: float = Field(ge=0)
    candidate_mean_search_calls: float = Field(ge=0)
    baseline_mean_duplicate_result_rate_percent: float = Field(ge=0, le=100)
    candidate_mean_duplicate_result_rate_percent: float = Field(ge=0, le=100)
    baseline_mean_repeated_query_rate_percent: float = Field(ge=0, le=100)
    candidate_mean_repeated_query_rate_percent: float = Field(ge=0, le=100)
    baseline_mean_first_relevant_call_when_found: float | None = Field(
        default=None, ge=1
    )
    candidate_mean_first_relevant_call_when_found: float | None = Field(
        default=None, ge=1
    )
    baseline_mean_minimum_relevant_rank_when_found: float | None = Field(
        default=None, ge=1
    )
    candidate_mean_minimum_relevant_rank_when_found: float | None = Field(
        default=None, ge=1
    )
    baseline_mean_input_tokens: float = Field(ge=0)
    candidate_mean_input_tokens: float = Field(ge=0)
    baseline_mean_total_tokens: float = Field(ge=0)
    candidate_mean_total_tokens: float = Field(ge=0)


class EvidenceSelectivityProbe(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-evidence-selectivity-probe-v0"
    ] = "browsecomp-plus-evidence-selectivity-probe-v0"
    created_at: str
    status: Literal["post_generation_diagnostic_not_effectiveness"] = (
        "post_generation_diagnostic_not_effectiveness"
    )
    provider_calls: Literal[0] = 0
    paired_observations: int = Field(gt=0)
    groups: list[EvidenceSelectivityGroup] = Field(min_length=4, max_length=4)
    observations: list[EvidenceSelectivityObservation] = Field(min_length=1)
    sources: dict[str, DecisionSource]
    observed_pattern: Literal[
        "retrieval_gain_with_unresolved_selection_or_synthesis_loss"
    ] = "retrieval_gain_with_unresolved_selection_or_synthesis_loss"
    next_candidate: Literal["evidence_progressive_disclosure_v0"] = (
        "evidence_progressive_disclosure_v0"
    )
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def artifact_is_complete(self) -> "EvidenceSelectivityProbe":
        if self.paired_observations != len(self.observations):
            raise ValueError("probe observation count differs")
        expected_groups = {
            "candidate_improvement",
            "candidate_regression",
            "both_correct",
            "both_incorrect",
        }
        if {group.paired_outcome for group in self.groups} != expected_groups:
            raise ValueError("probe outcome groups are incomplete")
        if sum(group.observations for group in self.groups) != self.paired_observations:
            raise ValueError("probe group counts do not cover observations")
        expected_sources = {
            "registration",
            "repeat_experiment",
            "repeat_comparison",
            "judge_comparison",
            "decision",
        }
        if set(self.sources) != expected_sources:
            raise ValueError("probe source set is incomplete")
        return self


def run_evidence_selectivity_probe(
    *,
    registration_path: Path,
    target_manifest_path: Path,
    output_path: Path,
) -> EvidenceSelectivityProbe:
    repository_root = _find_repository_root(registration_path.resolve())
    registration = _load_object(registration_path)
    if (
        registration.get("schema_version")
        != "browsecomp-plus-evidence-selectivity-probe-registration-v0"
        or registration.get("status") != "registered_before_probe_execution"
    ):
        raise ValueError("evidence selectivity probe is not registered")
    source_contract = _require_object(registration, "source_contract")
    fixed = _require_object(registration, "fixed_analysis")
    acceptance = _require_object(registration, "acceptance")
    if fixed.get("provider_calls") != 0 or acceptance.get(
        "provider_calls_must_equal"
    ) != 0:
        raise ValueError("probe registration permits provider calls")
    if acceptance.get("sealed_holdout_access") != "forbidden":
        raise ValueError("probe registration does not forbid sealed holdout")

    source_paths = {
        name: _resolve_registered_source(source_contract, name, repository_root)
        for name in (
            "repeat_experiment",
            "repeat_comparison",
            "judge_comparison",
            "decision",
        )
    }
    repeat = aggregate_repeat_experiment(
        manifest_path=source_paths["repeat_experiment"],
        target_manifest_path=target_manifest_path.resolve(),
        output_path=source_paths["repeat_comparison"],
        validate_existing=True,
    )
    judge = RepeatDevelopmentJudgeComparison.model_validate_json(
        source_paths["judge_comparison"].read_text(encoding="utf-8")
    )
    decision = _load_object(source_paths["decision"])
    if decision.get("decision") != "reject" or decision.get("next_action") != (
        "diagnose_evidence_selection_and_synthesis_from_saved_runs"
    ):
        raise ValueError("probe is bound to another decision state")
    if judge.repeat_comparison_sha256 != sha256(
        source_paths["repeat_comparison"].read_bytes()
    ).hexdigest():
        raise ValueError("Judge comparison targets another repeat comparison")

    judge_rows = {
        (row.trial_id, row.query_id, row.variant): row for row in judge.observations
    }
    observations: list[EvidenceSelectivityObservation] = []
    for trial in repeat.trials:
        diagnostics = {
            variant: _load_diagnostic(
                getattr(trial, f"{variant}_diagnostic"), repository_root
            )
            for variant in ("baseline", "candidate")
        }
        gold = _load_gold(trial.candidate_gold_slice, repository_root)
        gold_rows = {row.query_id: row for row in gold.rows}
        diagnostic_rows = {
            variant: {row.query_id: row for row in summary.rows}
            for variant, summary in diagnostics.items()
        }
        query_ids = sorted(diagnostic_rows["baseline"])
        if query_ids != sorted(diagnostic_rows["candidate"]) or query_ids != sorted(
            gold_rows
        ):
            raise ValueError("probe trial query sets differ")
        for query_id in query_ids:
            baseline_judge = judge_rows[(trial.trial_id, query_id, "baseline")]
            candidate_judge = judge_rows[(trial.trial_id, query_id, "candidate")]
            baseline_diagnostic = diagnostic_rows["baseline"][query_id]
            candidate_diagnostic = diagnostic_rows["candidate"][query_id]
            outcome = _paired_outcome(
                baseline_judge.correct, candidate_judge.correct
            )
            evidence_docids = set(gold_rows[query_id].evidence_docids)
            gold_docids = set(gold_rows[query_id].gold_docids)
            observations.append(
                EvidenceSelectivityObservation(
                    trial_id=trial.trial_id,
                    query_id=query_id,
                    paired_outcome=outcome,
                    baseline_correct=baseline_judge.correct,
                    candidate_correct=candidate_judge.correct,
                    baseline_evidence_recall=float(
                        baseline_diagnostic.evidence_recall or 0.0
                    ),
                    candidate_evidence_recall=float(
                        candidate_diagnostic.evidence_recall or 0.0
                    ),
                    evidence_recall_delta=round(
                        float(candidate_diagnostic.evidence_recall or 0.0)
                        - float(baseline_diagnostic.evidence_recall or 0.0),
                        6,
                    ),
                    baseline=_run_metrics(
                        _run_path(trial.baseline_summary.path, query_id, repository_root),
                        evidence_docids,
                        gold_docids,
                    ),
                    candidate=_run_metrics(
                        _run_path(trial.candidate_summary.path, query_id, repository_root),
                        evidence_docids,
                        gold_docids,
                    ),
                )
            )

    expected = int(acceptance["paired_observations_must_equal"])
    if len(observations) != expected or repeat.paired_query_observations != expected:
        raise ValueError("probe does not cover the registered paired observations")
    groups = [
        _aggregate_group(outcome, observations)
        for outcome in (
            "candidate_improvement",
            "candidate_regression",
            "both_correct",
            "both_incorrect",
        )
    ]
    artifact = EvidenceSelectivityProbe(
        created_at=datetime.now(timezone.utc).isoformat(),
        paired_observations=len(observations),
        groups=groups,
        observations=observations,
        sources={
            "registration": _source(registration_path, repository_root),
            **{
                name: _source(path, repository_root)
                for name, path in source_paths.items()
            },
        },
        claim_boundary=(
            "Post-generation development diagnostic over saved traces. It made "
            "zero provider calls, is not an effectiveness experiment, and does "
            "not support a model-capability or leaderboard claim."
        ),
    )
    if output_path.exists():
        raise ValueError("evidence selectivity probe output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return artifact


def _run_metrics(
    path: Path, evidence_docids: set[str], gold_docids: set[str]
) -> RunSelectivityMetrics:
    run = load_pi_browsecomp_run(path)
    normalized_queries = [" ".join(call.query.lower().split()) for call in run.search_calls]
    result_docids = [
        result.docid for call in run.search_calls for result in call.results
    ]
    unique_docids = set(result_docids)
    first_relevant_call = _first_matching_call(run.search_calls, evidence_docids)
    first_gold_call = _first_matching_call(run.search_calls, gold_docids)
    return RunSelectivityMetrics(
        search_calls=len(run.search_calls),
        unique_search_queries=len(set(normalized_queries)),
        repeated_query_calls=len(normalized_queries) - len(set(normalized_queries)),
        repeated_query_rate_percent=_percent(
            len(normalized_queries) - len(set(normalized_queries)),
            len(normalized_queries),
        ),
        result_slots=len(result_docids),
        unique_docids=len(unique_docids),
        duplicate_result_slots=len(result_docids) - len(unique_docids),
        duplicate_result_rate_percent=_percent(
            len(result_docids) - len(unique_docids), len(result_docids)
        ),
        relevant_unique_docids_seen=len(unique_docids & evidence_docids),
        gold_unique_docids_seen=len(unique_docids & gold_docids),
        first_relevant_call=first_relevant_call,
        first_gold_call=first_gold_call,
        minimum_relevant_rank=_minimum_matching_rank(
            run.search_calls, evidence_docids
        ),
        minimum_gold_rank=_minimum_matching_rank(run.search_calls, gold_docids),
        snippet_characters=sum(
            len(result.snippet)
            for call in run.search_calls
            for result in call.results
        ),
        input_tokens=run.usage.input_tokens,
        output_tokens=run.usage.output_tokens,
        total_tokens=run.usage.total_tokens,
        latency_ms=run.latency_ms,
    )


def _first_matching_call(calls: list[object], docids: set[str]) -> int | None:
    for call_index, call in enumerate(calls, start=1):
        if any(result.docid in docids for result in call.results):
            return call_index
    return None


def _minimum_matching_rank(calls: list[object], docids: set[str]) -> int | None:
    ranks = [
        rank
        for call in calls
        for rank, result in enumerate(call.results, start=1)
        if result.docid in docids
    ]
    return min(ranks) if ranks else None


def _aggregate_group(
    outcome: str, observations: list[EvidenceSelectivityObservation]
) -> EvidenceSelectivityGroup:
    rows = [row for row in observations if row.paired_outcome == outcome]
    if not rows:
        return EvidenceSelectivityGroup(
            paired_outcome=outcome,
            observations=0,
            mean_evidence_recall_delta=0,
            baseline_relevant_found_percent=0,
            candidate_relevant_found_percent=0,
            baseline_mean_search_calls=0,
            candidate_mean_search_calls=0,
            baseline_mean_duplicate_result_rate_percent=0,
            candidate_mean_duplicate_result_rate_percent=0,
            baseline_mean_repeated_query_rate_percent=0,
            candidate_mean_repeated_query_rate_percent=0,
            baseline_mean_input_tokens=0,
            candidate_mean_input_tokens=0,
            baseline_mean_total_tokens=0,
            candidate_mean_total_tokens=0,
        )
    return EvidenceSelectivityGroup(
        paired_outcome=outcome,
        observations=len(rows),
        mean_evidence_recall_delta=_mean(row.evidence_recall_delta for row in rows),
        baseline_relevant_found_percent=_percent(
            sum(row.baseline.relevant_unique_docids_seen > 0 for row in rows), len(rows)
        ),
        candidate_relevant_found_percent=_percent(
            sum(row.candidate.relevant_unique_docids_seen > 0 for row in rows), len(rows)
        ),
        baseline_mean_search_calls=_mean(row.baseline.search_calls for row in rows),
        candidate_mean_search_calls=_mean(row.candidate.search_calls for row in rows),
        baseline_mean_duplicate_result_rate_percent=_mean(
            row.baseline.duplicate_result_rate_percent for row in rows
        ),
        candidate_mean_duplicate_result_rate_percent=_mean(
            row.candidate.duplicate_result_rate_percent for row in rows
        ),
        baseline_mean_repeated_query_rate_percent=_mean(
            row.baseline.repeated_query_rate_percent for row in rows
        ),
        candidate_mean_repeated_query_rate_percent=_mean(
            row.candidate.repeated_query_rate_percent for row in rows
        ),
        baseline_mean_first_relevant_call_when_found=_optional_mean(
            row.baseline.first_relevant_call for row in rows
        ),
        candidate_mean_first_relevant_call_when_found=_optional_mean(
            row.candidate.first_relevant_call for row in rows
        ),
        baseline_mean_minimum_relevant_rank_when_found=_optional_mean(
            row.baseline.minimum_relevant_rank for row in rows
        ),
        candidate_mean_minimum_relevant_rank_when_found=_optional_mean(
            row.candidate.minimum_relevant_rank for row in rows
        ),
        baseline_mean_input_tokens=_mean(row.baseline.input_tokens for row in rows),
        candidate_mean_input_tokens=_mean(row.candidate.input_tokens for row in rows),
        baseline_mean_total_tokens=_mean(row.baseline.total_tokens for row in rows),
        candidate_mean_total_tokens=_mean(row.candidate.total_tokens for row in rows),
    )


def _paired_outcome(
    baseline_correct: bool, candidate_correct: bool
) -> Literal[
    "candidate_improvement",
    "candidate_regression",
    "both_correct",
    "both_incorrect",
]:
    if candidate_correct and not baseline_correct:
        return "candidate_improvement"
    if baseline_correct and not candidate_correct:
        return "candidate_regression"
    return "both_correct" if candidate_correct else "both_incorrect"


def _load_diagnostic(source: object, repository_root: Path) -> DiagnosticSummary:
    path = _resolve_run_source(source, repository_root)
    return DiagnosticSummary.model_validate_json(path.read_text(encoding="utf-8"))


def _load_gold(source: object, repository_root: Path) -> DevelopmentGoldSlice:
    path = _resolve_run_source(source, repository_root)
    return DevelopmentGoldSlice.model_validate_json(path.read_text(encoding="utf-8"))


def _resolve_run_source(source: object, repository_root: Path) -> Path:
    path = _resolve_repository_path(source.path, repository_root)
    if sha256(path.read_bytes()).hexdigest() != source.sha256:
        raise ValueError("probe source hash changed after repeat aggregation")
    return path


def _run_path(summary_path: str, query_id: str, repository_root: Path) -> Path:
    summary = _resolve_repository_path(summary_path, repository_root)
    run_path = summary.parent / query_id / "run.json"
    if not run_path.is_file():
        raise ValueError(f"probe run is missing: {run_path}")
    return run_path


def _resolve_registered_source(
    source_contract: dict[str, object], name: str, repository_root: Path
) -> Path:
    path_value = source_contract.get(f"{name}_path")
    hash_value = source_contract.get(f"{name}_sha256")
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise ValueError(f"probe registration lacks {name} source")
    path = _resolve_repository_path(path_value, repository_root)
    if sha256(path.read_bytes()).hexdigest() != hash_value:
        raise ValueError(f"registered {name} hash changed")
    return path


def _resolve_repository_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("probe source path must be repository-relative")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to(repository_root.resolve()) or not resolved.is_file():
        raise ValueError(f"probe source is missing or outside repository: {value}")
    return resolved


def _percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 6) if denominator else 0.0


def _mean(values: object) -> float:
    materialized = list(values)
    return round(mean(materialized), 6) if materialized else 0.0


def _optional_mean(values: object) -> float | None:
    materialized = [value for value in values if value is not None]
    return round(mean(materialized), 6) if materialized else None


def _source(path: Path, repository_root: Path) -> DecisionSource:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return DecisionSource(path=display, sha256=sha256(resolved.read_bytes()).hexdigest())


def _require_object(
    payload: dict[str, object], field_name: str
) -> dict[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"probe registration lacks {field_name}")
    return value


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("probe source must be a JSON object")
    return value


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "runs"
        ).is_dir():
            return candidate
    raise ValueError("could not locate repository root for evidence selectivity probe")
