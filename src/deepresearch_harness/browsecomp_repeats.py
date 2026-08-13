from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import (
    DevelopmentGoldSlice,
    DiagnosticSummary,
    _prediction_set_sha256,
    extract_exact_answer,
    normalize_exact_answer,
)
from .browsecomp_plus import (
    PiBrowseCompRun,
    load_pi_browsecomp_run,
    normalized_text_file_sha256,
)
from .pi_browsecomp import (
    OfficialRunExportManifest,
    PiSmokeItem,
    PiSmokeSummary,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepeatVariantInput(StrictContract):
    summary_path: str = Field(min_length=1)
    gold_slice_path: str = Field(min_length=1)
    diagnostic_path: str = Field(min_length=1)
    official_export_manifest_path: str = Field(min_length=1)


class RepeatPairInput(StrictContract):
    trial_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    execution_order: Literal["baseline_first", "candidate_first"]
    baseline: RepeatVariantInput
    candidate: RepeatVariantInput


class RepeatExperimentManifest(StrictContract):
    schema_version: Literal["browsecomp-plus-repeat-experiment-v0"] = (
        "browsecomp-plus-repeat-experiment-v0"
    )
    registered_at: str = Field(min_length=1)
    registration_status: Literal[
        "pre_generation", "reconstructed_after_interruption"
    ]
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_adapter_version: Literal["pi-browsecomp-v6"] = "pi-browsecomp-v6"
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    control_policy: Literal["answer_reserve_nonthinking_v0"]
    baseline_retriever_id: Literal["bm25"] = "bm25"
    candidate_retriever_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    candidate_retriever_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_trials: int = Field(default=3, ge=3)
    pairs: list[RepeatPairInput] = Field(min_length=3)

    @model_validator(mode="after")
    def pairs_are_independent_and_unique(self) -> "RepeatExperimentManifest":
        if len(self.pairs) < self.minimum_trials:
            raise ValueError("repeat manifest does not meet its minimum trial count")
        if self.candidate_retriever_id == self.baseline_retriever_id:
            raise ValueError("candidate retriever must differ from the baseline")
        trial_ids = [pair.trial_id for pair in self.pairs]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("repeat trial IDs must be unique")
        orders = [pair.execution_order for pair in self.pairs]
        if any(left == right for left, right in zip(orders, orders[1:])):
            raise ValueError("repeat execution order must alternate between variants")
        paths = [
            path
            for pair in self.pairs
            for path in (
                pair.baseline.summary_path,
                pair.baseline.gold_slice_path,
                pair.baseline.diagnostic_path,
                pair.baseline.official_export_manifest_path,
                pair.candidate.summary_path,
                pair.candidate.gold_slice_path,
                pair.candidate.diagnostic_path,
                pair.candidate.official_export_manifest_path,
            )
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("repeat trials cannot reuse source artifacts")
        return self


class SourceArtifact(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrialVariantMetrics(StrictContract):
    schema_complete_percent: float = Field(ge=0, le=100)
    strict_exact_percent: float = Field(ge=0, le=100)
    evidence_recall_percent: float = Field(ge=0, le=100)
    gold_recall_percent: float = Field(ge=0, le=100)
    search_calls_per_query: float = Field(ge=0)
    output_tokens_per_query: float = Field(ge=0)
    total_tokens_per_query: float = Field(ge=0)
    cost_usd_per_query: float = Field(ge=0)
    latency_ms_per_query: float = Field(ge=0)


class RepeatTrialResult(StrictContract):
    trial_id: str
    execution_order: Literal["baseline_first", "candidate_first"]
    baseline_summary: SourceArtifact
    baseline_gold_slice: SourceArtifact
    baseline_diagnostic: SourceArtifact
    baseline_official_export_manifest: SourceArtifact
    candidate_summary: SourceArtifact
    candidate_gold_slice: SourceArtifact
    candidate_diagnostic: SourceArtifact
    candidate_official_export_manifest: SourceArtifact
    baseline: TrialVariantMetrics
    candidate: TrialVariantMetrics


class MetricDistribution(StrictContract):
    trials: int = Field(ge=3)
    mean: float
    sample_stddev: float = Field(ge=0)
    minimum: float
    maximum: float


class VariantRepeatAggregate(StrictContract):
    retriever_id: str
    retriever_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    schema_complete_percent: MetricDistribution
    strict_exact_percent: MetricDistribution
    evidence_recall_percent: MetricDistribution
    gold_recall_percent: MetricDistribution
    search_calls_per_query: MetricDistribution
    output_tokens_per_query: MetricDistribution
    total_tokens_per_query: MetricDistribution
    cost_usd_per_query: MetricDistribution
    latency_ms_per_query: MetricDistribution
    total_cost_usd: float = Field(ge=0)


class PairedMetricSummary(StrictContract):
    metric: Literal[
        "schema_complete",
        "strict_exact",
        "evidence_recall",
        "gold_recall",
        "search_calls",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "latency_ms",
    ]
    preferred_direction: Literal["higher", "lower"]
    comparisons: int = Field(gt=0)
    candidate_wins: int = Field(ge=0)
    baseline_wins: int = Field(ge=0)
    ties: int = Field(ge=0)

    @model_validator(mode="after")
    def outcomes_sum_to_comparisons(self) -> "PairedMetricSummary":
        if self.candidate_wins + self.baseline_wins + self.ties != self.comparisons:
            raise ValueError("paired outcomes do not sum to comparison count")
        return self


class RepeatComparisonSummary(StrictContract):
    schema_version: Literal["browsecomp-plus-repeat-comparison-v0"] = (
        "browsecomp-plus-repeat-comparison-v0"
    )
    created_at: str
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    official_accuracy_status: Literal["planned_not_run"] = "planned_not_run"
    registration_status: Literal[
        "pre_generation", "reconstructed_after_interruption"
    ]
    experiment_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_version: Literal["pi-browsecomp-v6"] = "pi-browsecomp-v6"
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"]
    control_policy: Literal["answer_reserve_nonthinking_v0"]
    trial_count: int = Field(ge=3)
    queries_per_trial: int = Field(gt=0)
    paired_query_observations: int = Field(gt=0)
    baseline: VariantRepeatAggregate
    candidate: VariantRepeatAggregate
    paired_metrics: list[PairedMetricSummary] = Field(min_length=1)
    trials: list[RepeatTrialResult] = Field(min_length=3)


@dataclass(frozen=True)
class _QueryObservation:
    query_id: str
    schema_complete: float
    strict_exact: float
    evidence_recall: float
    gold_recall: float
    search_calls: float
    output_tokens: float
    total_tokens: float
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class _LoadedVariant:
    summary_ref: SourceArtifact
    gold_slice_ref: SourceArtifact
    diagnostic_ref: SourceArtifact
    official_export_manifest_ref: SourceArtifact
    summary: PiSmokeSummary
    metrics: TrialVariantMetrics
    observations: dict[str, _QueryObservation]
    prompt_hashes: dict[str, str]
    run_ids: set[str]
    gold_rows_sha256: str


def aggregate_repeat_experiment(
    *,
    manifest_path: Path,
    target_manifest_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> RepeatComparisonSummary:
    repository_root = _find_repository_root(manifest_path.resolve())
    _require_under_runs(manifest_path, repository_root)
    _require_under_runs(output_path, repository_root)
    existing: RepeatComparisonSummary | None = None
    if output_path.exists() and not validate_existing:
        raise ValueError("repeat comparison output already exists")
    if output_path.exists():
        existing = RepeatComparisonSummary.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

    manifest = RepeatExperimentManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    target_hash = normalized_text_file_sha256(target_manifest_path)
    if manifest.target_manifest_sha256 != target_hash:
        raise ValueError("repeat manifest targets a different benchmark manifest")

    trial_results: list[RepeatTrialResult] = []
    baseline_loaded: list[_LoadedVariant] = []
    candidate_loaded: list[_LoadedVariant] = []
    expected_query_ids: tuple[str, ...] | None = None
    expected_query_file_hash: str | None = None
    expected_gold_rows_hash: str | None = None
    all_run_ids: set[str] = set()

    for pair in manifest.pairs:
        baseline = _load_variant(
            source=pair.baseline,
            repository_root=repository_root,
            expected_model=manifest.model,
            expected_control_policy=manifest.control_policy,
            expected_adapter_version=manifest.expected_adapter_version,
            expected_retriever_id=manifest.baseline_retriever_id,
            expected_retriever_manifest_sha256=None,
            expected_target_manifest_sha256=target_hash,
        )
        candidate = _load_variant(
            source=pair.candidate,
            repository_root=repository_root,
            expected_model=manifest.model,
            expected_control_policy=manifest.control_policy,
            expected_adapter_version=manifest.expected_adapter_version,
            expected_retriever_id=manifest.candidate_retriever_id,
            expected_retriever_manifest_sha256=(
                manifest.candidate_retriever_manifest_sha256
            ),
            expected_target_manifest_sha256=target_hash,
        )
        query_ids = tuple(sorted(baseline.observations))
        if query_ids != tuple(sorted(candidate.observations)):
            raise ValueError(f"paired query IDs differ in trial {pair.trial_id}")
        if baseline.prompt_hashes != candidate.prompt_hashes:
            raise ValueError(f"paired prompts differ in trial {pair.trial_id}")
        if baseline.gold_rows_sha256 != candidate.gold_rows_sha256:
            raise ValueError(f"paired gold rows differ in trial {pair.trial_id}")
        if baseline.summary.development_queries_sha256 != (
            candidate.summary.development_queries_sha256
        ):
            raise ValueError(f"paired query files differ in trial {pair.trial_id}")
        if expected_query_ids is None:
            expected_query_ids = query_ids
            expected_query_file_hash = baseline.summary.development_queries_sha256
            expected_gold_rows_hash = baseline.gold_rows_sha256
        elif query_ids != expected_query_ids or (
            baseline.summary.development_queries_sha256 != expected_query_file_hash
        ):
            raise ValueError("repeat trials do not use one frozen query set")
        elif baseline.gold_rows_sha256 != expected_gold_rows_hash:
            raise ValueError("repeat trials do not use one frozen gold set")
        duplicate_run_ids = all_run_ids & (baseline.run_ids | candidate.run_ids)
        if duplicate_run_ids:
            raise ValueError("repeat trials reuse provider run IDs")
        all_run_ids.update(baseline.run_ids)
        all_run_ids.update(candidate.run_ids)

        baseline_loaded.append(baseline)
        candidate_loaded.append(candidate)
        trial_results.append(
            RepeatTrialResult(
                trial_id=pair.trial_id,
                execution_order=pair.execution_order,
                baseline_summary=baseline.summary_ref,
                baseline_gold_slice=baseline.gold_slice_ref,
                baseline_diagnostic=baseline.diagnostic_ref,
                baseline_official_export_manifest=(
                    baseline.official_export_manifest_ref
                ),
                candidate_summary=candidate.summary_ref,
                candidate_gold_slice=candidate.gold_slice_ref,
                candidate_diagnostic=candidate.diagnostic_ref,
                candidate_official_export_manifest=(
                    candidate.official_export_manifest_ref
                ),
                baseline=baseline.metrics,
                candidate=candidate.metrics,
            )
        )

    assert expected_query_ids is not None
    paired_metrics = _paired_metrics(baseline_loaded, candidate_loaded)
    artifact = RepeatComparisonSummary(
        created_at=(
            existing.created_at
            if existing is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        experiment_manifest_sha256=normalized_text_file_sha256(manifest_path),
        registration_status=manifest.registration_status,
        target_manifest_sha256=target_hash,
        model=manifest.model,
        control_policy=manifest.control_policy,
        trial_count=len(manifest.pairs),
        queries_per_trial=len(expected_query_ids),
        paired_query_observations=len(manifest.pairs) * len(expected_query_ids),
        baseline=_aggregate_variant(
            baseline_loaded,
            retriever_id=manifest.baseline_retriever_id,
            retriever_manifest_sha256=None,
        ),
        candidate=_aggregate_variant(
            candidate_loaded,
            retriever_id=manifest.candidate_retriever_id,
            retriever_manifest_sha256=(
                manifest.candidate_retriever_manifest_sha256
            ),
        ),
        paired_metrics=paired_metrics,
        trials=trial_results,
    )
    if existing is not None:
        if artifact != existing:
            raise ValueError("existing repeat comparison no longer matches its sources")
        return existing
    _atomic_write(output_path, artifact.model_dump_json(indent=2))
    return artifact


def _load_variant(
    *,
    source: RepeatVariantInput,
    repository_root: Path,
    expected_model: str,
    expected_control_policy: str,
    expected_adapter_version: str,
    expected_retriever_id: str,
    expected_retriever_manifest_sha256: str | None,
    expected_target_manifest_sha256: str,
) -> _LoadedVariant:
    summary_path = _resolve_runs_path(source.summary_path, repository_root)
    gold_slice_path = _resolve_runs_path(source.gold_slice_path, repository_root)
    diagnostic_path = _resolve_runs_path(source.diagnostic_path, repository_root)
    export_manifest_path = _resolve_runs_path(
        source.official_export_manifest_path, repository_root
    )
    summary_bytes = summary_path.read_bytes()
    gold_bytes = gold_slice_path.read_bytes()
    diagnostic_bytes = diagnostic_path.read_bytes()
    export_manifest_bytes = export_manifest_path.read_bytes()
    summary = PiSmokeSummary.model_validate_json(summary_bytes)
    gold = DevelopmentGoldSlice.model_validate_json(gold_bytes)
    diagnostic = DiagnosticSummary.model_validate_json(diagnostic_bytes)
    export_manifest = OfficialRunExportManifest.model_validate_json(
        export_manifest_bytes
    )
    if diagnostic.source_summary_sha256 != sha256(summary_bytes).hexdigest():
        raise ValueError("diagnostic is not bound to its repeat summary")
    if gold.source_summary_sha256 != sha256(summary_bytes).hexdigest():
        raise ValueError("gold slice is not bound to its repeat summary")
    if diagnostic.gold_slice_sha256 != sha256(gold_bytes).hexdigest():
        raise ValueError("diagnostic is not bound to its repeat gold slice")
    if gold.prediction_set_sha256 != _prediction_set_sha256(summary):
        raise ValueError("gold slice is not bound to the frozen prediction set")
    if summary.target_manifest_sha256 != expected_target_manifest_sha256:
        raise ValueError("repeat summary targets a different benchmark")
    if gold.target_manifest_sha256 != expected_target_manifest_sha256:
        raise ValueError("repeat gold slice targets a different benchmark")
    if summary.model != expected_model or summary.control_policy != expected_control_policy:
        raise ValueError("repeat summary changes the frozen model or control policy")
    if summary.retriever_id != expected_retriever_id:
        raise ValueError("repeat summary uses the wrong retriever")
    if summary.retriever_manifest_sha256 != expected_retriever_manifest_sha256:
        raise ValueError("repeat summary retriever manifest hash differs")
    if summary.total_output_budget_overshoot_tokens != 0:
        raise ValueError("repeat summary contains output-budget overshoot")
    _validate_official_export(
        export_manifest,
        export_manifest_path=export_manifest_path,
        summary=summary,
        summary_bytes=summary_bytes,
        expected_target_manifest_sha256=expected_target_manifest_sha256,
    )

    diagnostic_by_id = {row.query_id: row for row in diagnostic.rows}
    gold_by_id = {row.query_id: row for row in gold.rows}
    if set(diagnostic_by_id) != {item.query_id for item in summary.items}:
        raise ValueError("diagnostic and summary query IDs differ")
    if set(gold_by_id) != {item.query_id for item in summary.items}:
        raise ValueError("gold slice and summary query IDs differ")
    _validate_diagnostic_totals(diagnostic)
    observations: dict[str, _QueryObservation] = {}
    prompt_hashes: dict[str, str] = {}
    run_ids: set[str] = set()
    for item in summary.items:
        run_path = summary_path.parent / _safe_id(item.query_id) / "run.json"
        run_bytes = run_path.read_bytes()
        if item.run_sha256 != sha256(run_bytes).hexdigest():
            raise ValueError(f"repeat run hash mismatch for query {item.query_id}")
        run = load_pi_browsecomp_run(run_path)
        _validate_v6_run(
            run,
            item=item,
            expected_model=expected_model,
            expected_control_policy=expected_control_policy,
            expected_adapter_version=expected_adapter_version,
        )
        if run.run_id in run_ids:
            raise ValueError("repeat summary contains duplicate run IDs")
        run_ids.add(run.run_id)
        row = diagnostic_by_id[item.query_id]
        reference = gold_by_id[item.query_id]
        if row.evidence_recall is None or row.gold_recall is None:
            raise ValueError("repeat diagnostic must contain retrieval recall")
        retrieved = {
            result.docid for call in run.search_calls for result in call.results
        }
        predicted_answer = extract_exact_answer(run.answer_text)
        expected_exact = (
            predicted_answer is not None
            and normalize_exact_answer(predicted_answer)
            == normalize_exact_answer(reference.answer)
        )
        expected_evidence_recall = _recall(retrieved, set(reference.evidence_docids))
        expected_gold_recall = _recall(retrieved, set(reference.gold_docids))
        if (
            row.status != run.status
            or row.answer_schema_complete != run.answer_schema_complete
            or row.exact_answer_extracted != (predicted_answer is not None)
            or row.normalized_exact_match != expected_exact
            or row.search_calls != len(run.search_calls)
            or row.prediction_sha256 != item.prediction_sha256
            or row.reference_answer_sha256
            != sha256(reference.answer.encode("utf-8")).hexdigest()
            or not _optional_float_equal(row.evidence_recall, expected_evidence_recall)
            or not _optional_float_equal(row.gold_recall, expected_gold_recall)
        ):
            raise ValueError("repeat diagnostic row does not recompute from run and gold")
        observations[item.query_id] = _QueryObservation(
            query_id=item.query_id,
            schema_complete=float(row.answer_schema_complete),
            strict_exact=float(row.normalized_exact_match),
            evidence_recall=row.evidence_recall,
            gold_recall=row.gold_recall,
            search_calls=float(item.search_calls),
            output_tokens=float(item.output_tokens),
            total_tokens=float(item.total_tokens),
            cost_usd=item.cost_usd,
            latency_ms=float(item.latency_ms),
        )
        prompt_hashes[item.query_id] = run.prompt_sha256

    return _LoadedVariant(
        summary_ref=_source_ref(summary_path, repository_root),
        gold_slice_ref=_source_ref(gold_slice_path, repository_root),
        diagnostic_ref=_source_ref(diagnostic_path, repository_root),
        official_export_manifest_ref=_source_ref(
            export_manifest_path, repository_root
        ),
        summary=summary,
        metrics=_trial_metrics(summary, diagnostic),
        observations=observations,
        prompt_hashes=prompt_hashes,
        run_ids=run_ids,
        gold_rows_sha256=_gold_rows_sha256(gold),
    )


def _validate_official_export(
    export_manifest: OfficialRunExportManifest,
    *,
    export_manifest_path: Path,
    summary: PiSmokeSummary,
    summary_bytes: bytes,
    expected_target_manifest_sha256: str,
) -> None:
    if export_manifest.source_summary_sha256 != sha256(summary_bytes).hexdigest():
        raise ValueError("official export is not bound to its repeat summary")
    if export_manifest.target_manifest_sha256 != expected_target_manifest_sha256:
        raise ValueError("official export targets a different benchmark")
    if export_manifest.completed != summary.query_count or export_manifest.incomplete:
        raise ValueError("repeat official export must contain only completed predictions")
    summary_by_id = {item.query_id: item for item in summary.items}
    if set(summary_by_id) != {item.query_id for item in export_manifest.items}:
        raise ValueError("official export and repeat summary query IDs differ")
    for export_item in export_manifest.items:
        summary_item = summary_by_id[export_item.query_id]
        exported_path = (
            export_manifest_path.parent / "inputs" / f"{_safe_id(export_item.query_id)}.json"
        )
        exported_bytes = exported_path.read_bytes()
        if sha256(exported_bytes).hexdigest() != export_item.exported_sha256:
            raise ValueError("official evaluator input hash mismatch")
        if export_item.prediction_sha256 != summary_item.prediction_sha256:
            raise ValueError("official export prediction hash differs from repeat summary")
        exported = json.loads(exported_bytes)
        result = exported.get("result")
        terminal_result = result[-1] if isinstance(result, list) and result else None
        if (
            exported.get("query_id") != export_item.query_id
            or exported.get("status") != "completed"
            or not isinstance(result, list)
            or not result
            or not isinstance(terminal_result, dict)
            or terminal_result.get("type") != "output_text"
        ):
            raise ValueError("official evaluator input has an invalid completed shape")
        exported_prediction = terminal_result.get("output")
        if not isinstance(exported_prediction, str) or sha256(
            exported_prediction.encode("utf-8")
        ).hexdigest() != export_item.prediction_sha256:
            raise ValueError("official evaluator input prediction text mismatch")


def _validate_v6_run(
    run: PiBrowseCompRun,
    *,
    item: PiSmokeItem,
    expected_model: str,
    expected_control_policy: str,
    expected_adapter_version: str,
) -> None:
    if run.adapter_version != expected_adapter_version:
        raise ValueError("repeat run does not use the registered adapter version")
    if run.model != expected_model or run.control_policy != expected_control_policy:
        raise ValueError("repeat run changes the frozen model or control policy")
    if run.output_budget_overshoot_tokens != 0 or run.usage.output_tokens > 10_000:
        raise ValueError("repeat run violates the 10k output-token contract")
    if not run.model_requests or not run.provider_request_limits:
        raise ValueError("repeat run lacks provider request audit records")
    if run.query_id != item.query_id or run.status != item.status:
        raise ValueError("repeat run and summary item disagree")
    if run.answer_schema_complete != item.answer_schema_complete:
        raise ValueError("repeat run and summary schema flags disagree")
    if item.prediction_sha256 != sha256(run.answer_text.encode("utf-8")).hexdigest():
        raise ValueError("repeat prediction hash mismatch")


def _validate_diagnostic_totals(diagnostic: DiagnosticSummary) -> None:
    rows = diagnostic.rows
    evidence = [row.evidence_recall for row in rows if row.evidence_recall is not None]
    gold = [row.gold_recall for row in rows if row.gold_recall is not None]
    exact = sum(row.normalized_exact_match for row in rows)
    expected = {
        "query_count": len(rows),
        "schema_complete": sum(row.answer_schema_complete for row in rows),
        "exact_answer_extracted": sum(row.exact_answer_extracted for row in rows),
        "normalized_exact_match": exact,
        "normalized_exact_match_percent": round(exact / len(rows) * 100, 2),
        "evidence_recall_percent": (
            round(sum(evidence) / len(evidence) * 100, 2) if evidence else None
        ),
        "gold_recall_percent": (
            round(sum(gold) / len(gold) * 100, 2) if gold else None
        ),
    }
    for field_name, expected_value in expected.items():
        actual_value = getattr(diagnostic, field_name)
        if isinstance(expected_value, float):
            matches = isinstance(actual_value, (int, float)) and math.isclose(
                actual_value, expected_value, abs_tol=1e-9
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(f"repeat diagnostic {field_name} does not match rows")


def _gold_rows_sha256(gold: DevelopmentGoldSlice) -> str:
    canonical = json.dumps(
        [row.model_dump(mode="json") for row in sorted(gold.rows, key=lambda row: row.query_id)],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _recall(retrieved: set[str], relevant: set[str]) -> float | None:
    return len(retrieved & relevant) / len(relevant) if relevant else None


def _optional_float_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, abs_tol=1e-12)


def _trial_metrics(
    summary: PiSmokeSummary, diagnostic: DiagnosticSummary
) -> TrialVariantMetrics:
    count = summary.query_count
    if diagnostic.evidence_recall_percent is None or diagnostic.gold_recall_percent is None:
        raise ValueError("repeat diagnostic is missing macro recall")
    return TrialVariantMetrics(
        schema_complete_percent=round(diagnostic.schema_complete / count * 100, 6),
        strict_exact_percent=diagnostic.normalized_exact_match_percent,
        evidence_recall_percent=diagnostic.evidence_recall_percent,
        gold_recall_percent=diagnostic.gold_recall_percent,
        search_calls_per_query=round(summary.total_search_calls / count, 6),
        output_tokens_per_query=round(summary.total_output_tokens / count, 6),
        total_tokens_per_query=round(summary.total_tokens / count, 6),
        cost_usd_per_query=round(summary.total_cost_usd / count, 12),
        latency_ms_per_query=round(summary.total_latency_ms / count, 6),
    )


def _aggregate_variant(
    loaded: list[_LoadedVariant],
    *,
    retriever_id: str,
    retriever_manifest_sha256: str | None,
) -> VariantRepeatAggregate:
    fields = TrialVariantMetrics.model_fields
    distributions = {
        field_name: _distribution(
            [getattr(item.metrics, field_name) for item in loaded]
        )
        for field_name in fields
    }
    return VariantRepeatAggregate(
        retriever_id=retriever_id,
        retriever_manifest_sha256=retriever_manifest_sha256,
        **distributions,
        total_cost_usd=round(sum(item.summary.total_cost_usd for item in loaded), 12),
    )


def _paired_metrics(
    baseline: list[_LoadedVariant], candidate: list[_LoadedVariant]
) -> list[PairedMetricSummary]:
    directions: dict[str, Literal["higher", "lower"]] = {
        "schema_complete": "higher",
        "strict_exact": "higher",
        "evidence_recall": "higher",
        "gold_recall": "higher",
        "search_calls": "lower",
        "output_tokens": "lower",
        "total_tokens": "lower",
        "cost_usd": "lower",
        "latency_ms": "lower",
    }
    counts = {metric: [0, 0, 0] for metric in directions}
    for baseline_trial, candidate_trial in zip(baseline, candidate, strict=True):
        for query_id in sorted(baseline_trial.observations):
            baseline_row = baseline_trial.observations[query_id]
            candidate_row = candidate_trial.observations[query_id]
            for metric, direction in directions.items():
                baseline_value = getattr(baseline_row, metric)
                candidate_value = getattr(candidate_row, metric)
                if math.isclose(candidate_value, baseline_value, abs_tol=1e-12):
                    counts[metric][2] += 1
                elif (candidate_value > baseline_value) == (direction == "higher"):
                    counts[metric][0] += 1
                else:
                    counts[metric][1] += 1
    comparisons = len(baseline) * len(baseline[0].observations)
    return [
        PairedMetricSummary(
            metric=metric,
            preferred_direction=direction,
            comparisons=comparisons,
            candidate_wins=counts[metric][0],
            baseline_wins=counts[metric][1],
            ties=counts[metric][2],
        )
        for metric, direction in directions.items()
    ]


def _distribution(values: list[float]) -> MetricDistribution:
    return MetricDistribution(
        trials=len(values),
        mean=round(mean(values), 6),
        sample_stddev=round(stdev(values), 6),
        minimum=round(min(values), 6),
        maximum=round(max(values), 6),
    )


def _source_ref(path: Path, repository_root: Path) -> SourceArtifact:
    return SourceArtifact(
        path=path.resolve().relative_to(repository_root.resolve()).as_posix(),
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def _resolve_runs_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("repeat source paths must be repository-relative")
    resolved = (repository_root / path).resolve()
    _require_under_runs(resolved, repository_root)
    if not resolved.is_file():
        raise ValueError(f"repeat source artifact is missing: {value}")
    return resolved


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("repeat experiment artifacts must remain under ignored runs/")


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate repository root")


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not safe or safe in {".", ".."}:
        raise ValueError("query ID cannot be represented safely")
    return safe


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
