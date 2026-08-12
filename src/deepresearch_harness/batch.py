from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel, Field

from .benchmark import BenchmarkScore, score_run, validate_suite_assets
from .contracts import HarnessConfig, RunState
from .experiment import ExperimentManifest, validate_experiment_manifest
from .pipeline import BaselineResearchPipeline, LocalCorpusCollector, SearchWritePipeline
from .providers import LLMProvider, provider_from_config


class BatchRunRecord(BaseModel):
    task_id: str
    variant: str
    status: str
    run_id: str | None = None
    state_path: str | None = None
    score_path: str | None = None
    error: str | None = None


class VariantAggregate(BaseModel):
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    mean_evidence_id_recall: float | None = None
    mean_evidence_id_precision: float | None = None
    mean_evidence_obligation_recall: float | None = None
    mean_citation_structural_integrity: float | None = None
    total_tokens: int = Field(ge=0)
    total_estimated_cost_usd: float = Field(ge=0)


class BatchSummary(BaseModel):
    experiment_id: str
    status: str
    started_at: datetime
    finished_at: datetime
    manifest_path: str
    output_dir: str
    records: list[BatchRunRecord]
    aggregates: dict[str, VariantAggregate]


def run_experiment_batch(
    *,
    manifest_path: Path,
    config: HarnessConfig,
    output_root: Path,
    provider: LLMProvider | None = None,
    enforce_provider_match: bool = True,
) -> BatchSummary:
    manifest = validate_experiment_manifest(manifest_path)
    if enforce_provider_match:
        _validate_provider_match(manifest, config)
    suite_path = (manifest_path.parent / manifest.suite_path).resolve()
    suite, _ = validate_suite_assets(suite_path)
    corpus_path = suite_path.parent / suite.corpus_path
    collector = LocalCorpusCollector(corpus_path)
    runtime_provider = provider or provider_from_config(config)

    started_at = datetime.now(timezone.utc)
    batch_dir = _create_batch_dir(output_root, manifest.experiment_id, started_at)
    (batch_dir / "manifest.snapshot.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    records: list[BatchRunRecord] = []
    scores: dict[str, list[BenchmarkScore]] = {variant: [] for variant in manifest.variants}
    usages: dict[str, list[tuple[int, float]]] = {variant: [] for variant in manifest.variants}

    pipeline_classes = {
        "b0_search_write": SearchWritePipeline,
        "b1_plan_search_ledger_write": BaselineResearchPipeline,
    }
    for variant in manifest.variants:
        for task in suite.tasks:
            task_output = batch_dir / variant / task.id
            pipeline = pipeline_classes[variant](
                provider=runtime_provider,
                collector=collector,
                output_dir=task_output,
                max_evidence=config.run.max_evidence,
                budget_limits=manifest.budget,
            )
            try:
                state = pipeline.run(task.question)
                score = score_run(task, state)
                run_dir = Path(state.report_path).parent
                score_path = run_dir / "score.auto.json"
                score_path.write_text(score.model_dump_json(indent=2), encoding="utf-8")
                scores[variant].append(score)
                total_tokens = state.total_usage.input_tokens + state.total_usage.output_tokens
                usages[variant].append((total_tokens, state.total_usage.estimated_cost_usd))
                records.append(
                    BatchRunRecord(
                        task_id=task.id,
                        variant=variant,
                        status="succeeded",
                        run_id=state.run_id,
                        state_path=str(run_dir / "state.json"),
                        score_path=str(score_path),
                    )
                )
            except Exception as error:
                state_paths = sorted(task_output.glob("*/state.json"), key=lambda path: path.stat().st_mtime)
                failed_state = RunState.model_validate_json(state_paths[-1].read_text(encoding="utf-8")) if state_paths else None
                if failed_state is not None:
                    total_tokens = failed_state.total_usage.input_tokens + failed_state.total_usage.output_tokens
                    usages[variant].append((total_tokens, failed_state.total_usage.estimated_cost_usd))
                records.append(
                    BatchRunRecord(
                        task_id=task.id,
                        variant=variant,
                        status="failed",
                        run_id=failed_state.run_id if failed_state else None,
                        state_path=str(state_paths[-1]) if state_paths else None,
                        error=str(error),
                    )
                )

    aggregates = {
        variant: _aggregate_variant(
            variant_scores=scores[variant],
            variant_usages=usages[variant],
            failed=sum(record.variant == variant and record.status == "failed" for record in records),
        )
        for variant in manifest.variants
    }
    finished_at = datetime.now(timezone.utc)
    summary = BatchSummary(
        experiment_id=manifest.experiment_id,
        status="succeeded" if all(record.status == "succeeded" for record in records) else "partial",
        started_at=started_at,
        finished_at=finished_at,
        manifest_path=str(manifest_path.resolve()),
        output_dir=str(batch_dir),
        records=records,
        aggregates=aggregates,
    )
    (batch_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def _validate_provider_match(manifest: ExperimentManifest, config: HarnessConfig) -> None:
    actual = config.provider
    expected = manifest.provider
    mismatches: list[str] = []
    if actual.kind != expected.kind:
        mismatches.append("kind")
    if actual.model != expected.model:
        mismatches.append("model")
    if (actual.base_url or "").rstrip("/") != str(expected.base_url).rstrip("/"):
        mismatches.append("base_url")
    if actual.thinking_mode != expected.thinking_mode:
        mismatches.append("thinking_mode")
    if actual.pricing != expected.pricing:
        mismatches.append("pricing")
    if mismatches:
        raise ValueError(f"runtime provider does not match frozen manifest fields: {', '.join(mismatches)}")


def _create_batch_dir(root: Path, experiment_id: str, started_at: datetime) -> Path:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    batch_dir = root / experiment_id / timestamp
    suffix = 1
    while batch_dir.exists():
        batch_dir = root / experiment_id / f"{timestamp}-{suffix}"
        suffix += 1
    batch_dir.mkdir(parents=True)
    return batch_dir


def _aggregate_variant(
    *,
    variant_scores: list[BenchmarkScore],
    variant_usages: list[tuple[int, float]],
    failed: int,
) -> VariantAggregate:
    if not variant_scores:
        return VariantAggregate(completed=0, failed=failed, total_tokens=0, total_estimated_cost_usd=0.0)
    return VariantAggregate(
        completed=len(variant_scores),
        failed=failed,
        mean_evidence_id_recall=fmean(score.evidence_id_recall for score in variant_scores),
        mean_evidence_id_precision=fmean(score.evidence_id_precision for score in variant_scores),
        mean_evidence_obligation_recall=fmean(score.evidence_obligation_recall for score in variant_scores),
        mean_citation_structural_integrity=fmean(score.citation_structural_integrity for score in variant_scores),
        total_tokens=sum(tokens for tokens, _ in variant_usages),
        total_estimated_cost_usd=sum(cost for _, cost in variant_usages),
    )
