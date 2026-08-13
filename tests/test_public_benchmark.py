from pathlib import Path

import pytest
from pydantic import ValidationError

from deepresearch_harness.pipeline import BenchmarkPlanDraft, BenchmarkResearchPipeline, LocalCorpusCollector
from deepresearch_harness.providers import FakeProvider
from deepresearch_harness.public_benchmark import (
    LiveDRBenchManifest,
    LiveDRBenchTask,
    exact_main_claim_metrics,
    official_shape_compatible,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_main_claim_metric_normalizes_entity_names() -> None:
    task = LiveDRBenchTask(
        key=4,
        category="entities",
        question="Find the matching people.",
        ground_truths=[["Alice Smith", "Bob Jones"]],
    )

    metrics = exact_main_claim_metrics(task, [["alice smith", "Charlie"]])

    assert metrics.matches == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_exact_main_claim_metric_ignores_non_scored_fields() -> None:
    task = LiveDRBenchTask(
        key=31,
        category="novel-datasets-identification",
        question="Identify the dataset.",
        ground_truths=[{"title": "Target Dataset", "year": 2024}],
        eval_info={"main_claims": ["title"]},
    )

    metrics = exact_main_claim_metrics(task, [{"title": "target dataset", "year": 1999}])

    assert metrics.f1 == 1.0
    assert metrics.compared_fields == ["title"]


def test_public_manifest_rejects_duplicate_task_keys() -> None:
    with pytest.raises(ValidationError, match="task_keys must be unique"):
        LiveDRBenchManifest(
            benchmark_id="test",
            status="frozen",
            dataset_id="microsoft/LiveDRBench",
            dataset_revision="a" * 40,
            task_keys=[4, 4],
            source_repository="https://github.com/microsoft/LiveDRBench",
            source_repository_commit="b" * 40,
            selection_rationale="Test selection.",
            evaluator="compatibility_exact_main_claim_v1",
            official_evaluator_status="planned_not_run",
        )


def test_benchmark_pipeline_persists_structured_answer_and_citations(tmp_path: Path) -> None:
    pipeline = BenchmarkResearchPipeline(
        provider=FakeProvider(),
        collector=LocalCorpusCollector(ROOT / "examples" / "offline_corpus.json"),
        output_dir=tmp_path,
    )

    state = pipeline.run('Return a JSON array of rollout controls.')
    report = Path(state.report_path).read_text(encoding="utf-8")

    assert state.structured_answer == []
    assert '```json\n[]\n```' in report
    assert all(citation.marker in report for citation in state.citations)
    assert "## Sources" in report


def test_benchmark_plan_supplies_a_default_audit_step() -> None:
    draft = BenchmarkPlanDraft.model_validate({"search_queries": ["focused query"]})

    assert draft.steps[0].id == "identify"
    assert draft.search_queries == ["focused query"]


def test_official_shape_check_rejects_missing_outer_alternative() -> None:
    task = LiveDRBenchTask(
        key=31,
        category="novel-datasets-identification",
        question="Identify the dataset.",
        ground_truths=[{"title": "Target Dataset"}],
    )

    assert not official_shape_compatible(task, [])
    assert official_shape_compatible(task, [{"title": "Candidate"}])
