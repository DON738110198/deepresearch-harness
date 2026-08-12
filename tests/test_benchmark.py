from pathlib import Path

import pytest

from deepresearch_harness.benchmark import FailureFocus, HumanReportAnnotation, score_run, validate_suite_assets
from deepresearch_harness.contracts import Citation, Claim, Evidence, RunState, Task


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "benchmarks" / "pilot_v0" / "tasks.json"


def test_pilot_suite_has_ten_balanced_tasks_and_valid_evidence_references() -> None:
    suite, corpus = validate_suite_assets(SUITE_PATH)

    assert len(suite.tasks) == 10
    assert len(corpus) >= 30
    assert all(record.synthetic for record in corpus)
    counts = {focus: 0 for focus in FailureFocus}
    for task in suite.tasks:
        counts[task.failure_focus] += 1
    assert set(counts.values()) == {2}


def test_score_separates_deterministic_and_human_metrics() -> None:
    suite, corpus = validate_suite_assets(SUITE_PATH)
    task = suite.tasks[0]
    by_id = {record.id: record for record in corpus}
    selected = [by_id["t01-blast-radius"], by_id["t01-rollback"]]
    run = RunState(
        run_id="test-run",
        task=Task(id=task.id, question=task.question),
        evidence=[
            Evidence(id=item.id, title=item.title, url=item.url, excerpt=item.snippet, query="phased rollout")
            for item in selected
        ],
        claims=[Claim(id="claim-1", text="A staged release limits initial impact.", evidence_ids=["t01-blast-radius"])],
        citations=[Citation(id="citation-1", claim_id="claim-1", evidence_id="t01-blast-radius", marker="[1]")],
    )

    automatic = score_run(task, run)
    assert automatic.evidence_id_recall == pytest.approx(2 / 3)
    assert automatic.evidence_id_precision == 1.0
    assert automatic.evidence_obligation_recall == pytest.approx(2 / 3)
    assert automatic.citation_structural_integrity == 1.0
    assert automatic.obligation_coverage is None
    assert automatic.annotation_status == "not_annotated"

    annotation = HumanReportAnnotation(
        task_id=task.id,
        covered_obligation_ids=["blast-radius", "rollback"],
        supported_claim_ids=["claim-1"],
        citation_supported_claim_ids=["claim-1"],
    )
    annotated = score_run(task, run, annotation)
    assert annotated.obligation_coverage == pytest.approx(2 / 3)
    assert annotated.citation_support_rate == 1.0
    assert annotated.unsupported_claim_rate == 0.0
    assert annotated.irrelevant_claim_rate == 0.0
    assert annotated.conflict_handling_rate is None
    assert annotated.annotation_status == "human_annotated"


def test_annotation_rejects_unknown_obligation() -> None:
    suite, _ = validate_suite_assets(SUITE_PATH)
    task = suite.tasks[0]
    run = RunState(run_id="test-run", task=Task(id=task.id, question=task.question))
    annotation = HumanReportAnnotation(task_id=task.id, covered_obligation_ids=["unknown"])

    with pytest.raises(ValueError, match="unknown obligation"):
        score_run(task, run, annotation)


def test_claim_without_citation_does_not_receive_perfect_integrity() -> None:
    suite, _ = validate_suite_assets(SUITE_PATH)
    task = suite.tasks[0]
    run = RunState(
        run_id="test-run",
        task=Task(id=task.id, question=task.question),
        claims=[Claim(id="claim-1", text="Uncited conclusion.", evidence_ids=["missing-from-run"])],
    )

    score = score_run(task, run)
    assert score.citation_structural_integrity == 0.0
