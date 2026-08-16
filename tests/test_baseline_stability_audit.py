from __future__ import annotations

from deepresearch_harness.baseline_stability_audit import classify_baseline_stability


def test_stability_classifier_separates_retrieval_and_preview_failures() -> None:
    assert (
        classify_baseline_stability(
            judge_correct_trials=0,
            gold_doc_retrieved_trials=0,
            gold_answer_span_trials=0,
        )
        == "persistent_retrieval_miss"
    )
    assert (
        classify_baseline_stability(
            judge_correct_trials=0,
            gold_doc_retrieved_trials=3,
            gold_answer_span_trials=0,
        )
        == "gold_doc_present_span_missing"
    )


def test_stability_classifier_preserves_answer_instability() -> None:
    assert (
        classify_baseline_stability(
            judge_correct_trials=2,
            gold_doc_retrieved_trials=3,
            gold_answer_span_trials=3,
        )
        == "unstable_answer"
    )
