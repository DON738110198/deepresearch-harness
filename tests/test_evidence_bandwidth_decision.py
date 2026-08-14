from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.evidence_bandwidth_decision import (
    EvidenceBandwidthGate,
    SynthesisLossDiagnostic,
    choose_next_action,
)


GATE_IDS = (
    "registration_status",
    "trial_count",
    "query_count",
    "candidate_schema_complete",
    "output_budget_overshoot",
    "evidence_recall_delta",
    "evidence_query_wins_minus_losses",
    "judge_accuracy_delta",
    "judge_parse_failures",
    "judge_request_failures",
    "candidate_to_baseline_search_call_ratio_min",
    "candidate_to_baseline_search_call_ratio_max",
    "candidate_to_baseline_total_token_ratio",
    "candidate_to_baseline_provider_cost_ratio",
    "combined_provider_cost",
    "final_provider_failures",
)


def _gates(*failed: str) -> list[EvidenceBandwidthGate]:
    return [
        EvidenceBandwidthGate(
            gate_id=gate_id,
            observed=0 if gate_id in failed else 1,
            operator="eq",
            threshold=1,
            passed=gate_id not in failed,
        )
        for gate_id in GATE_IDS
    ]


@pytest.mark.parametrize(
    ("failed", "expected"),
    [
        ((), "promote_evidence_bandwidth_and_cluster_remaining_failures"),
        (
            ("candidate_to_baseline_total_token_ratio",),
            "calibrate_adaptive_evidence_bandwidth_budget",
        ),
        (
            ("judge_accuracy_delta",),
            "diagnose_evidence_selection_and_synthesis_from_saved_runs",
        ),
        (
            ("evidence_recall_delta",),
            "diagnose_retrieval_bandwidth_from_saved_replays",
        ),
        (
            ("candidate_schema_complete",),
            "audit_incomplete_or_structurally_invalid_grid",
        ),
    ],
)
def test_next_action_prioritizes_the_earliest_failed_layer(
    failed: tuple[str, ...], expected: str
) -> None:
    assert choose_next_action(_gates(*failed)) == expected


def test_synthesis_loss_counts_must_be_nested() -> None:
    with pytest.raises(ValidationError):
        SynthesisLossDiagnostic(
            paired_comparisons=3,
            candidate_improvements=1,
            candidate_regressions=1,
            paired_ties=1,
            regressions_with_candidate_relevant_evidence=1,
            regressions_with_candidate_evidence_advantage=2,
            improvements_with_candidate_evidence_advantage=1,
            baseline_incorrect=1,
            baseline_format_failures=0,
            baseline_no_relevant_doc_retrieved=1,
            baseline_relevant_evidence_but_incorrect=0,
            candidate_incorrect=1,
            candidate_format_failures=0,
            candidate_no_relevant_doc_retrieved=0,
            candidate_relevant_evidence_but_incorrect=1,
        )
