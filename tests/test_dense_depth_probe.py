from __future__ import annotations

import pytest

from deepresearch_harness.dense_depth_probe import (
    DenseDepthProbeRegistration,
    DepthProbeDecisionRule,
    choose_depth_probe_decision,
)


def _rule() -> DepthProbeDecisionRule:
    return DepthProbeDecisionRule(
        top5_reproduction_tolerance_pp=0.000001,
        minimum_depth20_delta_pp=10.0,
        minimum_depth50_delta_pp=10.0,
        if_depth20_passes=(
            "select_candidate_generation_plus_reranking_as_next_layer"
        ),
        if_only_depth50_passes=(
            "test_depth_efficiency_and_reranking_before_live_generation"
        ),
        if_depth50_fails=(
            "reject_depth_expansion_and_diagnose_query_or_corpus_mismatch"
        ),
    )


@pytest.mark.parametrize(
    ("depth20", "depth50", "expected"),
    [
        (10.0, 10.0, "select_candidate_generation_plus_reranking_as_next_layer"),
        (9.9, 10.0, "test_depth_efficiency_and_reranking_before_live_generation"),
        (9.9, 9.9, "reject_depth_expansion_and_diagnose_query_or_corpus_mismatch"),
    ],
)
def test_depth_probe_decision_is_bound_to_registered_thresholds(
    depth20: float, depth50: float, expected: str
) -> None:
    assert (
        choose_depth_probe_decision(
            depth20_delta_pp=depth20,
            depth50_delta_pp=depth50,
            rule=_rule(),
        )
        == expected
    )


def test_depth_probe_registration_rejects_post_hoc_depth_order() -> None:
    payload = {
        "schema_version": "browsecomp-plus-dense-depth-probe-v0",
        "status": "preregistered_not_run",
        "registered_at": "2026-08-14T00:00:00Z",
        "target_manifest_sha256": "a" * 64,
        "problem": {
            "decision_path": "runs/decision.json",
            "decision_sha256": "b" * 64,
            "retrieval_pool_probe_path": "runs/probe.json",
            "retrieval_pool_probe_sha256": "c" * 64,
            "live_dense_evidence_recall_delta_pp": 3.0,
            "frozen_query_dense_top5_delta_pp": 4.0,
            "frozen_query_rrf_top5_delta_pp": 3.0,
            "bm25_dense_union_pool_delta_pp": 7.0,
            "diagnosis": "fixture",
        },
        "hypothesis": "fixture",
        "fixed_contract": {
            "provider_calls": 0,
            "source_queries": "frozen_agent_search_calls",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "model_revision": "d" * 40,
            "retriever_manifest": "benchmarks/retrievers.json",
            "retriever_manifest_sha256": "e" * 64,
            "depths": [5, 20, 10, 50],
            "baseline": "stored_bm25_top5",
            "trial_count": 1,
            "queries_per_trial": 1,
            "sealed_holdout_access": "forbidden",
        },
        "sources": [
            {
                "trial_id": "trial-01",
                "summary_path": "runs/summary.json",
                "summary_sha256": "f" * 64,
                "gold_path": "runs/gold.json",
                "gold_sha256": "1" * 64,
                "top5_replay_path": "runs/replay.json",
                "top5_replay_sha256": "2" * 64,
            }
        ],
        "decision_rule": _rule().model_dump(),
        "claim_boundary": "fixture",
    }

    with pytest.raises(ValueError, match="unique and increasing"):
        DenseDepthProbeRegistration.model_validate(payload)
