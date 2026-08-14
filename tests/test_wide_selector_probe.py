from __future__ import annotations

from deepresearch_harness.wide_selector_probe import (
    SelectorSpec,
    apply_selector,
    weighted_reciprocal_rank_fusion,
)


def test_weighted_rrf_is_deterministic_and_rewards_agreement() -> None:
    assert weighted_reciprocal_rank_fusion(
        ["a", "shared", "b"],
        ["c", "shared", "d"],
        bm25_weight=1.0,
        dense_weight=1.0,
        k=60,
        depth=3,
    ) == ["shared", "a", "c"]


def test_dense_rank_portfolio_uses_frozen_one_based_ranks() -> None:
    selector = SelectorSpec(
        selector_id="dense-geometric-v0",
        kind="fixed_dense_rank_portfolio",
        dense_ranks=[1, 2, 4, 8, 16],
    )
    dense = [f"d-{index}" for index in range(1, 21)]

    assert apply_selector(
        selector,
        bm25_docids=[],
        dense_docids=dense,
        output_depth=5,
    ) == ["d-1", "d-2", "d-4", "d-8", "d-16"]
