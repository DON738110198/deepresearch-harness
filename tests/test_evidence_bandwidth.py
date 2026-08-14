from __future__ import annotations

import pytest

from deepresearch_harness.evidence_bandwidth import allocate_waterfill_caps


def test_waterfill_gives_every_result_a_minimum_then_shares_remainder() -> None:
    assert allocate_waterfill_caps(
        [3, 10, 10], total_budget=12, minimum_per_result=2
    ) == [3, 5, 4]


def test_waterfill_stops_when_documents_are_exhausted() -> None:
    assert allocate_waterfill_caps(
        [1, 2], total_budget=20, minimum_per_result=3
    ) == [1, 2]


def test_waterfill_rejects_an_underfunded_budget() -> None:
    with pytest.raises(ValueError, match="cannot fund"):
        allocate_waterfill_caps([10, 10], total_budget=5, minimum_per_result=3)
