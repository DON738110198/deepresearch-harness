from deepresearch_harness.progressive_disclosure_calibration import (
    _nearest_rank_percentile,
    _round_up,
)


def test_calibration_budget_helpers_are_deterministic() -> None:
    assert _nearest_rank_percentile([10, 30, 20, 40], 0.5) == 20
    assert _nearest_rank_percentile([10, 30, 20, 40], 0.9) == 40
    assert _round_up(513, 512) == 1024
    assert _round_up(1024, 512) == 1024
