from __future__ import annotations

import pytest

from deepresearch_harness.obligation_open_audit import classify_mechanism_effect


@pytest.mark.parametrize(
    ("outcome", "opens", "answer_bearing", "expected"),
    [
        (
            "improvement",
            1,
            True,
            "supported_gain_from_answer_bearing_gold_open",
        ),
        ("improvement", 0, False, "gain_without_answer_bearing_gold_open"),
        ("regression", 0, False, "regression_without_successful_open"),
        ("regression", 1, False, "regression_after_non_answer_bearing_open"),
        ("both_correct", 1, True, "stable_with_answer_bearing_gold_open"),
        ("both_wrong", 0, False, "stable_other"),
    ],
)
def test_classify_mechanism_effect(
    outcome: str,
    opens: int,
    answer_bearing: bool,
    expected: str,
) -> None:
    assert (
        classify_mechanism_effect(
            paired_outcome=outcome,
            successful_opens=opens,
            answer_bearing_gold_open=answer_bearing,
        )
        == expected
    )


def test_answer_bearing_open_requires_successful_tool_result() -> None:
    with pytest.raises(ValueError, match="requires a successful open"):
        classify_mechanism_effect(
            paired_outcome="improvement",
            successful_opens=0,
            answer_bearing_gold_open=True,
        )
