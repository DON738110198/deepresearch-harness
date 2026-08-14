from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearch_harness.evidence_preview_calibration import (
    EvidencePreviewCalibration,
)


def _payload() -> dict[str, object]:
    return {
        "created_at": "2026-08-15T00:00:00+00:00",
        "status": "calibration_passed",
        "source_summary_sha256": "0" * 64,
        "gold_slice_sha256": "1" * 64,
        "source_query_count": 1,
        "dense_result_count": 10,
        "relevant_dense_hit_count": 1,
        "baseline_selectable_relevant_hits": 0,
        "candidate_selectable_relevant_hits": 1,
        "candidate_maximum_search_ingress_tokens": 28_160,
        "rows": [
            {
                "query_id": "q1",
                "search_index": 1,
                "docid": "d1",
                "baseline_visible_sha256": "2" * 64,
                "candidate_visible_sha256": "3" * 64,
                "baseline_visible_title": False,
                "baseline_visible_date": False,
                "baseline_query_term_matches": 0,
                "baseline_selectable": False,
                "candidate_visible_title": True,
                "candidate_visible_date": True,
                "candidate_query_term_matches": 2,
                "candidate_selectable": True,
            }
        ],
        "selected_policy": "query_window_v0_64",
        "claim_boundary": "Offline calibration only.",
    }


def test_preview_calibration_selects_only_a_budgeted_complete_candidate() -> None:
    calibration = EvidencePreviewCalibration.model_validate(_payload())

    assert calibration.status == "calibration_passed"
    assert calibration.selected_policy == "query_window_v0_64"


def test_preview_calibration_rejects_inconsistent_selection() -> None:
    payload = _payload()
    payload["candidate_selectable_relevant_hits"] = 0

    with pytest.raises(ValidationError, match="candidate selectable count"):
        EvidencePreviewCalibration.model_validate(payload)
