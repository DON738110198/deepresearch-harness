from __future__ import annotations

import pytest

from deepresearch_harness.evidence_preview import (
    build_query_aware_lead_preview,
    format_query_aware_dense_lead,
)


def test_query_preview_compacts_metadata_and_selects_matching_passage() -> None:
    contents = """---
title: Royals 10-0 Giants
date: 2014-10-28
---
Unrelated introductory material about ticket sales.

WORLD SERIES - GAME 6. The win tied the series 3-3 and forced Game 7.
"""

    preview = build_query_aware_lead_preview(
        contents,
        "World Series Game 6 win forced Game 7 Tuesday",
    )

    assert preview.title == "Royals 10-0 Giants"
    assert preview.date == "2014-10-28"
    assert "forced Game 7" in preview.passage
    assert {"world", "series", "game", "forced"}.issubset(
        preview.matched_query_terms
    )
    rendered = format_query_aware_dense_lead("88560", preview)
    assert "docid=88560" in rendered
    assert "Title: Royals 10-0 Giants" in rendered
    assert "Date: 2014-10-28" in rendered


def test_query_preview_falls_back_to_first_meaningful_passage() -> None:
    preview = build_query_aware_lead_preview(
        "---\ntitle: Fixture\n---\nA meaningful opening passage without overlap.",
        "orthogonal terms",
    )

    assert preview.passage == "A meaningful opening passage without overlap."
    assert preview.matched_query_terms == ()


def test_query_preview_rejects_blank_inputs() -> None:
    with pytest.raises(ValueError, match="contents"):
        build_query_aware_lead_preview(" ", "query")
    with pytest.raises(ValueError, match="query"):
        build_query_aware_lead_preview("contents", " ")
