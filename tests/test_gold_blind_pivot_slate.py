from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.gold_blind_pivot_slate import (
    build_gold_blind_pivot_slate,
    extract_visible_body,
    score_gold_blind_pivot_slate,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _analyze(value: str) -> tuple[str, ...]:
    stop = {"the", "this", "question", "base", "one", "two"}
    return tuple(
        token.casefold().strip(".,:;!?")
        for token in value.split()
        if token.casefold().strip(".,:;!?") not in stop
    )


def test_body_extraction_removes_wrapper_and_frontmatter() -> None:
    body, wrapper, frontmatter = extract_visible_body(
        "[BM25 anchor]\n---\ntitle: Inline\ncoordinates: inline\n---\nArgentina body"
    )
    assert body == "Argentina body"
    assert wrapper is True
    assert frontmatter is True


def test_gold_blind_slate_is_persisted_before_scoring_and_passes(tmp_path: Path) -> None:
    registration_path = _fixture(tmp_path)
    output_dir = tmp_path / "runs" / "result"
    slate_path = output_dir / "slate.json"
    slate = build_gold_blind_pivot_slate(
        registration_path=registration_path,
        output_path=slate_path,
        analyze=_analyze,
        document_frequency=lambda term: {"argentina": 10, "naep": 5}.get(term, 20_000),
    )

    assert slate.status == "selected_without_gold"
    assert slate.selection_failures == 0
    assert [item.selected[0].analyzed_term for item in slate.items] == [
        "argentina",
        "naep",
    ]
    serialized = slate_path.read_text(encoding="utf-8")
    assert "secret-answer" not in serialized
    assert "gold-1" not in serialized
    assert "coordinates" not in serialized

    result = score_gold_blind_pivot_slate(
        registration_path=registration_path,
        slate_path=slate_path,
        output_path=output_dir / "audit.json",
        analyze=_analyze,
        search=lambda query, limit: {
            "base one Argentina": ("gold-1", "other"),
            "base two NAEP": ("gold-2", "other"),
        }.get(query, ("other",))[:limit],
    )

    assert result.decision == "gold_blind_pivot_candidate"
    assert result.selector_gold_hit_cases_at20 == 2
    assert result.retained_oracle_rescue_cases == 2
    assert result.offline_bm25_queries == 2
    assert result.answer_string_leak_cases == result.gold_docid_leak_cases == 0
    assert result.provider_calls == result.online_search_calls == result.judge_calls == 0


def test_slate_builder_does_not_open_gold_or_oracle_artifacts(tmp_path: Path) -> None:
    registration_path = _fixture(tmp_path)
    (tmp_path / "runs" / "prior" / "gold.json").unlink()
    (tmp_path / "runs" / "prior" / "oracle.json").unlink()

    slate = build_gold_blind_pivot_slate(
        registration_path=registration_path,
        output_path=tmp_path / "runs" / "result" / "slate.json",
        analyze=_analyze,
        document_frequency=lambda term: {"argentina": 10, "naep": 5}.get(term, 20_000),
    )

    assert slate.status == "selected_without_gold"
    assert slate.selected_pivot_count == 2


def _fixture(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    benchmark = tmp_path / "benchmarks" / "browsecomp_plus_v0"
    runs = tmp_path / "runs"
    runs.mkdir()
    baseline = runs / "baseline"
    for query_id, question, query, docid, body in (
        (
            "q1",
            "question one",
            "base one",
            "visible-1",
            "[BM25 anchor]\n---\ntitle: Inline\ncoordinates: inline\n---\nArgentina connects clue.",
        ),
        (
            "q2",
            "question two",
            "base two",
            "visible-2",
            "[Dense lead]\nNAEP connects education.",
        ),
    ):
        _write_json(
            baseline / query_id / "request.json",
            {"query_id": query_id, "question": question},
        )
        _write_json(
            baseline / query_id / "run.json",
            {
                "query_id": query_id,
                "search_calls": [
                    {
                        "outcome": "ok",
                        "query": query,
                        "results": [{"docid": docid, "snippet": body}],
                    }
                ],
            },
        )
    gold_path = runs / "prior" / "gold.json"
    _write_json(
        gold_path,
        {
            "rows": [
                {"query_id": "q1", "answer": "secret-answer-one", "gold_docids": ["gold-1"]},
                {"query_id": "q2", "answer": "secret-answer-two", "gold_docids": ["gold-2"]},
            ]
        },
    )
    index = runs / "external" / "bm25"
    index.mkdir(parents=True)
    segment = index / "segments_1"
    segment.write_bytes(b"fixture-index")
    source_path = benchmark / "source.json"
    _write_json(
        source_path,
        {
            "query_ids": ["q1", "q2"],
            "baseline_run_root": baseline.relative_to(tmp_path).as_posix(),
            "gold_slice": {
                "path": gold_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(gold_path),
            },
            "document_index_path": index.relative_to(tmp_path).as_posix(),
            "document_count": 2,
            "document_index_files": [
                {
                    "name": segment.name,
                    "bytes": segment.stat().st_size,
                    "sha256": _hash(segment),
                }
            ],
        },
    )
    oracle_path = runs / "prior" / "oracle.json"
    _write_json(
        oracle_path,
        {
            "items": [
                {"query_id": "q1", "rescued": True},
                {"query_id": "q2", "rescued": True},
            ]
        },
    )
    registration_path = benchmark / "registration.json"
    _write_json(
        registration_path,
        {
            "schema_version": "gold-blind-visible-pivot-slate-registration-v0",
            "status": "posthoc_registered_after_oracle",
            "registered_at": "2026-08-16T00:00:00+00:00",
            "purpose": "fixture gold-blind selector",
            "source_registration": {
                "path": source_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(source_path),
            },
            "oracle_result": {
                "path": oracle_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(oracle_path),
            },
            "baseline_run_root": baseline.relative_to(tmp_path).as_posix(),
            "query_ids": ["q1", "q2"],
            "selector_input": {
                "question_source": "saved_request_json",
                "query_and_evidence_source": "saved_successful_run_search_calls",
                "gold_documents_available_to_selector": False,
                "gold_answer_available_to_selector": False,
            },
            "body_extraction": {
                "strip_leading_bracket_wrapper": True,
                "strip_yaml_frontmatter": True,
                "frontmatter_only_candidates_must_equal": 0,
            },
            "candidate_policy": {
                "surface_pattern_id": "ascii_capitalized_token_v0",
                "analyzer_id": "pyserini_default_english_v1.2.0",
                "minimum_surface_characters": 3,
                "maximum_surface_characters": 40,
                "require_single_analyzed_term": True,
                "maximum_document_frequency": 10000,
                "exclude_question_and_recorded_query_terms": True,
                "candidate_order": "document_frequency_ascending_then_support_descending_then_first_seen",
                "slate_size_per_case": 1,
                "source_query_policy": "first_supporting_query",
            },
            "retrieval": {
                "composition": "append_surface_form_to_provenance_query_v0",
                "analyzer_id": "pyserini_default_english_v1.2.0",
                "bm25_k1": 0.9,
                "bm25_b": 0.4,
                "top_k": 20,
                "maximum_pivot_queries_per_case": 1,
            },
            "acceptance": {
                "minimum_selector_gold_doc_recall_at20_cases": 2,
                "minimum_retained_oracle_rescue_cases": 2,
                "required_selection_failures": 0,
                "required_frontmatter_only_selected_candidates": 0,
            },
            "budgets": {
                "maximum_offline_bm25_queries": 2,
                "maximum_provider_calls": 0,
                "maximum_online_search_calls": 0,
                "maximum_judge_calls": 0,
            },
            "sealed_holdout_access": "forbidden",
            "claim_boundary": "fixture calibration only",
        },
    )
    return registration_path
