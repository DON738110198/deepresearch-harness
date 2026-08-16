from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.visible_pivot_bridge import (
    collect_frozen_pivot_cases,
    load_visible_pivot_registration,
    run_visible_pivot_oracle,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _analyze(value: str) -> tuple[str, ...]:
    return tuple(value.casefold().replace(".", "").split())


def test_visible_pivot_oracle_rescues_with_saved_snippet_term(tmp_path: Path) -> None:
    registration_path = _fixture(tmp_path)
    registration = load_visible_pivot_registration(registration_path)
    assert registration.budgets.maximum_provider_calls == 0
    frozen = collect_frozen_pivot_cases(registration_path)
    assert [item.query_id for item in frozen] == ["q1", "q2"]

    rankings = {
        "base one": ("visible-1", "other-1"),
        "base two": ("visible-2", "other-2"),
        "base one pivot": ("gold-1", "visible-1"),
    }
    result = run_visible_pivot_oracle(
        registration_path=registration_path,
        output_path=tmp_path / "runs" / "result" / "audit.json",
        analyze=_analyze,
        load_document=lambda docid: {
            "gold-1": "pivot answer-one",
            "gold-2": "answer-two unrelated",
        }[docid],
        document_frequency=lambda term: {"pivot": 3}.get(term, 20_000),
        search=lambda query, limit: rankings.get(query, ("other-1", "other-2"))[:limit],
    )

    assert result.decision == "visible_pivot_sufficient"
    assert result.visible_pivot_gold_hit_cases_at20 == 1
    assert result.baseline_gold_hit_cases_at20 == 0
    assert result.items[0].selected_pivot_term == "pivot"
    assert result.items[1].candidates == ()
    assert result.provider_calls == result.online_search_calls == result.judge_calls == 0
    assert result.offline_bm25_queries <= 6


def test_visible_pivot_excludes_answer_and_query_terms(tmp_path: Path) -> None:
    registration_path = _fixture(tmp_path)
    result = run_visible_pivot_oracle(
        registration_path=registration_path,
        output_path=tmp_path / "runs" / "result" / "audit.json",
        analyze=_analyze,
        load_document=lambda docid: {
            "gold-1": "answer-one base",
            "gold-2": "answer-two base",
        }[docid],
        document_frequency=lambda _term: 3,
        search=lambda query, limit: {
            "base one": ("visible-1", "other-1"),
            "base two": ("visible-2", "other-2"),
        }.get(query, ("other",))[:limit],
    )

    assert result.visible_pivot_gold_hit_cases_at20 == 0
    assert all(not item.candidates for item in result.items)
    assert result.decision == "freeze_visible_pivot_branch"


def test_visible_pivot_rejects_changed_baseline_ranking(tmp_path: Path) -> None:
    registration_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="BM25 baseline ranking changed"):
        run_visible_pivot_oracle(
            registration_path=registration_path,
            output_path=tmp_path / "runs" / "result" / "audit.json",
            analyze=_analyze,
            load_document=lambda docid: "pivot",
            document_frequency=lambda _term: 3,
            search=lambda _query, _limit: ("changed",),
        )


def _fixture(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    benchmark = tmp_path / "benchmarks" / "browsecomp_plus_v0"
    runs = tmp_path / "runs"
    runs.mkdir()

    dense_path = runs / "prior" / "dense.json"
    _write_json(
        dense_path,
        {
            "decision": "freeze_dense_channel",
            "dense_gold_hit_cases_at20": 0,
            "dense_gold_hit_cases_at100": 1,
            "items": [{"query_id": "q1"}, {"query_id": "q2"}],
        },
    )
    passage_path = runs / "prior" / "passage.json"
    _write_json(
        passage_path,
        {
            "decision": "freeze_passage_index_branch",
            "full_document_gold_hit_cases_at20": 0,
            "items": [
                {
                    "query_id": "q1",
                    "gold_docids": ["gold-1"],
                    "queries": [
                        {
                            "query": "base one",
                            "query_sha256": _text_hash("base one"),
                            "full_document_top20": ["visible-1", "other-1"],
                        }
                    ],
                },
                {
                    "query_id": "q2",
                    "gold_docids": ["gold-2"],
                    "queries": [
                        {
                            "query": "base two",
                            "query_sha256": _text_hash("base two"),
                            "full_document_top20": ["visible-2", "other-2"],
                        }
                    ],
                },
            ],
        },
    )
    gold_path = runs / "prior" / "gold.json"
    _write_json(
        gold_path,
        {
            "rows": [
                {
                    "query_id": "q1",
                    "question": "question one",
                    "answer": "answer-one",
                    "gold_docids": ["gold-1"],
                },
                {
                    "query_id": "q2",
                    "question": "question two",
                    "answer": "answer-two",
                    "gold_docids": ["gold-2"],
                },
            ]
        },
    )
    baseline = runs / "baseline"
    _write_json(
        baseline / "q1" / "run.json",
        {
            "query_id": "q1",
            "search_calls": [
                {
                    "outcome": "ok",
                    "query": "base one",
                    "results": [
                        {"docid": "visible-1", "snippet": "pivot base answer-one"}
                    ],
                }
            ],
        },
    )
    _write_json(
        baseline / "q2" / "run.json",
        {
            "query_id": "q2",
            "search_calls": [
                {
                    "outcome": "ok",
                    "query": "base two",
                    "results": [
                        {"docid": "visible-2", "snippet": "base answer-two"}
                    ],
                }
            ],
        },
    )
    index = runs / "external" / "bm25"
    index.mkdir(parents=True)
    segment = index / "segments_1"
    segment.write_bytes(b"fixture-index")

    registration_path = benchmark / "registration.json"
    _write_json(
        registration_path,
        {
            "schema_version": "visible-pivot-bridge-registration-v0",
            "status": "posthoc_registered_failure_cluster",
            "registered_at": "2026-08-16T00:00:00+00:00",
            "purpose": "fixture visible pivot",
            "dense_rank_result": {
                "path": dense_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(dense_path),
            },
            "passage_index_result": {
                "path": passage_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(passage_path),
            },
            "gold_slice": {
                "path": gold_path.relative_to(tmp_path).as_posix(),
                "sha256": _hash(gold_path),
            },
            "baseline_run_root": baseline.relative_to(tmp_path).as_posix(),
            "query_ids": ["q1", "q2"],
            "document_index_path": index.relative_to(tmp_path).as_posix(),
            "document_count": 2,
            "document_index_files": [
                {
                    "name": segment.name,
                    "bytes": segment.stat().st_size,
                    "sha256": _hash(segment),
                }
            ],
            "visible_evidence_source": "trial1_saved_successful_search_result_snippets",
            "pivot_policy": {
                "analyzer_id": "pyserini_default_english_v1.2.0",
                "minimum_term_characters": 3,
                "maximum_document_frequency": 10000,
                "maximum_candidate_terms_per_case": 2,
                "candidate_order": "document_frequency_ascending_then_term",
                "require_term_in_visible_non_gold_snippet": True,
                "require_term_in_gold_document": True,
                "excluded_vocabularies": [
                    "raw_question",
                    "recorded_generated_queries",
                    "gold_answer",
                ],
            },
            "retrieval": {
                "base_query_source": "trial1_recorded_successful_search_calls",
                "composition": "append_single_analyzed_pivot_token_v0",
                "analyzer_id": "pyserini_default_english_v1.2.0",
                "bm25_k1": 0.9,
                "bm25_b": 0.4,
                "top_k": 20,
                "stop_after_first_case_rescue": True,
            },
            "acceptance": {
                "minimum_visible_pivot_gold_doc_recall_at20_cases": 1,
                "required_baseline_gold_doc_recall_at20_cases": 0,
            },
            "budgets": {
                "maximum_offline_bm25_queries": 6,
                "maximum_provider_calls": 0,
                "maximum_online_search_calls": 0,
                "maximum_judge_calls": 0,
            },
            "sealed_holdout_access": "forbidden",
            "claim_boundary": "fixture diagnostic only",
        },
    )
    return registration_path
