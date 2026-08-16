from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.evidence_span_oracle import (
    answer_coverage,
    derive_answer_obligation_query,
    derive_answer_obligation_query_v1,
    load_evidence_span_oracle_registration,
    run_evidence_span_oracle,
    select_answer_obligation_span,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_obligation_query_uses_the_final_interrogative() -> None:
    question = (
        "A long clue describes a paper. The author worked elsewhere. "
        "What grant code appears in the Acknowledgments section?"
    )
    assert derive_answer_obligation_query(question) == (
        "What grant code appears in the Acknowledgments section"
    )


def test_v1_obligation_query_recovers_an_early_declarative_request() -> None:
    question = (
        "I am seeking the name of a project mentioned in a thesis. "
        "The author later earned a PhD. "
        "The author also presented at a symposium."
    )
    assert derive_answer_obligation_query(question) == (
        "The author also presented at a symposium."
    )
    assert derive_answer_obligation_query_v1(question) == (
        "I am seeking the name of a project mentioned in a thesis."
    )


def test_v1_span_records_the_selected_compiler() -> None:
    contents = (
        "The symposium was held in Bergen. " * 80
        + "The thesis describes the Aurora project in the methods chapter."
    )
    question = (
        "I am seeking the name of a project mentioned in a thesis. "
        "The author also presented at a symposium."
    )
    selected = select_answer_obligation_span(
        contents,
        question,
        maximum_span_characters=400,
        selector_id="answer_obligation_window_v1",
    )
    assert selected.selector_id == "answer_obligation_window_v1"
    assert selected.obligation_query == (
        "I am seeking the name of a project mentioned in a thesis."
    )
    assert "Aurora project" in selected.content


def test_v2_prioritises_an_explicit_section_anchor() -> None:
    contents = (
        "Background material. " * 80
        + "ACKNOWLEDGMENT This work was funded as part of ASCENT Project 10. "
        + "References. " * 80
        + "The Purdue University master thesis project team presented at a symposium. "
        * 20
    )
    question = (
        "I am seeking the name of a project that was mentioned in the "
        "acknowledgments section of a master's thesis submitted to Purdue University. "
        "The author also presented at a symposium."
    )
    selected = select_answer_obligation_span(
        contents,
        question,
        maximum_span_characters=400,
        selector_id="answer_obligation_window_v2",
    )
    assert selected.selector_id == "answer_obligation_window_v2"
    assert "ASCENT Project 10" in selected.content
    assert "acknowledgment" in selected.matched_section_anchors
    assert selected.score > 0


def test_span_selector_reaches_answer_far_beyond_the_head() -> None:
    contents = (
        "Introduction and unrelated biography. " * 150
        + "ACKNOWLEDGMENTS This work used grant HG00981-01A1 from Public Health Service. "
        + "References and unrelated material. " * 20
    )
    question = "What grant code appears in the Acknowledgments from Public Health Service?"
    selected = select_answer_obligation_span(
        contents, question, maximum_span_characters=400
    )
    assert selected.start_character > 2_000
    assert "HG00981-01A1" in selected.content
    assert "acknowledgments" in selected.matched_obligation_terms


def test_answer_coverage_handles_multi_document_name_lists() -> None:
    coverage = answer_coverage(
        "Turbett, Sweet, Carriero, Wolfe",
        "Turbett and Sweet appear here. Carriero and Wolfe appear elsewhere.",
    )
    assert coverage.all_atoms_present
    assert coverage.coverage == 1.0


def _write_fixture_registration(root: Path) -> Path:
    benchmark = root / "benchmarks" / "probe"
    run_root = root / "runs" / "candidate" / "q1"
    judge_root = root / "runs" / "judge" / "evaluation"
    index = root / "runs" / "index"
    benchmark.mkdir(parents=True)
    run_root.mkdir(parents=True)
    (judge_root / "results").mkdir(parents=True)
    index.mkdir(parents=True)
    diagnostic = root / "runs" / "diagnostic.json"
    judge = judge_root / "summary.json"
    gold = root / "runs" / "gold.json"
    frozen = benchmark / "frozen.txt"
    diagnostic.write_text(
        json.dumps({"rows": [{"query_id": "q1", "gold_recall": 1.0}]}),
        encoding="utf-8",
    )
    judge.write_text(
        json.dumps({"observations": [{"query_id": "q1", "correct": False}]}),
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps({"rows": [{"query_id": "q1", "gold_docids": ["doc-1"]}]}),
        encoding="utf-8",
    )
    frozen.write_text("frozen", encoding="utf-8")
    (judge_root / "results" / "q1_eval.json").write_text(
        json.dumps({"correct_answer": "CODE-123"}), encoding="utf-8"
    )
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Question: Which grant code is in Acknowledgments?\n\n"
                                    "Your response should be cited."
                                ),
                            }
                        ],
                    }
                ],
                "search_calls": [{"results": [{"docid": "doc-1"}]}],
            }
        ),
        encoding="utf-8",
    )
    registration = benchmark / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "evidence-span-availability-oracle-registration-v0",
                "status": "posthoc_registered_after_single_case_spotcheck",
                "registered_at": "2026-08-15T00:00:00+00:00",
                "purpose": "test",
                "candidate_run_root": "runs/candidate",
                "diagnostic": {"path": "runs/diagnostic.json", "sha256": _hash(diagnostic)},
                "judge_summary": {"path": "runs/judge/evaluation/summary.json", "sha256": _hash(judge)},
                "gold_slice": {"path": "runs/gold.json", "sha256": _hash(gold)},
                "query_ids": ["q1"],
                "previously_spotchecked_query_ids": [],
                "document_index_path": "runs/index",
                "selector_id": "answer_obligation_window_v0",
                "maximum_span_characters": 400,
                "frozen_artifacts": [{"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}],
                "acceptance": {
                    "minimum_selected_span_hit_cases": 1,
                    "minimum_uninspected_selected_span_hit_cases": 1,
                    "minimum_selected_over_head_delta": 1,
                    "missing_documents_must_equal": 0,
                },
                "sealed_holdout_access": "forbidden",
                "claim_boundary": "posthoc test only",
            }
        ),
        encoding="utf-8",
    )
    return registration


def test_oracle_is_zero_provider_and_persists_a_gated_result(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    registration = _write_fixture_registration(root)
    document = "head " * 200 + "Acknowledgments grant code CODE-123"
    result = run_evidence_span_oracle(
        registration_path=registration,
        output_path=root / "runs" / "oracle" / "result.json",
        document_loader=lambda _docid: document,
    )
    assert result.decision == "pass"
    assert result.provider_calls == 0
    assert result.search_calls == 0
    assert result.head_span_hit_cases == 0
    assert result.selected_span_hit_cases == 1


def test_registration_rejects_changed_source(tmp_path: Path) -> None:
    registration = _write_fixture_registration(tmp_path / "repo")
    diagnostic = tmp_path / "repo" / "runs" / "diagnostic.json"
    diagnostic.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        load_evidence_span_oracle_registration(registration)
