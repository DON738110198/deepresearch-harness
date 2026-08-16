from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.contracts import Usage
from deepresearch_harness.evidence_span_oracle import SelectedEvidenceSpan
from deepresearch_harness.post_run_overlay import (
    PreparedOverlayCase,
    PreparedOverlaySpan,
    RepairProposal,
    apply_literal_support_gate,
    build_overlay_prompt,
    run_post_run_overlay_calibration,
)
from deepresearch_harness.document_target_oracle import (
    ObligationTargetSlate,
    TargetDocument,
    TargetSlateSearchCall,
)
from deepresearch_harness.providers import Completion, LLMProvider


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _span(docid: str, content: str) -> PreparedOverlaySpan:
    return PreparedOverlaySpan(
        docid=docid,
        document_sha256=sha256(content.encode("utf-8")).hexdigest(),
        selection=SelectedEvidenceSpan(
            selector_id="answer_obligation_window_v2",
            obligation_query="project in acknowledgments",
            content=content,
            start_character=0,
            end_character=len(content),
            matched_section_anchors=("acknowledgment",),
            matched_obligation_terms=("project", "acknowledgments"),
            matched_question_terms=("project", "acknowledgments"),
            score=1.0,
        ),
    )


def test_literal_gate_applies_only_a_cited_verbatim_replacement() -> None:
    span = _span(
        "doc-1", "ACKNOWLEDGMENT This work supported the Aurora Project."
    )
    proposal = RepairProposal(
        action="replace",
        short_answer="Aurora Project",
        support_docids=("doc-1",),
        evidence_quote="This work supported the Aurora Project.",
    )
    decision = apply_literal_support_gate(
        baseline_answer="Exact Answer: Unknown",
        spans=(span,),
        proposal=proposal,
    )
    assert decision.applied
    assert decision.reason == "literal_supported_replacement"
    assert "Exact Answer: Aurora Project" in decision.candidate_answer
    assert "[doc-1]" in decision.candidate_answer


def test_literal_gate_keeps_baseline_when_quote_is_not_in_the_span() -> None:
    baseline = "Exact Answer: Unknown"
    proposal = RepairProposal(
        action="replace",
        short_answer="Aurora Project",
        support_docids=("doc-1",),
        evidence_quote="A fabricated quote containing Aurora Project.",
    )
    decision = apply_literal_support_gate(
        baseline_answer=baseline,
        spans=(_span("doc-1", "The Aurora Project is named here."),),
        proposal=proposal,
    )
    assert not decision.applied
    assert decision.reason == "quote_not_literal_in_cited_span"
    assert decision.candidate_answer == baseline


def test_evidence_only_prompt_changes_only_baseline_visibility() -> None:
    case = PreparedOverlayCase(
        query_id="q1",
        question="Which project is named?",
        baseline_answer="Exact Answer: Wrong",
        baseline_answer_sha256=sha256(b"Exact Answer: Wrong").hexdigest(),
        target_slate=ObligationTargetSlate(
            question="Which project is named?",
            obligation_query="Which project is named",
            selected_search_calls=(
                TargetSlateSearchCall(
                    search_call_index=1,
                    query="project named",
                    matched_obligation_terms=("project",),
                    obligation_term_coverage=0.5,
                ),
            ),
            targets=(
                TargetDocument(
                    docid="doc-1",
                    channel="bm25_anchor",
                    search_call_index=1,
                    result_rank=1,
                    search_query="project named",
                    matched_obligation_terms=("project",),
                    obligation_term_coverage=0.5,
                ),
            ),
        ),
        spans=(_span("doc-1", "The Aurora Project is named here."),),
    )
    visible = json.loads(build_overlay_prompt(case))
    blind = json.loads(build_overlay_prompt(case, prompt_variant="evidence_only_v1"))
    assert visible.pop("baseline_response") == case.baseline_answer
    assert "baseline_response" not in blind
    assert visible == blind


class _OverlayProvider(LLMProvider):
    name = "fixture"
    model = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        *,
        stage: str,
        prompt: str,
        json_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> Completion:
        self.calls += 1
        assert stage == "post_run_overlay"
        assert json_output
        assert max_output_tokens == 200
        text = json.dumps(
            {
                "action": "replace",
                "short_answer": "Aurora Project",
                "support_docids": ["answer-anchor"],
                "evidence_quote": "This work supported the Aurora Project.",
            }
        )
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=100,
                output_tokens=20,
                estimated_cost_usd=0.001,
            ),
            latency_ms=10,
        )


def _write_overlay_fixture(root: Path) -> Path:
    benchmark = root / "benchmarks" / "probe"
    run_root = root / "runs" / "baseline" / "q1"
    index = root / "runs" / "index"
    benchmark.mkdir(parents=True)
    run_root.mkdir(parents=True)
    index.mkdir(parents=True)
    baseline_judge = root / "runs" / "baseline-judge.json"
    gold = root / "runs" / "gold.json"
    target_oracle = root / "runs" / "target-oracle.json"
    span_oracle = root / "runs" / "span-oracle.json"
    judge_manifest = benchmark / "judge.json"
    judge_calibration = root / "runs" / "judge-calibration.json"
    frozen = benchmark / "frozen.txt"
    baseline_judge.write_text(
        json.dumps({"observations": [{"query_id": "q1", "correct": False}]}),
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "query_id": "q1",
                        "question": "Which project is named?",
                        "answer": "Aurora Project",
                        "gold_docids": ["answer-anchor"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    target_oracle.write_text(json.dumps({"decision": "pass"}), encoding="utf-8")
    span_oracle.write_text(json.dumps({"decision": "pass"}), encoding="utf-8")
    judge_manifest.write_text("{}", encoding="utf-8")
    judge_calibration.write_text("{}", encoding="utf-8")
    frozen.write_text("frozen", encoding="utf-8")
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "answer_text": "Exact Answer: Unknown",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Question: Which project is named in the "
                                    "acknowledgments section?\n\n"
                                    "Your response should be cited."
                                ),
                            }
                        ],
                    }
                ],
                "search_calls": [
                    {
                        "query": "project acknowledgments",
                        "outcome": "ok",
                        "results": [
                            {
                                "docid": "answer-anchor",
                                "snippet": "[BM25 anchor: full evidence]\nCandidate",
                            },
                            {
                                "docid": "noise-lead",
                                "snippet": "[Dense lead preview; docid=noise-lead]\nNoise",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registration = benchmark / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "monotonic-post-run-overlay-registration-v0",
                "status": "outcome_selected_development_calibration",
                "registered_at": "2026-08-16T00:00:00+00:00",
                "purpose": "test",
                "baseline_run_root": "runs/baseline",
                "baseline_judge": {
                    "path": "runs/baseline-judge.json",
                    "sha256": _hash(baseline_judge),
                },
                "gold_slice": {"path": "runs/gold.json", "sha256": _hash(gold)},
                "target_oracle": {
                    "path": "runs/target-oracle.json",
                    "sha256": _hash(target_oracle),
                },
                "span_oracle": {
                    "path": "runs/span-oracle.json",
                    "sha256": _hash(span_oracle),
                },
                "query_ids": ["q1"],
                "document_index_path": "runs/index",
                "target_selector_id": "obligation_channel_slate_v0",
                "maximum_target_search_calls": 1,
                "target_slots_per_channel": 1,
                "maximum_targets_per_case": 2,
                "span_selector_id": "answer_obligation_window_v2",
                "maximum_span_characters": 400,
                "provider": {
                    "model": "fixture-model",
                    "base_url": "https://example.invalid",
                    "api_key_env": "FIXTURE_KEY",
                    "maximum_calls": 1,
                    "maximum_output_tokens_per_call": 200,
                    "maximum_estimated_cost_usd": 0.01,
                },
                "judge": {
                    "manifest": {
                        "path": "benchmarks/probe/judge.json",
                        "sha256": _hash(judge_manifest),
                    },
                    "calibration": {
                        "path": "runs/judge-calibration.json",
                        "sha256": _hash(judge_calibration),
                    },
                    "base_url": "http://127.0.0.1:1/v1",
                    "served_model_name": "fixture-judge",
                    "maximum_calls": 1,
                },
                "frozen_artifacts": [
                    {"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}
                ],
                "acceptance": {
                    "minimum_literal_supported_replacements": 1,
                    "minimum_normalized_exact_matches": 1,
                    "minimum_judge_correct": 1,
                    "maximum_proposal_parse_failures": 0,
                    "maximum_unsupported_replacements": 0,
                    "maximum_provider_cost_usd": 0.01,
                },
                "sealed_holdout_access": "forbidden",
                "claim_boundary": "fixture only",
            }
        ),
        encoding="utf-8",
    )
    return registration


def test_overlay_runner_is_resumable_and_records_cost(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    registration = _write_overlay_fixture(root)
    provider = _OverlayProvider()
    judge_calls = 0

    def judge(_prompt: str) -> tuple[str, dict[str, object]]:
        nonlocal judge_calls
        judge_calls += 1
        return (
            "extracted_final_answer: Aurora Project\n"
            "reasoning: exact\n"
            "correct: yes\n"
            "confidence: 99",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    documents = {
        "answer-anchor": (
            "ACKNOWLEDGMENT This work supported the Aurora Project."
        ),
        "noise-lead": "Unrelated evidence.",
    }
    output = root / "runs" / "overlay"
    result = run_post_run_overlay_calibration(
        registration_path=registration,
        output_dir=output,
        provider=provider,
        document_loader=documents.get,
        judge=judge,
    )
    assert result.decision == "pass"
    assert result.normalized_exact_matches == 1
    assert result.judge_correct == 1
    assert result.provider_estimated_cost_usd == 0.001
    assert provider.calls == 1
    assert judge_calls == 1

    resumed = run_post_run_overlay_calibration(
        registration_path=registration,
        output_dir=output,
        provider=provider,
        document_loader=lambda _docid: None,
        judge=lambda _prompt: (_ for _ in ()).throw(AssertionError("no Judge call")),
        resume=True,
    )
    assert resumed == result
    assert provider.calls == 1
    assert judge_calls == 1
