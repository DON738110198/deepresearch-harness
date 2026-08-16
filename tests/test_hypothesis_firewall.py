from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.contracts import Usage
from deepresearch_harness.hypothesis_firewall import (
    FirewallAnchor,
    HypothesisFirewallSlate,
    build_firewall_prompt,
    run_firewall_probe,
    validate_slate_for_case,
)
from deepresearch_harness.providers import Completion, LLMProvider


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _slate() -> HypothesisFirewallSlate:
    return HypothesisFirewallSlate(
        anchors=(
            FirewallAnchor(
                role="seed_expansion",
                type="work",
                term="Genghis Khan",
                query="John Man Genghis Khan 2014 book subtitle",
            ),
            FirewallAnchor(
                role="counter_hypothesis",
                type="domain",
                term="Mongol history",
                query="Mongol history 2014 book acknowledgments subtitle",
            ),
            FirewallAnchor(
                role="counter_hypothesis",
                type="entity",
                term="Kublai Khan",
                query="Kublai Khan 2014 author book subtitle acknowledgments",
            ),
        )
    )


def test_firewall_prompt_tags_seed_and_excludes_answer_channels() -> None:
    prompt = build_firewall_prompt(
        question="Which 2014 book subtitle is requested?",
        unresolved_obligation="Identify the exact subtitle.",
        unverified_prior_subject="John Man",
    )
    payload = json.loads(prompt)
    assert payload["input"] == {
        "question": "Which 2014 book subtitle is requested?",
        "unresolved_obligation": "Identify the exact subtitle.",
        "unverified_prior_subject": "John Man",
    }
    assert "JSON" in prompt
    for forbidden in (
        "prior_exact_answer",
        "draft_answer",
        "prior_query",
        "gold_docids",
        "Unable to determine",
    ):
        assert forbidden not in prompt


def test_firewall_requires_seed_expansion_and_independent_counters() -> None:
    validate_slate_for_case(
        question="Which 2014 book subtitle is requested?",
        unverified_prior_subject="John Man",
        slate=_slate(),
    )
    copied = _slate().model_copy(
        update={
            "anchors": (
                _slate().anchors[0],
                FirewallAnchor(
                    role="counter_hypothesis",
                    type="entity",
                    term="John Man",
                    query="John Man 2014 author book subtitle",
                ),
                _slate().anchors[2],
            )
        }
    )
    with pytest.raises(ValueError, match="adds no vocabulary"):
        validate_slate_for_case(
            question="Which 2014 book subtitle is requested?",
            unverified_prior_subject="John Man",
            slate=copied,
        )


class StubProvider(LLMProvider):
    name = "stub"
    model = "frozen-model"

    def complete(
        self,
        *,
        stage: str,
        prompt: str,
        json_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> Completion:
        assert stage == "hypothesis_firewall_slate"
        assert json_output is True
        assert max_output_tokens == 320
        return Completion(
            text=_slate().model_dump_json(),
            usage=Usage(input_tokens=50, output_tokens=50, estimated_cost_usd=0.001),
            latency_ms=5,
        )


def test_offline_firewall_probe_searches_all_anchors_and_scores_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    benchmark_dir = root / "benchmarks" / "probe"
    source_dir = root / "runs" / "source"
    case_dir = source_dir / "q1"
    benchmark_dir.mkdir(parents=True)
    case_dir.mkdir(parents=True)
    summary = source_dir / "summary.json"
    request = case_dir / "request.json"
    run = case_dir / "run.json"
    gold = source_dir / "gold.json"
    baseline = source_dir / "baseline.json"
    oracle = source_dir / "oracle.json"
    frozen = benchmark_dir / "frozen.txt"
    summary.write_text(json.dumps({"items": [{"query_id": "q1", "status": "succeeded"}]}), encoding="utf-8")
    request.write_text(json.dumps({"question": "Which 2014 book subtitle is requested?"}), encoding="utf-8")
    run.write_text(json.dumps({"answer_first_audit": {"audit_status": "open", "subject_hypothesis": "John Man"}}), encoding="utf-8")
    gold.write_text(json.dumps({"rows": [{"query_id": "q1", "gold_docids": ["gold-doc"]}]}), encoding="utf-8")
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "counter-candidate-replay-v0",
                "created_at": "2026-08-15T00:00:00+00:00",
                "status": "post_generation_diagnostic_not_effectiveness",
                "spec_sha256": "a" * 64,
                "query_count": 1,
                "search_calls": 0,
                "source_selected_gold_hit_cases": 0,
                "replay_selected_gold_hit_cases": 0,
                "any_candidate_gold_hit_cases": 0,
                "unselected_rescue_cases": 0,
                "diagnosis": "candidate_generation_bottleneck",
                "items": [],
                "claim_boundary": "test",
            }
        ),
        encoding="utf-8",
    )
    oracle.write_text(json.dumps({"gold_hit_cases": 1}), encoding="utf-8")
    frozen.write_text("fixed", encoding="utf-8")
    registration = benchmark_dir / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "hypothesis-firewall-slate-registration-v0",
                "status": "registered_pre_provider",
                "registered_at": "2026-08-15T00:00:00+00:00",
                "purpose": "test",
                "source_summary": {"path": "runs/source/summary.json", "sha256": _hash(summary)},
                "case_sources": [{"query_id": "q1", "request": {"path": "runs/source/q1/request.json", "sha256": _hash(request)}, "run": {"path": "runs/source/q1/run.json", "sha256": _hash(run)}}],
                "gold_slice": {"path": "runs/source/gold.json", "sha256": _hash(gold)},
                "baseline_replay": {"path": "runs/source/baseline.json", "sha256": _hash(baseline)},
                "oracle_replay": {"path": "runs/source/oracle.json", "sha256": _hash(oracle)},
                "query_ids": ["q1"],
                "fixed_contract": {
                    "model": "frozen-model",
                    "thinking_mode": "disabled",
                    "system_prompt_policy": "fixed",
                    "retriever_id": "fixed",
                    "search_url": "http://127.0.0.1:8768/search",
                    "context_fields": ["question", "unresolved_obligation", "unverified_prior_subject"],
                    "forbidden_context_fields": ["draft_answer", "prior_exact_answer", "prior_query", "gold_docids"],
                    "anchors_per_case": 3,
                    "max_search_results": 20,
                    "max_planner_output_tokens": 320,
                    "provider_attempts_per_case": 1,
                    "search_calls_per_case": 3,
                    "sealed_holdout_access": "forbidden",
                },
                "frozen_artifacts": [{"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}],
                "acceptance": {
                    "minimum_candidate_gold_hit_cases": 1,
                    "minimum_gold_hit_case_delta": 1,
                    "parse_failures_must_equal": 0,
                    "provider_request_failures_must_equal": 0,
                    "search_failures_must_equal": 0,
                    "maximum_provider_cost_usd": 0.05,
                },
                "claim_boundary": "test",
            }
        ),
        encoding="utf-8",
    )

    def fake_search(_url: str, _run_id: str, query: str, _timeout: int):
        return (("gold-doc",), 1) if "genghis" in query.casefold() else (("noise",), 1)

    monkeypatch.setattr("deepresearch_harness.hypothesis_firewall._search", fake_search)
    result = run_firewall_probe(
        registration_path=registration,
        provider=StubProvider(),
        output_path=root / "runs" / "probe" / "result.json",
    )
    assert result.decision == "pass"
    assert result.search_calls == 3
    assert result.candidate_gold_hit_cases == 1
    assert result.items[0].raw_completion_text == _slate().model_dump_json()
