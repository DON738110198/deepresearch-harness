from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.contracts import Usage
from deepresearch_harness.counter_hypothesis_packet import (
    BridgeCandidate,
    BridgePacket,
    build_counter_hypothesis_prompt,
    run_counter_probe,
    validate_packet_for_question,
)
from deepresearch_harness.providers import Completion, LLMProvider


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _packet() -> BridgePacket:
    return BridgePacket(
        bridges=(
            BridgeCandidate(type="domain", term="architect"),
            BridgeCandidate(type="geography", term="New Zealand"),
            BridgeCandidate(type="entity", term="design practice"),
        ),
        selected=0,
        query="architect adopted partner dismissed practice knighthood",
    )


def test_packet_requires_distinct_bridge_types() -> None:
    with pytest.raises(ValueError, match="three distinct types"):
        BridgePacket(
            bridges=(
                BridgeCandidate(type="domain", term="architect"),
                BridgeCandidate(type="domain", term="lawyer"),
                BridgeCandidate(type="entity", term="design practice"),
            ),
            selected=0,
            query="architect adopted partner dismissed practice knighthood",
        )


def test_prompt_is_structurally_draft_blind_and_packet_adds_bridge() -> None:
    question = "Who was adopted, made partner, dismissed, and later knighted?"
    obligation = "Identify the exact person and connect all clues."
    prompt = build_counter_hypothesis_prompt(
        question=question,
        unresolved_obligation=obligation,
    )
    payload = json.loads(prompt)
    assert payload["input"] == {
        "question": question,
        "unresolved_obligation": obligation,
    }
    assert "JSON" in prompt
    for forbidden in (
        "Peter Lampl",
        "John Man",
        "Cannot be determined",
        "prior_query",
        "gold_docids",
        "audited_draft",
    ):
        assert forbidden not in prompt
    assert validate_packet_for_question(question=question, packet=_packet()) == (
        "architect",
    )


def test_packet_rejects_generic_or_honorific_only_novelty() -> None:
    packet = _packet().model_copy(
        update={
            "bridges": (
                BridgeCandidate(type="domain", term="sir"),
                *_packet().bridges[1:],
            ),
            "query": "sir adopted partner dismissed practice knighthood",
        }
    )
    with pytest.raises(ValueError, match="generic or honorific"):
        validate_packet_for_question(
            question="Who was adopted, made partner, dismissed, and later knighted?",
            packet=packet,
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
        assert stage == "counter_hypothesis_packet"
        assert json_output is True
        assert max_output_tokens == 192
        return Completion(
            text=_packet().model_dump_json(),
            usage=Usage(input_tokens=20, output_tokens=20, estimated_cost_usd=0.001),
            latency_ms=7,
        )


def test_offline_probe_persists_raw_completion_and_scores_gate(
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
    gold = source_dir / "gold.json"
    baseline = source_dir / "baseline.json"
    oracle = source_dir / "oracle.json"
    frozen = benchmark_dir / "frozen.txt"
    summary.write_text(json.dumps({"items": [{"query_id": "q1", "status": "succeeded"}]}), encoding="utf-8")
    request.write_text(json.dumps({"question": "Who was adopted and later knighted?"}), encoding="utf-8")
    gold.write_text(json.dumps({"rows": [{"query_id": "q1", "gold_docids": ["gold-doc"]}]}), encoding="utf-8")
    baseline.write_text(json.dumps({"items": [{"query_id": "q1", "candidate_gold_hits": []}]}), encoding="utf-8")
    oracle.write_text(json.dumps({"gold_hit_cases": 1}), encoding="utf-8")
    frozen.write_text("fixed", encoding="utf-8")
    registration = benchmark_dir / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "draft-blind-counter-hypothesis-registration-v0",
                "status": "registered_pre_provider",
                "registered_at": "2026-08-15T00:00:00+00:00",
                "purpose": "offline test",
                "source_summary": {"path": "runs/source/summary.json", "sha256": _hash(summary)},
                "case_requests": [{"query_id": "q1", "path": "runs/source/q1/request.json", "sha256": _hash(request)}],
                "gold_slice": {"path": "runs/source/gold.json", "sha256": _hash(gold)},
                "baseline_probe": {"path": "runs/source/baseline.json", "sha256": _hash(baseline)},
                "oracle_replay": {"path": "runs/source/oracle.json", "sha256": _hash(oracle)},
                "query_ids": ["q1"],
                "fixed_contract": {
                    "model": "frozen-model",
                    "thinking_mode": "disabled",
                    "system_prompt_policy": "fixed",
                    "retriever_id": "fixed-retriever",
                    "search_url": "http://127.0.0.1:8768/search",
                    "context_fields": ["question", "unresolved_obligation"],
                    "forbidden_context_fields": ["draft_answer", "prior_candidate", "prior_query", "gold_docids"],
                    "bridges_per_case": 3,
                    "max_search_results": 20,
                    "max_planner_output_tokens": 192,
                    "provider_attempts_per_case": 1,
                    "search_calls_per_case": 1,
                    "sealed_holdout_access": "forbidden",
                },
                "frozen_artifacts": [{"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}],
                "acceptance": {
                    "minimum_candidate_gold_hit_cases": 1,
                    "minimum_gold_hit_case_delta": 1,
                    "parse_failures_must_equal": 0,
                    "request_failures_must_equal": 0,
                    "maximum_provider_cost_usd": 0.05,
                },
                "claim_boundary": "offline test only",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "deepresearch_harness.counter_hypothesis_packet._search",
        lambda *_: (("gold-doc", "noise"), 4),
    )
    result = run_counter_probe(
        registration_path=registration,
        provider=StubProvider(),
        output_path=root / "runs" / "probe" / "result.json",
    )
    assert result.decision == "pass"
    assert result.context_isolation_verified is True
    assert result.candidate_gold_hit_cases == 1
    assert result.items[0].raw_completion_text == _packet().model_dump_json()
    assert result.items[0].raw_completion_sha256 is not None
