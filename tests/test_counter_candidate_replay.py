from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.contracts import Usage
from deepresearch_harness.counter_candidate_replay import (
    build_replay_query,
    run_candidate_replay,
)
from deepresearch_harness.counter_hypothesis_packet import (
    BridgeCandidate,
    BridgePacket,
    CounterProbeItem,
    CounterProbeResult,
    DecisionGate,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _packet() -> BridgePacket:
    return BridgePacket(
        bridges=(
            BridgeCandidate(type="entity", term="Haile Gebrselassie"),
            BridgeCandidate(type="geography", term="Ethiopia"),
            BridgeCandidate(type="topic", term="marathon running"),
        ),
        selected=0,
        query="Haile Gebrselassie foundation founded year",
    )


def test_replay_query_substitutes_instead_of_accumulating_selected_bridge() -> None:
    assert build_replay_query(_packet(), 0) == "Haile Gebrselassie foundation founded year"
    assert build_replay_query(_packet(), 1) == "ethiopia foundation founded year"
    assert build_replay_query(_packet(), 2) == "marathon running foundation founded year"


def test_replay_uses_raw_packet_from_validation_failure_and_finds_unselected_rescue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    benchmark_dir = root / "benchmarks" / "probe"
    source_dir = root / "runs" / "source"
    benchmark_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    raw = _packet().model_dump_json()
    source = source_dir / "probe.json"
    result = CounterProbeResult(
        created_at="2026-08-15T00:00:00+00:00",
        status="failed",
        decision="reject",
        registration_sha256="a" * 64,
        query_count=1,
        provider_attempts=1,
        search_calls=0,
        parse_failures=1,
        request_failures=0,
        baseline_gold_hit_cases=0,
        candidate_gold_hit_cases=0,
        gold_hit_case_delta=0,
        total_usage=Usage(),
        provider_cost_observability="complete",
        context_isolation_verified=True,
        items=(
            CounterProbeItem(
                query_id="q1",
                status="failed",
                question_sha256="b" * 64,
                prompt_sha256="c" * 64,
                context_fields=("question", "unresolved_obligation"),
                baseline_gold_hits=(),
                packet=None,
                selected_novel_terms=(),
                candidate_docids=(),
                candidate_gold_hits=(),
                candidate_gold_hit_ranks=(),
                raw_completion_sha256=sha256(raw.encode()).hexdigest(),
                raw_completion_text=raw,
                usage=Usage(),
                provider_latency_ms=1,
                search_latency_ms=0,
                error="validation failure",
            ),
        ),
        gates=(DecisionGate(gate_id="x", observed=0, operator="eq", threshold=1, passed=False),),
        next_action="close_without_end_to_end_expansion",
        claim_boundary="test",
    )
    source.write_text(result.model_dump_json(), encoding="utf-8")
    gold = source_dir / "gold.json"
    gold.write_text(
        json.dumps({"rows": [{"query_id": "q1", "gold_docids": ["gold-doc"]}]}),
        encoding="utf-8",
    )
    frozen = benchmark_dir / "frozen.txt"
    frozen.write_text("fixed", encoding="utf-8")
    spec = benchmark_dir / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "counter-candidate-replay-spec-v0",
                "status": "registered_before_local_replay",
                "registered_at": "2026-08-15T00:00:00+00:00",
                "source_result": {"path": "runs/source/probe.json", "sha256": _hash(source)},
                "gold_slice": {"path": "runs/source/gold.json", "sha256": _hash(gold)},
                "query_ids": ["q1"],
                "search_url": "http://127.0.0.1:8768/search",
                "retriever_id": "fixed",
                "candidates_per_case": 3,
                "max_search_results": 20,
                "query_transformation": "replace_selected_bridge_in_preserved_query_v0",
                "frozen_artifacts": [{"path": "benchmarks/probe/frozen.txt", "sha256": _hash(frozen)}],
                "claim_boundary": "test",
            }
        ),
        encoding="utf-8",
    )

    def fake_search(_url: str, _run_id: str, query: str, _timeout: int):
        return (("gold-doc",), 1) if query.startswith("ethiopia") else (("noise",), 1)

    monkeypatch.setattr(
        "deepresearch_harness.counter_candidate_replay._search",
        fake_search,
    )
    replay = run_candidate_replay(
        spec_path=spec,
        output_path=root / "runs" / "replay" / "result.json",
    )
    assert replay.search_calls == 3
    assert replay.any_candidate_gold_hit_cases == 1
    assert replay.unselected_rescue_cases == 1
    assert replay.diagnosis == "selection_bottleneck_present"
