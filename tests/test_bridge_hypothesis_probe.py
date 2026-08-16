from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.bridge_hypothesis_probe import (
    BridgeHypothesis,
    BridgeQueryPlan,
    build_bridge_prompt,
    run_oracle_replay,
    validate_plan_for_case,
)
from deepresearch_harness.repair_query_probe import RepairQueryCase


def _case() -> RepairQueryCase:
    draft = "Explanation: The clues suggest a lawyer. Exact Answer: Wrong Name Confidence: 20%"
    return RepairQueryCase(
        query_id="q1",
        question="Who was adopted, made partner, dismissed, and later knighted?",
        draft_answer=draft,
        draft_answer_sha256=sha256(draft.encode()).hexdigest(),
        exact_answer="Wrong Name",
        subject_hypothesis="a lawyer",
        audit_reasons=("cited_evidence_does_not_support_exact_answer",),
        uncertainty_phrases=("suggest",),
        failed_repair_query='"Wrong Name" biography',
        baseline_docids=("noise",),
        gold_docids=("gold-doc",),
    )


def _plan() -> BridgeQueryPlan:
    return BridgeQueryPlan(
        unresolved_obligation="Identify the person.",
        hypotheses=(
            BridgeHypothesis(
                bridge_type="domain",
                bridge_terms=("architect",),
                rationale="Practice and partner can refer to architecture.",
            ),
            BridgeHypothesis(
                bridge_type="geography",
                bridge_terms=("New Zealand",),
                rationale="The prime-minister clue may indicate New Zealand.",
            ),
            BridgeHypothesis(
                bridge_type="entity",
                bridge_terms=("design firm",),
                rationale="A design-firm biography may use the career vocabulary.",
            ),
        ),
        selected_index=0,
        query="architect adopted partner dismissed practice knighthood",
    )


def test_bridge_plan_requires_distinct_hypothesis_types() -> None:
    with pytest.raises(ValueError, match="distinct types"):
        BridgeQueryPlan(
            unresolved_obligation="Identify the person.",
            hypotheses=(
                BridgeHypothesis(bridge_type="domain", bridge_terms=("architect",), rationale="a"),
                BridgeHypothesis(bridge_type="domain", bridge_terms=("lawyer",), rationale="b"),
                BridgeHypothesis(bridge_type="entity", bridge_terms=("designer",), rationale="c"),
            ),
            selected_index=0,
            query="architect adopted partner dismissed practice knighthood",
        )


def test_bridge_plan_must_add_and_use_a_novel_term() -> None:
    case = _case()
    plan = _plan()
    assert validate_plan_for_case(
        case=case,
        baseline_query="adopted lawyer partner dismissed practice knighthood",
        plan=plan,
    ) == ("architect",)
    prompt = build_bridge_prompt(case, "adopted lawyer partner dismissed practice knighthood")
    assert "Return exactly one JSON object" in prompt
    assert "gold-doc" not in prompt

    repeated = plan.model_copy(
        update={
            "hypotheses": (
                BridgeHypothesis(bridge_type="domain", bridge_terms=("lawyer",), rationale="a"),
                plan.hypotheses[1],
                plan.hypotheses[2],
            ),
            "query": "lawyer adopted partner dismissed practice knighthood",
        }
    )
    with pytest.raises(ValueError, match="adds no term"):
        validate_plan_for_case(
            case=case,
            baseline_query="adopted lawyer partner dismissed practice knighthood",
            plan=repeated,
        )


def test_posthoc_oracle_is_explicit_and_scores_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    benchmark_dir = root / "benchmarks" / "probe"
    runs_dir = root / "runs" / "source"
    benchmark_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    summary = runs_dir / "summary.json"
    gold = runs_dir / "gold.json"
    summary.write_text(json.dumps({"items": []}), encoding="utf-8")
    gold.write_text(
        json.dumps({"rows": [{"query_id": "q1", "gold_docids": ["gold-doc"]}]}),
        encoding="utf-8",
    )
    spec = benchmark_dir / "oracle.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "contrastive-bridge-oracle-spec-v0",
                "status": "posthoc_not_preregistered",
                "source_summary": {
                    "path": "runs/source/summary.json",
                    "sha256": sha256(summary.read_bytes()).hexdigest(),
                },
                "gold_slice": {
                    "path": "runs/source/gold.json",
                    "sha256": sha256(gold.read_bytes()).hexdigest(),
                },
                "search_url": "http://127.0.0.1:8768/search",
                "retriever_id": "fixed",
                "max_search_results": 20,
                "cases": [
                    {
                        "query_id": "q1",
                        "bridge_type": "domain",
                        "bridge_terms": ["architect"],
                        "query": "architect adopted partner dismissed practice knighthood",
                        "provenance": "manual_after_gold_inspection",
                    }
                ],
                "claim_boundary": "post-hoc diagnostic only",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "deepresearch_harness.bridge_hypothesis_probe._search",
        lambda *_: (("gold-doc", "noise"), 3),
    )
    result = run_oracle_replay(
        spec_path=spec,
        output_path=root / "runs" / "oracle" / "result.json",
    )
    assert result.status == "posthoc_oracle_diagnostic"
    assert result.gold_hit_cases == 1
    assert result.items[0].gold_hit_ranks == (1,)
