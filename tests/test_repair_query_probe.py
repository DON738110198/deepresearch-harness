from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.contracts import Usage
from deepresearch_harness.providers import Completion, LLMProvider
from deepresearch_harness.repair_query_probe import (
    RepairQueryCase,
    RepairQueryPlan,
    build_repair_query_prompt,
    load_probe_cases,
    load_registration,
    run_probe,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StubProvider(LLMProvider):
    name = "stub"
    model = "deepseek-v4-flash"

    def complete(
        self,
        *,
        stage: str,
        prompt: str,
        json_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> Completion:
        assert stage == "repair_query"
        assert json_output is True
        assert max_output_tokens == 128
        assert "gold-doc" not in prompt
        return Completion(
            text=json.dumps(
                {
                    "query": "architect adopted partner dismissed practice knighthood",
                    "unresolved_obligation": "Identify the person from the rare career constraints.",
                    "strategy": "counter_hypothesis",
                }
            ),
            usage=Usage(
                input_tokens=100,
                output_tokens=20,
                estimated_cost_usd=0.001,
            ),
            latency_ms=12,
        )


def test_repair_query_plan_rejects_refusal_and_non_search_text() -> None:
    with pytest.raises(ValueError, match="refusal"):
        RepairQueryPlan(
            query="cannot be determined from available evidence biography",
            unresolved_obligation="Find the entity.",
            strategy="entity_verification",
        )
    with pytest.raises(ValueError, match="5 to 18"):
        RepairQueryPlan(
            query="too short",
            unresolved_obligation="Find the entity.",
            strategy="entity_verification",
        )


def test_prompt_excludes_gold_and_marks_failed_query() -> None:
    case = RepairQueryCase(
        query_id="q1",
        question="Which architect matched these career constraints?",
        draft_answer="Exact Answer: Unsupported Name\nConfidence: 20%",
        draft_answer_sha256="a" * 64,
        exact_answer="Unsupported Name",
        subject_hypothesis="a lawyer",
        audit_reasons=("cited_evidence_does_not_support_exact_answer",),
        uncertainty_phrases=("unverified",),
        failed_repair_query='"Unsupported Name" biography',
        baseline_docids=("noise",),
        gold_docids=("gold-doc",),
    )
    prompt = build_repair_query_prompt(case)
    assert "gold-doc" not in prompt
    assert "Return exactly one JSON object" in prompt
    assert json.loads(prompt)["input"]["failed_repair_query"] == '"Unsupported Name" biography'
    assert "counter-hypothesis" in prompt


def test_probe_is_prediction_bound_and_scores_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    registration_dir = root / "benchmarks" / "probe"
    source_dir = root / "runs" / "source"
    query_dir = source_dir / "q1"
    frozen_path = root / "src" / "frozen.txt"
    registration_dir.mkdir(parents=True)
    query_dir.mkdir(parents=True)
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text("frozen\n", encoding="utf-8")

    draft = "Explanation: unsupported.\nExact Answer: Unsupported Name\nConfidence: 20%"
    draft_hash = sha256(draft.encode("utf-8")).hexdigest()
    (source_dir / "summary.json").write_text(
        json.dumps({"items": [{"query_id": "q1", "status": "succeeded"}]}),
        encoding="utf-8",
    )
    (source_dir / "gold.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "query_id": "q1",
                        "gold_docids": ["gold-doc"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (query_dir / "request.json").write_text(
        json.dumps({"question": "Which person matches the rare career constraints?"}),
        encoding="utf-8",
    )
    (query_dir / "run.json").write_text(
        json.dumps(
            {
                "answer_first_audit": {
                    "audit_status": "open",
                    "draft_answer_sha256": draft_hash,
                    "exact_answer": "Unsupported Name",
                    "subject_hypothesis": "a lawyer",
                    "reasons": ["cited_evidence_does_not_support_exact_answer"],
                    "explicit_uncertainty_phrases": ["unsupported"],
                    "repair_query": '"Unsupported Name" biography',
                    "repair_returned_docids": ["noise"],
                },
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": draft}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registration_path = registration_dir / "registration.json"
    registration = {
        "schema_version": "obligation-repair-query-probe-registration-v0",
        "status": "registered_pre_provider",
        "registered_at": "2026-08-15T00:00:00+00:00",
        "purpose": "test",
        "source_summary": {
            "path": "runs/source/summary.json",
            "sha256": _hash(source_dir / "summary.json"),
        },
        "gold_slice": {
            "path": "runs/source/gold.json",
            "sha256": _hash(source_dir / "gold.json"),
        },
        "query_ids": ["q1"],
        "fixed_contract": {
            "model": "deepseek-v4-flash",
            "thinking_mode": "disabled",
            "system_prompt_policy": "provider default",
            "retriever_id": "fixed-retriever",
            "search_url": "http://127.0.0.1:8768/search",
            "max_search_results": 20,
            "max_planner_output_tokens": 128,
            "provider_attempts_per_case": 1,
            "search_calls_per_case": 1,
            "sealed_holdout_access": "forbidden",
        },
        "frozen_artifacts": [
            {"path": "src/frozen.txt", "sha256": _hash(frozen_path)}
        ],
        "acceptance": {
            "minimum_candidate_gold_hit_cases": 1,
            "minimum_gold_hit_case_delta": 1,
            "parse_failures_must_equal": 0,
            "request_failures_must_equal": 0,
            "maximum_provider_cost_usd": 0.01,
        },
        "claim_boundary": "test only",
    }
    registration_path.write_text(json.dumps(registration), encoding="utf-8")

    monkeypatch.setattr(
        "deepresearch_harness.repair_query_probe._search",
        lambda **_: (("gold-doc", "noise-2"), 4),
    )
    loaded = load_registration(registration_path)
    cases = load_probe_cases(repository_root=root, registration=loaded)
    assert cases[0].draft_answer == draft
    result = run_probe(
        registration_path=registration_path,
        provider=StubProvider(),
        output_path=root / "runs" / "probe" / "result.json",
    )
    assert result.decision == "pass"
    assert result.provider_attempts == 1
    assert result.search_calls == 1
    assert result.baseline_gold_hit_cases == 0
    assert result.candidate_gold_hit_cases == 1
    assert result.gold_hit_case_delta == 1
