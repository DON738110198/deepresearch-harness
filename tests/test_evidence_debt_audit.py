from __future__ import annotations

from deepresearch_harness.browsecomp_plus import PiBrowseCompRun
from deepresearch_harness.evidence_debt_audit import audit_pi_run


def _run(*, answer: str, snippets: dict[str, str]) -> PiBrowseCompRun:
    return PiBrowseCompRun.model_validate(
        {
            "schema_version": "pi-browsecomp-run-v0",
            "adapter_version": "pi-browsecomp-v8",
            "pi_version": "0.84.1",
            "run_id": "run-1",
            "query_id": "q1",
            "model": "deepseek-v4-flash",
            "thinking_level": "high",
            "max_search_results": 20,
            "compilation_thinking_level": "off",
            "control_policy": "answer_reserve_nonthinking_v0",
            "system_prompt": "",
            "prompt_sha256": "a" * 64,
            "started_at": "2026-08-15T00:00:00Z",
            "latency_ms": 1,
            "status": "succeeded",
            "stop_reason": "completed",
            "answer_text": answer,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 2,
                "cost_usd": 0.01,
            },
            "exploration_stop_reason": None,
            "bootstrap_stop_reason": None,
            "bootstrap_output_tokens": 0,
            "bootstrap_prompt_sha256": None,
            "exploration_output_tokens": 1,
            "compilation_output_tokens": 0,
            "answer_compiler_invoked": False,
            "answer_compiler_prompt_sha256": None,
            "first_tool_deadline_triggered": False,
            "first_tool_deadline_prompt_sha256": None,
            "answer_schema_complete": all(
                marker in answer
                for marker in ("Explanation:", "Exact Answer:", "Confidence:")
            ),
            "output_budget_overshoot_tokens": 0,
            "model_requests": 1,
            "provider_request_limits": [],
            "search_calls": [
                {
                    "query": "query",
                    "outcome": "ok",
                    "latency_ms": 1,
                    "results": [
                        {"docid": docid, "score": 1.0, "snippet": snippet}
                        for docid, snippet in snippets.items()
                    ],
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Question: In what year did Haile Example found the "
                                "Haile Example Foundation?\n\nYour response should be formatted."
                            ),
                        }
                    ],
                }
            ],
        }
    )


def test_audit_opens_debt_when_cited_year_relation_is_unsupported() -> None:
    run = _run(
        answer=(
            "Explanation: The clues point to Haile Example, but the source does not "
            "directly confirm the foundation year [20].\n"
            "Exact Answer: 2002\nConfidence: 85%"
        ),
        snippets={"20": "Haile Other competed in 2002 and won a medal."},
    )

    audit = audit_pi_run(run, source_run_sha256="b" * 64)

    assert audit.status == "open"
    assert audit.reasons == (
        "cited_evidence_does_not_support_exact_answer",
        "explicit_unresolved_evidence",
    )
    assert audit.provider_calls == 0
    assert len(audit.repair_queries) == 2
    assert "Haile Example" in audit.repair_queries[0]


def test_audit_resolves_supported_entity_alias() -> None:
    run = _run(
        answer=(
            "Explanation: The source identifies Guillem Vilella Falgueras, also known "
            "as Guille Milkyway [42].\n"
            "Exact Answer: Guillem Vilella Falgueras (Guille Milkyway)\nConfidence: 90%"
        ),
        snippets={"42": "Guille Milkyway is the stage name of Guillem Vilella Falgueras."},
    )

    audit = audit_pi_run(run, source_run_sha256="b" * 64)

    assert audit.status == "supported"
    assert audit.supporting_cited_docids == ("42",)
    assert audit.repair_queries == ()


def test_audit_conservatively_keeps_high_confidence_uncited_match_closed() -> None:
    run = _run(
        answer=(
            "Explanation: The book title follows from the identified author [42].\n"
            "Exact Answer: A Long Book Subtitle\nConfidence: 85%"
        ),
        snippets={"42": "The author published many books."},
    )

    audit = audit_pi_run(run, source_run_sha256="b" * 64)

    assert audit.status == "no_repair_trigger"
    assert audit.supporting_cited_docids == ()


def test_audit_marks_missing_answer_schema_unscorable() -> None:
    run = _run(answer="I do not know [20].", snippets={"20": "Evidence"})

    audit = audit_pi_run(run, source_run_sha256="b" * 64)

    assert audit.status == "unscorable"
    assert audit.reasons == ("answer_schema_missing",)
