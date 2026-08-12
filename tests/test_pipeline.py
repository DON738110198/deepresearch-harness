import json
from pathlib import Path

import pytest

from deepresearch_harness.contracts import BudgetLimits, RunState, RunStatus
from deepresearch_harness.pipeline import (
    BaselineResearchPipeline,
    BudgetExceeded,
    LocalCorpusCollector,
    ObligationEvidenceDebtPipeline,
    SearchWritePipeline,
)
from deepresearch_harness.providers import FakeProvider


ROOT = Path(__file__).resolve().parents[1]


def make_pipeline(tmp_path: Path) -> BaselineResearchPipeline:
    return BaselineResearchPipeline(
        provider=FakeProvider(),
        collector=LocalCorpusCollector(ROOT / "examples" / "offline_corpus.json"),
        output_dir=tmp_path,
    )


def test_offline_baseline_persists_auditable_cited_run(tmp_path: Path) -> None:
    state = make_pipeline(tmp_path).run("What evidence supports a phased rollout?")

    assert state.status is RunStatus.SUCCEEDED
    assert state.plan is not None
    assert state.evidence and state.claims and state.citations
    assert [event.stage for event in state.trace] == ["plan", "collect", "ledger", "write"]
    assert state.total_usage.input_tokens > 0

    persisted = RunState.model_validate_json((tmp_path / state.run_id / "state.json").read_text(encoding="utf-8"))
    report = Path(persisted.report_path).read_text(encoding="utf-8")
    assert persisted.status is RunStatus.SUCCEEDED
    assert all(citation.marker in report for citation in persisted.citations)


def test_fake_provider_is_deterministic(tmp_path: Path) -> None:
    first = make_pipeline(tmp_path / "first").run("Does a phased rollout need observability?")
    second = make_pipeline(tmp_path / "second").run("Does a phased rollout need observability?")

    assert first.plan == second.plan
    assert [claim.text for claim in first.claims] == [claim.text for claim in second.claims]
    assert json.loads((tmp_path / "first" / first.run_id / "state.json").read_text(encoding="utf-8"))["task"]["question"] == "Does a phased rollout need observability?"


def test_b0_search_write_uses_one_llm_call_and_shared_audit_contract(tmp_path: Path) -> None:
    pipeline = SearchWritePipeline(
        provider=FakeProvider(),
        collector=LocalCorpusCollector(ROOT / "examples" / "offline_corpus.json"),
        output_dir=tmp_path,
    )

    state = pipeline.run("What evidence supports a phased rollout?")

    assert state.variant == "b0_search_write"
    assert state.plan is None
    assert [event.stage for event in state.trace] == ["collect", "direct_write"]
    assert state.claims and state.citations
    assert Path(state.report_path).exists()


def test_b1_stops_with_explicit_reason_when_call_budget_is_exhausted(tmp_path: Path) -> None:
    pipeline = BaselineResearchPipeline(
        provider=FakeProvider(),
        collector=LocalCorpusCollector(ROOT / "examples" / "offline_corpus.json"),
        output_dir=tmp_path,
        budget_limits=BudgetLimits(max_llm_calls=1, max_output_tokens_per_call=1000),
    )

    with pytest.raises(BudgetExceeded, match="LLM call budget exhausted"):
        pipeline.run("What evidence supports a phased rollout?")

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    state = RunState.model_validate_json((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state.status is RunStatus.FAILED
    assert state.stop_reason == "budget_exhausted"
    assert state.trace[-1].stage == "ledger"
    assert state.trace[-1].outcome == "error"


def test_b2_persists_obligation_to_evidence_debt_links_without_extra_calls(tmp_path: Path) -> None:
    pipeline = ObligationEvidenceDebtPipeline(
        provider=FakeProvider(),
        collector=LocalCorpusCollector(ROOT / "examples" / "offline_corpus.json"),
        output_dir=tmp_path,
    )

    state = pipeline.run("What evidence supports a phased rollout?")

    assert state.variant == "b2_obligation_evidence_debt"
    assert state.plan is not None and len(state.plan.obligations) == 3
    assert {item.obligation_id for item in state.evidence_debts} == {
        item.id for item in state.plan.obligations
    }
    assert all(item.status == "resolved" for item in state.evidence_debts)
    assert [event.stage for event in state.trace] == ["plan", "collect", "ledger", "write"]
    assert len([event for event in state.trace if event.provider == "fake"]) == 3

    persisted = RunState.model_validate_json((tmp_path / state.run_id / "state.json").read_text(encoding="utf-8"))
    assert persisted.evidence_debts == state.evidence_debts
