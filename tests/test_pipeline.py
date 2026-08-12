import json
from pathlib import Path

from deepresearch_harness.contracts import RunState, RunStatus
from deepresearch_harness.pipeline import BaselineResearchPipeline, LocalCorpusCollector
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
