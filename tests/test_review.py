from pathlib import Path

from deepresearch_harness.batch import run_experiment_batch
from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.providers import FakeProvider
from deepresearch_harness.review import prepare_blind_review


ROOT = Path(__file__).resolve().parents[1]


def test_review_packet_hides_variants_and_writes_separate_key(tmp_path: Path) -> None:
    config = HarnessConfig.model_validate({"provider": {"kind": "fake"}, "run": {"max_evidence": 6}})
    summary = run_experiment_batch(
        manifest_path=ROOT / "experiments" / "pilot_v0" / "token_matched.json",
        config=config,
        output_root=tmp_path / "batch",
        provider=FakeProvider(),
        enforce_provider_match=False,
    )
    summary_path = Path(summary.output_dir) / "summary.json"

    packet_path, template_path, key_path = prepare_blind_review(
        summary_path=summary_path,
        suite_path=ROOT / "benchmarks" / "pilot_v0" / "tasks.json",
        output_dir=tmp_path / "review",
    )

    packet_text = packet_path.read_text(encoding="utf-8")
    assert "b0_search_write" not in packet_text
    assert "b1_plan_search_ledger_write" not in packet_text
    assert '"query"' not in packet_text
    assert '"candidate_id": "A"' in packet_text
    assert len(__import__("json").loads(template_path.read_text(encoding="utf-8"))) == 20
    key_text = key_path.read_text(encoding="utf-8")
    assert "b0_search_write" in key_text
    assert "b1_plan_search_ledger_write" in key_text
