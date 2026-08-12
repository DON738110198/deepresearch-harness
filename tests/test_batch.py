from pathlib import Path

from deepresearch_harness.batch import run_experiment_batch
from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.providers import FakeProvider


ROOT = Path(__file__).resolve().parents[1]


def test_offline_batch_persists_all_variant_task_records(tmp_path: Path) -> None:
    config = HarnessConfig.model_validate(
        {
            "provider": {"kind": "fake"},
            "run": {"max_evidence": 6},
        }
    )

    summary = run_experiment_batch(
        manifest_path=ROOT / "experiments" / "pilot_v0" / "token_matched.json",
        config=config,
        output_root=tmp_path,
        provider=FakeProvider(),
        enforce_provider_match=False,
    )

    assert len(summary.records) == 20
    assert summary.status == "succeeded"
    assert all(aggregate.completed == 10 for aggregate in summary.aggregates.values())
    assert (Path(summary.output_dir) / "summary.json").exists()
    assert all(Path(record.state_path).exists() for record in summary.records)
    assert all(Path(record.score_path).exists() for record in summary.records)
