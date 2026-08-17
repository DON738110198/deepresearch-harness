import json
import sys
from pathlib import Path

import pytest

from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.cli import main
from deepresearch_harness.livedrbench_fresh_pair import (
    load_and_validate_fresh_public_pair,
    prepare_fresh_public_pair,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "benchmarks" / "livedrbench_fresh_public_v0" / "registration.json"


def _base_config() -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "provider": {
                "kind": "openai_compatible",
                "model": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
                "thinking_mode": "disabled",
                "pricing": {
                    "input_per_million_usd": 0.14,
                    "output_per_million_usd": 0.28,
                },
            },
            "search": {"kind": "local"},
            "run": {
                "max_evidence": 6,
                "budget": {
                    "max_total_tokens": 8000,
                    "max_estimated_cost_usd": 0.002,
                    "max_llm_calls": 3,
                    "max_output_tokens_per_call": 2048,
                },
            },
        }
    )


def _write_config(path: Path, config: HarnessConfig) -> None:
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def test_prepare_pair_freezes_derived_search_arms_without_network_or_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "common.json"
    _write_config(config_path, _base_config())

    manifest = prepare_fresh_public_pair(
        registration_path=REGISTRATION,
        base_config_path=config_path,
        output_dir=tmp_path / "runs",
        run_label="fresh-tavily-v0",
    )

    run_dir = tmp_path / "runs" / "fresh-tavily-v0"
    assert manifest.provider_calls_before_generation == 0
    assert manifest.selected_task_keys == (10, 23, 38, 86, 99)
    assert [arm.search_kind for arm in manifest.arms] == ["duckduckgo", "tavily"]
    assert manifest.executor_status == "not_implemented"
    assert manifest.retry_policy == "failed_only_explicit_resume_v0"
    baseline = json.loads((run_dir / "baseline.config.snapshot.json").read_text(encoding="utf-8"))
    candidate = json.loads((run_dir / "candidate.config.snapshot.json").read_text(encoding="utf-8"))
    assert baseline["search"]["kind"] == "duckduckgo"
    assert "api_key_env" not in baseline["search"] or baseline["search"]["api_key_env"] is None
    assert candidate["search"]["kind"] == "tavily"
    assert candidate["search"]["api_key_env"] == "TAVILY_API_KEY"
    assert candidate["search"]["max_search_calls"] == 5
    assert load_and_validate_fresh_public_pair(run_dir / "pair_manifest.json") == manifest
    assert (
        prepare_fresh_public_pair(
            registration_path=REGISTRATION,
            base_config_path=config_path,
            output_dir=tmp_path / "runs",
            run_label="fresh-tavily-v0",
        )
        == manifest
    )


def test_prepare_pair_rejects_a_changed_registered_budget(tmp_path: Path) -> None:
    config = _base_config().model_copy(
        update={
            "run": _base_config().run.model_copy(
                update={"max_evidence": 5}
            )
        }
    )
    config_path = tmp_path / "common.json"
    _write_config(config_path, config)

    with pytest.raises(ValueError, match="evidence cap"):
        prepare_fresh_public_pair(
            registration_path=REGISTRATION,
            base_config_path=config_path,
            output_dir=tmp_path / "runs",
            run_label="fresh-tavily-v1",
        )


def test_pair_validation_rejects_tampered_config_snapshot(tmp_path: Path) -> None:
    config_path = tmp_path / "common.json"
    _write_config(config_path, _base_config())
    manifest = prepare_fresh_public_pair(
        registration_path=REGISTRATION,
        base_config_path=config_path,
        output_dir=tmp_path / "runs",
        run_label="fresh-tavily-v2",
    )
    snapshot = tmp_path / "runs" / manifest.run_label / "candidate.config.snapshot.json"
    snapshot.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate config snapshot hash"):
        load_and_validate_fresh_public_pair(
            tmp_path / "runs" / manifest.run_label / "pair_manifest.json"
        )


def test_cli_registers_pair_without_network_or_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "common.json"
    _write_config(config_path, _base_config())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deepresearch-harness",
            "register-livedrbench-fresh-pair",
            "--registration",
            str(REGISTRATION),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-label",
            "fresh-tavily-cli",
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out
    assert "status=registered_before_generation" in output
    assert "provider_calls_before_generation=0" in output
