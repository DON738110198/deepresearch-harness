import json
import sys
from pathlib import Path

import pytest

from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.cli import main
import deepresearch_harness.livedrbench_fresh_pair as fresh_pair
from deepresearch_harness.livedrbench_fresh_pair import (
    PairTaskAttempt,
    execute_fresh_public_pair,
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
    assert manifest.executor_status == "implemented_unexecuted"
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


def test_executor_preserves_successes_and_retries_only_failed_tasks_when_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "common.json"
    _write_config(config_path, _base_config())
    manifest = prepare_fresh_public_pair(
        registration_path=REGISTRATION,
        base_config_path=config_path,
        output_dir=tmp_path / "runs",
        run_label="fresh-tavily-execute",
    )
    tasks = tuple(
        fresh_pair.LiveDRBenchTask(
            key=key,
            category="entities",
            question=f"question {key}",
            ground_truths=[],
        )
        for key in manifest.selected_task_keys
    )
    calls: list[tuple[str, int, int]] = []

    def fake_run_task_attempt(**kwargs: object) -> PairTaskAttempt:
        task = kwargs["task"]
        config = kwargs["config"]
        attempt_index = kwargs["attempt_index"]
        assert isinstance(task, fresh_pair.LiveDRBenchTask)
        assert isinstance(config, HarnessConfig)
        assert isinstance(attempt_index, int)
        calls.append((config.search.kind, task.key, attempt_index))
        failed_once = config.search.kind == "tavily" and task.key == 10 and attempt_index == 1
        return PairTaskAttempt(
            key=task.key,
            category=task.category,
            attempt_index=attempt_index,
            status="failed" if failed_once else "succeeded",
            error_type="RuntimeError" if failed_once else None,
        )

    monkeypatch.setattr(fresh_pair, "validate_fresh_public_dataset", lambda _: tasks)
    monkeypatch.setattr(fresh_pair, "_preflight_execution_keys", lambda *_: None)
    monkeypatch.setattr(fresh_pair, "_run_task_attempt", fake_run_task_attempt)
    monkeypatch.setattr(fresh_pair, "_validate_execution", lambda *_: None)
    monkeypatch.setattr(fresh_pair, "_validate_success_attempt", lambda *_: None)

    first = execute_fresh_public_pair(
        pair_manifest_path=tmp_path / "runs" / manifest.run_label / "pair_manifest.json"
    )
    assert first.status == "completed_with_failures"
    assert len(calls) == 10
    with pytest.raises(ValueError, match="resume_failed"):
        execute_fresh_public_pair(
            pair_manifest_path=tmp_path / "runs" / manifest.run_label / "pair_manifest.json"
        )

    second = execute_fresh_public_pair(
        pair_manifest_path=tmp_path / "runs" / manifest.run_label / "pair_manifest.json",
        resume_failed=True,
    )
    assert second.status == "succeeded"
    assert calls[-1] == ("tavily", 10, 2)
    assert len(calls) == 11


def test_executor_rejects_missing_tavily_key_before_fetching_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "common.json"
    _write_config(config_path, _base_config())
    manifest = prepare_fresh_public_pair(
        registration_path=REGISTRATION,
        base_config_path=config_path,
        output_dir=tmp_path / "runs",
        run_label="fresh-tavily-preflight",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        fresh_pair,
        "validate_fresh_public_dataset",
        lambda _: (_ for _ in ()).throw(AssertionError("dataset fetch must not occur")),
    )

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        execute_fresh_public_pair(
            pair_manifest_path=tmp_path / "runs" / manifest.run_label / "pair_manifest.json"
        )
