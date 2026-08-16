from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepresearch_harness import development_profile
from deepresearch_harness.development_profile import (
    DevelopmentProfileRegistration,
    preflight_development_profile,
)
from deepresearch_harness.tool_health import SearchServiceHealth


def _reference(path: str, marker: str = "a") -> dict[str, str]:
    return {"path": path, "sha256": marker * 64}


def _registration_payload() -> dict[str, object]:
    return {
        "schema_version": "browsecomp-plus-development-profile-registration-v0",
        "status": "registered_before_provider_call",
        "registered_at": "2026-08-17T00:00:00+08:00",
        "purpose": "Profile a frozen policy over all development questions.",
        "artifacts": {
            "failure_cluster_route": _reference("benchmarks/route.json"),
            "target_manifest": _reference("benchmarks/target.json", "b"),
            "query_partitions": {
                **_reference("benchmarks/partitions.json", "c"),
                "normalized_sha256": "d" * 64,
            },
            "development_queries": _reference("runs/queries.json", "e"),
            "retriever_manifest": _reference("benchmarks/retriever.json", "f"),
        },
        "policy": {
            "policy_id": "pi-v10-query-aware-progressive-disclosure-failclosed",
            "adapter_version": "pi-browsecomp-v10",
            "adapter_contract": _reference("integrations/contract.mjs", "1"),
            "adapter_runner": _reference("integrations/runner.mjs", "2"),
            "preview_module": _reference("src/preview.py", "3"),
            "retrieval_server": _reference("src/server.py", "4"),
            "operational_patch_only": True,
        },
        "execution": {
            "model": "deepseek-v4-flash",
            "thinking_level": "high",
            "system_prompt_policy": "empty",
            "control_policy": "answer_reserve_nonthinking_v0",
            "query_partition": "development",
            "query_count": 175,
            "concurrency": 1,
            "max_output_tokens_per_query": 10000,
            "max_iterations_per_query": 100,
            "maximum_search_calls_per_query": 8,
            "maximum_open_calls_per_query": 8,
            "max_search_results": 20,
            "search_url": "http://127.0.0.1:8768/search",
            "retriever_id": "retriever-v0",
            "provider_key_env": "DEEPSEEK_API_KEY",
            "failed_only_resume_limit": 2,
            "budget_exhausted_retry": "forbidden",
            "output_directory": "runs/profile/v10",
            "maximum_provider_cost_usd": 2.0,
        },
        "evaluation": {
            "judge_url": "http://127.0.0.1:18015/v1",
            "served_model_name": "qwen3-32b-bf16-judge",
            "metric_status": "calibrated_development_diagnostic_not_official",
            "sealed_holdout_access": "forbidden",
        },
        "acceptance": {
            "succeeded_must_equal": 175,
            "failed_must_equal": 0,
            "budget_exhausted_must_equal": 0,
            "output_budget_overshoot_tokens_must_equal": 0,
            "search_and_open_transport_failures_must_equal": 0,
            "minimum_schema_complete": 168,
            "judge_parse_failures_must_equal": 0,
            "judge_request_failures_must_equal": 0,
            "minimum_scored_wrong_cases_for_taxonomy": 30,
            "accuracy_promotion_threshold": "none_profile_only",
        },
        "planned_metrics": ["Judge accuracy", "Token", "cost"],
        "claim_boundary": "Development profile only.",
    }


def test_profile_registration_requires_96_percent_schema_gate() -> None:
    payload = _registration_payload()
    payload["acceptance"]["minimum_schema_complete"] = 167  # type: ignore[index]

    with pytest.raises(ValidationError, match="at least 96%"):
        DevelopmentProfileRegistration.model_validate(payload)


def test_preflight_records_no_provider_call_and_never_persists_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration = DevelopmentProfileRegistration.model_validate(
        _registration_payload()
    )
    query_path = tmp_path / "runs" / "queries.json"
    query_path.parent.mkdir(parents=True)
    query_path.write_text(
        json.dumps({"queries_sha256": "9" * 64}), encoding="utf-8"
    )
    registration_path = tmp_path / "benchmarks" / "registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        development_profile,
        "load_development_profile_registration",
        lambda _: registration,
    )
    monkeypatch.setattr(development_profile, "_repository_root", lambda _: tmp_path)

    def healthy(*_: object, **__: object) -> SearchServiceHealth:
        return SearchServiceHealth(
            health_url="http://127.0.0.1:8768/health",
            retriever_id="retriever-v0",
        )

    output = tmp_path / "runs" / "preflight.json"
    result = preflight_development_profile(
        registration_path=registration_path,
        output_path=output,
        environment={"DEEPSEEK_API_KEY": "secret-not-for-artifacts"},
        health_check=healthy,
    )

    assert result.provider_calls == 0
    assert result.provider_key_present is True
    assert "secret-not-for-artifacts" not in output.read_text(encoding="utf-8")


def test_preflight_fails_before_health_check_when_key_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration = DevelopmentProfileRegistration.model_validate(
        _registration_payload()
    )
    registration_path = tmp_path / "benchmarks" / "registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        development_profile,
        "load_development_profile_registration",
        lambda _: registration,
    )
    monkeypatch.setattr(development_profile, "_repository_root", lambda _: tmp_path)

    with pytest.raises(ValueError, match="environment variable is missing"):
        preflight_development_profile(
            registration_path=registration_path,
            output_path=tmp_path / "runs" / "preflight.json",
            environment={},
        )
