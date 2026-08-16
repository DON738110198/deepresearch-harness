from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch_harness import pi_browsecomp, tool_health
from deepresearch_harness.pi_browsecomp import _run_adapter_after_search_preflight
from deepresearch_harness.tool_health import (
    SearchServiceUnavailable,
    health_url_for_search,
    require_search_service_health,
)
from deepresearch_harness.tool_health_validation import ToolHealthBoundaryValidation


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_health_url_is_derived_without_changing_authority() -> None:
    assert (
        health_url_for_search("http://127.0.0.1:8769/search")
        == "http://127.0.0.1:8769/health"
    )


def test_health_probe_binds_retriever_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tool_health,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            {"status": "ok", "retriever_id": "candidate"}
        ),
    )
    health = require_search_service_health(
        "http://127.0.0.1:8769/search",
        expected_retriever_id="candidate",
    )
    assert health.retriever_id == "candidate"


def test_health_probe_fails_closed_on_retriever_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_health,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            {"status": "ok", "retriever_id": "wrong"}
        ),
    )
    with pytest.raises(SearchServiceUnavailable, match="retriever_id_mismatch"):
        require_search_service_health(
            "http://127.0.0.1:8769/search",
            expected_retriever_id="candidate",
        )


def test_provider_subprocess_is_not_started_when_search_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_started = False

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise SearchServiceUnavailable("candidate service is down")

    def provider_call(*_args: object, **_kwargs: object) -> None:
        nonlocal provider_started
        provider_started = True

    monkeypatch.setattr(pi_browsecomp, "require_search_service_health", unavailable)
    monkeypatch.setattr(pi_browsecomp.subprocess, "run", provider_call)

    with pytest.raises(SearchServiceUnavailable, match="service is down"):
        _run_adapter_after_search_preflight(
            search_url="http://127.0.0.1:8769/search",
            retriever_id="candidate",
            command=["node", "runner.mjs", "request.json"],
            adapter_dir=Path("adapter"),
            timeout_seconds=1,
        )
    assert provider_started is False


def test_validation_result_cannot_accept_a_failed_check() -> None:
    payload = {
        "created_at": "2026-08-16T00:00:00Z",
        "decision": "accept",
        "execution_incident": {"path": "runs/incident.json", "sha256": "a" * 64},
        "source_sha256": {"source": "b" * 64},
        "checks": [{"check_id": "fixture", "passed": False, "detail": "failed"}],
        "live_services": [
            {"port": 8768, "health_url": "http://127.0.0.1:8768/health", "retriever_id": "baseline"},
            {"port": 8769, "health_url": "http://127.0.0.1:8769/health", "retriever_id": "candidate"},
        ],
        "next_action": "reregister_clean_effectiveness_repeats",
        "claim_boundary": "integrity only",
    }
    with pytest.raises(ValueError, match="decision differs"):
        ToolHealthBoundaryValidation.model_validate(payload)
