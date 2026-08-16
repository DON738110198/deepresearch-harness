from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_span_oracle import ArtifactReference
from .tool_health import SearchServiceUnavailable, require_search_service_health


BASELINE_RETRIEVER_ID = (
    "bm25-anchor-5+qwen3-embedding-0.6b-lead-15-query_window_v0-64"
)
CANDIDATE_RETRIEVER_ID = (
    f"{BASELINE_RETRIEVER_ID}-anchor-reopen-answer_obligation_window_v0"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolHealthValidationCheck(StrictContract):
    check_id: str = Field(min_length=1)
    passed: bool
    detail: str = Field(min_length=1)


class LiveSearchServiceSnapshot(StrictContract):
    port: int = Field(ge=1, le=65535)
    health_url: str = Field(min_length=1)
    retriever_id: str = Field(min_length=1)


class ToolHealthBoundaryValidation(StrictContract):
    schema_version: Literal["tool-health-boundary-validation-v0"] = (
        "tool-health-boundary-validation-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["offline_contract_and_live_service_smoke"] = (
        "offline_contract_and_live_service_smoke"
    )
    decision: Literal["accept", "reject"]
    execution_incident: ArtifactReference
    source_sha256: dict[str, str]
    checks: tuple[ToolHealthValidationCheck, ...] = Field(min_length=1)
    live_services: tuple[LiveSearchServiceSnapshot, LiveSearchServiceSnapshot]
    provider_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    next_action: Literal[
        "reregister_clean_effectiveness_repeats",
        "keep_effectiveness_runs_blocked",
    ]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_matches_checks(self) -> "ToolHealthBoundaryValidation":
        expected = "accept" if all(check.passed for check in self.checks) else "reject"
        if self.decision != expected:
            raise ValueError("tool-health decision differs from checks")
        expected_next = (
            "reregister_clean_effectiveness_repeats"
            if self.decision == "accept"
            else "keep_effectiveness_runs_blocked"
        )
        if self.next_action != expected_next:
            raise ValueError("tool-health next action differs from decision")
        return self


def validate_tool_health_boundary(
    *, root: Path, node_executable: Path, output_path: Path
) -> ToolHealthBoundaryValidation:
    root = root.resolve()
    output_path = output_path.resolve()
    if not output_path.is_relative_to((root / "runs").resolve()):
        raise ValueError("tool-health validation output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("tool-health validation output already exists")
    node_executable = node_executable.resolve()
    if not node_executable.is_file():
        raise ValueError("Node executable is missing")

    checks = [
        _run_check(
            check_id="python_fail_closed_tests",
            command=[
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_tool_health.py",
                "tests/test_browsecomp_plus.py",
            ],
            root=root,
        ),
        _run_check(
            check_id="node_tool_failure_contract_tests",
            command=[
                str(node_executable),
                "--test",
                "integrations/pi-browsecomp/v10/contract.test.mjs",
                "integrations/pi-browsecomp/v14/contract.test.mjs",
            ],
            root=root,
        ),
    ]

    services: list[LiveSearchServiceSnapshot] = []
    for port, retriever_id in (
        (8768, BASELINE_RETRIEVER_ID),
        (8769, CANDIDATE_RETRIEVER_ID),
    ):
        health = require_search_service_health(
            f"http://127.0.0.1:{port}/search",
            expected_retriever_id=retriever_id,
        )
        services.append(
            LiveSearchServiceSnapshot(
                port=port,
                health_url=health.health_url,
                retriever_id=health.retriever_id,
            )
        )
    checks.append(
        ToolHealthValidationCheck(
            check_id="live_retriever_identity_binding",
            passed=True,
            detail="8768 and 8769 returned the exact frozen retriever IDs",
        )
    )

    try:
        require_search_service_health(
            "http://127.0.0.1:8769/search",
            expected_retriever_id="intentionally-wrong-retriever",
        )
    except SearchServiceUnavailable as error:
        checks.append(
            ToolHealthValidationCheck(
                check_id="retriever_drift_fails_closed",
                passed="retriever_id_mismatch" in str(error),
                detail=str(error),
            )
        )
    else:
        checks.append(
            ToolHealthValidationCheck(
                check_id="retriever_drift_fails_closed",
                passed=False,
                detail="wrong retriever identity was accepted",
            )
        )

    run_id = f"tool-health-validation-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    search = _post_json(
        "http://127.0.0.1:8769/search",
        {"run_id": run_id, "query": "deep research evidence citation validation"},
    )
    results = search.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("candidate live search returned no results")
    docid = results[0].get("docid")
    if not isinstance(docid, str) or not docid:
        raise ValueError("candidate live search result has no docid")
    opened = _post_json(
        "http://127.0.0.1:8769/open",
        {
            "run_id": run_id,
            "docid": docid,
            "obligation_query": "What exact evidence supports the answer?",
        },
    )
    open_result = opened.get("result")
    open_passed = (
        isinstance(open_result, dict)
        and open_result.get("outcome") == "opened"
        and isinstance(open_result.get("content"), str)
        and bool(open_result["content"])
    )
    checks.append(
        ToolHealthValidationCheck(
            check_id="candidate_search_and_obligation_open_smoke",
            passed=open_passed,
            detail=(
                f"search_results={len(results)};docid={docid};"
                f"open_outcome={open_result.get('outcome') if isinstance(open_result, dict) else None}"
            ),
        )
    )

    source_paths = (
        "src/deepresearch_harness/tool_health.py",
        "src/deepresearch_harness/pi_browsecomp.py",
        "integrations/pi-browsecomp/v10/contract.mjs",
        "integrations/pi-browsecomp/v10/runner.mjs",
        "integrations/pi-browsecomp/v14/contract.mjs",
        "integrations/pi-browsecomp/v14/runner.mjs",
        "tests/test_tool_health.py",
    )
    source_hashes = {path: _sha256_file(root / path) for path in source_paths}
    incident_path = (
        root
        / "runs/browsecomp_plus_v0/obligation-span-repeats-v0-20260816/execution-audit.json"
    )
    incident = ArtifactReference(
        path=incident_path.relative_to(root).as_posix(),
        sha256=_sha256_file(incident_path),
    )
    decision: Literal["accept", "reject"] = (
        "accept" if all(check.passed for check in checks) else "reject"
    )
    result = ToolHealthBoundaryValidation(
        created_at=_utc_now(),
        decision=decision,
        execution_incident=incident,
        source_sha256=source_hashes,
        checks=tuple(checks),
        live_services=(services[0], services[1]),
        next_action=(
            "reregister_clean_effectiveness_repeats"
            if decision == "accept"
            else "keep_effectiveness_runs_blocked"
        ),
        claim_boundary=(
            "This validates execution integrity only: search dependencies are identity-bound, "
            "a failed preflight cannot launch the provider subprocess, transport-level tool "
            "errors fail the Node run, and the candidate search/open service is live. It makes "
            "no provider or Judge call, accesses no sealed holdout, and is not an effectiveness "
            "or model-capability result."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _run_check(
    *, check_id: str, command: list[str], root: Path
) -> ToolHealthValidationCheck:
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    combined = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    detail = combined[-2000:] if combined else f"exit_code={completed.returncode}"
    return ToolHealthValidationCheck(
        check_id=check_id,
        passed=completed.returncode == 0,
        detail=detail,
    )


def _post_json(url: str, payload: dict[str, str]) -> dict[str, object]:
    encoded = json.dumps(payload).encode("utf-8")
    with urlopen(
        Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        ),
        timeout=30,
    ) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("tool service returned a non-object response")
    return value
