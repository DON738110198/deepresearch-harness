from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_debt_audit import DebtReason, EvidenceDebtAudit, audit_saved_trace


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalibrationSource(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationRow(StrictContract):
    query_id: str = Field(min_length=1)
    paired_outcome: Literal["candidate_regression", "candidate_improvement"]
    expected_repair_trigger: bool
    observed_status: Literal["supported", "no_repair_trigger", "open", "unscorable"]
    observed_repair_trigger: bool
    matched_expectation: bool
    reasons: tuple[DebtReason, ...]
    repair_queries: tuple[str, ...] = Field(max_length=2)
    provider_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    source: CalibrationSource


class CalibrationGate(StrictContract):
    gate_id: Literal[
        "regression_trigger_recall",
        "improvement_false_trigger_count",
        "provider_calls",
        "search_calls",
        "maximum_repair_queries",
    ]
    observed: float | int
    operator: Literal["eq", "le"]
    threshold: float | int
    passed: bool


class EvidenceDebtAuditCalibration(StrictContract):
    schema_version: Literal["browsecomp-plus-evidence-debt-audit-calibration-v0"] = (
        "browsecomp-plus-evidence-debt-audit-calibration-v0"
    )
    created_at: str
    status: Literal["outcome_selected_calibration_not_effectiveness_evidence"] = (
        "outcome_selected_calibration_not_effectiveness_evidence"
    )
    decision: Literal["pass", "reject"]
    regression_count: int = Field(gt=0)
    regression_trigger_count: int = Field(ge=0)
    regression_trigger_recall: float = Field(ge=0, le=1)
    improvement_count: int = Field(gt=0)
    improvement_false_trigger_count: int = Field(ge=0)
    provider_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    maximum_repair_queries: int = Field(ge=0, le=2)
    rows: list[CalibrationRow] = Field(min_length=1)
    gates: list[CalibrationGate] = Field(min_length=5, max_length=5)
    sources: dict[str, CalibrationSource]
    next_action: Literal[
        "implement_typed_pi_v11_checkpoint",
        "revise_or_abandon_trigger_before_live_calls",
    ]
    claim_boundary: Literal[
        "Known-outcome saved-trace calibration only; no benchmark effectiveness claim."
    ] = "Known-outcome saved-trace calibration only; no benchmark effectiveness claim."

    @model_validator(mode="after")
    def decision_matches_gates(self) -> "EvidenceDebtAuditCalibration":
        expected = "pass" if all(gate.passed for gate in self.gates) else "reject"
        if self.decision != expected:
            raise ValueError("calibration decision differs from its gates")
        expected_action = (
            "implement_typed_pi_v11_checkpoint"
            if self.decision == "pass"
            else "revise_or_abandon_trigger_before_live_calls"
        )
        if self.next_action != expected_action:
            raise ValueError("calibration next action differs from its decision")
        if self.regression_count + self.improvement_count != len(self.rows):
            raise ValueError("calibration row counts differ from outcome counts")
        return self


def calibrate_evidence_debt_audit(
    *,
    registration_path: Path,
    output_path: Path,
    validate_existing: bool = False,
) -> EvidenceDebtAuditCalibration:
    existing_created_at: str | None = None
    if validate_existing:
        existing_created_at = EvidenceDebtAuditCalibration.model_validate_json(
            output_path.read_text(encoding="utf-8")
        ).created_at
    registration = _load_object(registration_path)
    root = _repository_root(registration_path)
    implementation = registration["implementation"]
    module_path = root / str(implementation["audit_module_path"])
    _require_hash(module_path, str(implementation["audit_module_sha256"]))

    rows: list[CalibrationRow] = []
    audits: list[EvidenceDebtAudit] = []
    for raw_case in registration["cases"]:
        case = dict(raw_case)
        run_path = root / str(case["run_path"])
        _require_hash(run_path, str(case["run_sha256"]))
        audit = audit_saved_trace(run_path)
        if audit.query_id != str(case["query_id"]):
            raise ValueError("registered query id differs from saved trace")
        observed_trigger = audit.status == "open"
        expected_trigger = bool(case["expected_repair_trigger"])
        audits.append(audit)
        rows.append(
            CalibrationRow(
                query_id=audit.query_id,
                paired_outcome=case["paired_outcome"],
                expected_repair_trigger=expected_trigger,
                observed_status=audit.status,
                observed_repair_trigger=observed_trigger,
                matched_expectation=observed_trigger == expected_trigger,
                reasons=audit.reasons,
                repair_queries=audit.repair_queries,
                provider_calls=audit.provider_calls,
                search_calls=audit.search_calls,
                source=CalibrationSource(
                    path=str(case["run_path"]),
                    sha256=str(case["run_sha256"]),
                ),
            )
        )

    regression_rows = [
        row for row in rows if row.paired_outcome == "candidate_regression"
    ]
    improvement_rows = [
        row for row in rows if row.paired_outcome == "candidate_improvement"
    ]
    regression_trigger_count = sum(row.observed_repair_trigger for row in regression_rows)
    regression_recall = regression_trigger_count / len(regression_rows)
    false_triggers = sum(row.observed_repair_trigger for row in improvement_rows)
    provider_calls = sum(audit.provider_calls for audit in audits)
    search_calls = sum(audit.search_calls for audit in audits)
    maximum_repair_queries = max(len(audit.repair_queries) for audit in audits)

    acceptance = registration["acceptance"]
    gates = [
        _gate(
            "regression_trigger_recall",
            regression_recall,
            "eq",
            float(acceptance["regression_trigger_recall_must_equal"]),
        ),
        _gate(
            "improvement_false_trigger_count",
            false_triggers,
            "eq",
            int(acceptance["improvement_false_trigger_count_must_equal"]),
        ),
        _gate(
            "provider_calls",
            provider_calls,
            "eq",
            int(acceptance["provider_calls_must_equal"]),
        ),
        _gate(
            "search_calls",
            search_calls,
            "eq",
            int(acceptance["search_calls_must_equal"]),
        ),
        _gate(
            "maximum_repair_queries",
            maximum_repair_queries,
            "le",
            int(acceptance["maximum_repair_queries_per_open_case"]),
        ),
    ]
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = EvidenceDebtAuditCalibration(
        created_at=existing_created_at or datetime.now(timezone.utc).isoformat(),
        decision=decision,
        regression_count=len(regression_rows),
        regression_trigger_count=regression_trigger_count,
        regression_trigger_recall=regression_recall,
        improvement_count=len(improvement_rows),
        improvement_false_trigger_count=false_triggers,
        provider_calls=provider_calls,
        search_calls=search_calls,
        maximum_repair_queries=maximum_repair_queries,
        rows=rows,
        gates=gates,
        sources={
            "registration": CalibrationSource(
                path=str(registration_path.relative_to(root)).replace("\\", "/"),
                sha256=_file_sha256(registration_path),
            ),
            "audit_module": CalibrationSource(
                path=str(module_path.relative_to(root)).replace("\\", "/"),
                sha256=_file_sha256(module_path),
            ),
        },
        next_action=(
            "implement_typed_pi_v11_checkpoint"
            if decision == "pass"
            else "revise_or_abandon_trigger_before_live_calls"
        ),
    )
    serialized = result.model_dump_json(indent=2) + "\n"
    if validate_existing:
        if output_path.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing calibration artifact differs from recomputation")
    else:
        _atomic_write(output_path, serialized)
    return result


def _gate(
    gate_id: str,
    observed: float | int,
    operator: Literal["eq", "le"],
    threshold: float | int,
) -> CalibrationGate:
    passed = observed == threshold if operator == "eq" else observed <= threshold
    return CalibrationGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _repository_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError("could not locate repository root")


def _require_hash(path: Path, expected: str) -> None:
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(f"hash mismatch for {path}: expected {expected}, observed {observed}")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
