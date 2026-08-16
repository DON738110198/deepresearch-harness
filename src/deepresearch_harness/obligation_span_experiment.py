from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_span_oracle import ArtifactReference, answer_coverage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObligationSpanCalibrationAcceptance(StrictContract):
    minimum_opened_span_cases: int = Field(ge=1)
    minimum_answer_bearing_span_cases: int = Field(ge=1)
    minimum_judge_correct_cases: int = Field(ge=1)
    minimum_judge_correct_delta: int = Field(ge=1)
    schema_complete_must_equal: int = Field(ge=1)
    generation_failures_must_equal: Literal[0] = 0
    judge_parse_failures_must_equal: Literal[0] = 0
    judge_request_failures_must_equal: Literal[0] = 0
    maximum_provider_cost_usd: float = Field(gt=0)


class ObligationSpanCalibrationRegistration(StrictContract):
    schema_version: Literal["obligation-span-calibration-registration-v0"] = (
        "obligation-span-calibration-registration-v0"
    )
    status: Literal["registered_pre_provider", "registered_pre_decision_no_provider"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    query_artifact: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    oracle_result: ArtifactReference
    baseline_summary: ArtifactReference
    baseline_judge: ArtifactReference
    fixed_contract: dict[str, object]
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: ObligationSpanCalibrationAcceptance
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def gate_matches_cases(self) -> "ObligationSpanCalibrationRegistration":
        count = len(self.query_ids)
        if count != len(set(self.query_ids)):
            raise ValueError("calibration query IDs must be unique")
        for value in (
            self.acceptance.minimum_opened_span_cases,
            self.acceptance.minimum_answer_bearing_span_cases,
            self.acceptance.minimum_judge_correct_cases,
            self.acceptance.schema_complete_must_equal,
        ):
            if value > count:
                raise ValueError("calibration gate exceeds registered case count")
        return self


class DecisionGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class ObligationSpanCaseDecision(StrictContract):
    query_id: str = Field(min_length=1)
    opened_spans: int = Field(ge=0)
    opened_docids: tuple[str, ...]
    answer_bearing_span: bool
    baseline_judge_correct: bool
    candidate_judge_correct: bool


class ObligationSpanCalibrationDecision(StrictContract):
    schema_version: Literal["obligation-span-calibration-decision-v0"] = (
        "obligation-span-calibration-decision-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded", "failed"]
    decision: Literal["advance_to_fresh_slice", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    opened_span_cases: int = Field(ge=0)
    answer_bearing_span_cases: int = Field(ge=0)
    baseline_judge_correct_cases: int = Field(ge=0)
    candidate_judge_correct_cases: int = Field(ge=0)
    judge_correct_delta: int
    schema_complete: int = Field(ge=0)
    generation_failures: int = Field(ge=0)
    judge_parse_failures: int = Field(ge=0)
    judge_request_failures: int = Field(ge=0)
    provider_cost_usd: float = Field(ge=0)
    items: tuple[ObligationSpanCaseDecision, ...]
    gates: tuple[DecisionGate, ...]
    claim_boundary: str = Field(min_length=1)


def load_obligation_span_registration(
    path: Path,
) -> ObligationSpanCalibrationRegistration:
    registration = ObligationSpanCalibrationRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    for artifact in (
        registration.query_artifact,
        registration.oracle_result,
        registration.baseline_summary,
        registration.baseline_judge,
        *registration.frozen_artifacts,
    ):
        artifact_path = (root / artifact.path).resolve()
        if not artifact_path.is_relative_to(root) or not artifact_path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")
    return registration


def decide_obligation_span_calibration(
    *,
    registration_path: Path,
    candidate_summary_path: Path,
    candidate_judge_path: Path,
    output_path: Path,
) -> ObligationSpanCalibrationDecision:
    registration = load_obligation_span_registration(registration_path)
    root = registration_path.resolve().parents[2]
    for path in (candidate_summary_path, candidate_judge_path, output_path.parent):
        if not path.resolve().is_relative_to((root / "runs").resolve()):
            raise ValueError("calibration runtime artifacts must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("calibration decision output already exists")

    baseline_judge = json.loads(
        (root / registration.baseline_judge.path).read_text(encoding="utf-8")
    )
    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    candidate_judge = json.loads(candidate_judge_path.read_text(encoding="utf-8"))
    baseline_by_id = {
        str(row["query_id"]): bool(row["correct"])
        for row in baseline_judge["observations"]
    }
    candidate_by_id = {
        str(row["query_id"]): bool(row["correct"])
        for row in candidate_judge["observations"]
    }
    summary_by_id = {
        str(row["query_id"]): row for row in candidate_summary["items"]
    }

    items: list[ObligationSpanCaseDecision] = []
    for query_id in registration.query_ids:
        if query_id not in baseline_by_id or query_id not in candidate_by_id:
            raise ValueError(f"Judge result is missing registered case {query_id}")
        summary_row = summary_by_id.get(query_id)
        if summary_row is None:
            raise ValueError(f"candidate summary is missing registered case {query_id}")
        recorded_run_path = Path(str(summary_row["run_path"]))
        run_path = (
            root / recorded_run_path
            if recorded_run_path.parts
            and recorded_run_path.parts[0].casefold() == "runs"
            else candidate_summary_path.parent / recorded_run_path
        )
        if not run_path.resolve().is_relative_to(candidate_summary_path.parent.resolve()):
            raise ValueError(f"candidate run path escapes its summary: {query_id}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        evaluation_path = candidate_judge_path.parent / "results" / f"{query_id}_eval.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        correct_answer = str(evaluation["correct_answer"])
        opened = [
            call
            for call in run.get("evidence_open_calls", [])
            if call.get("outcome") == "ok"
            and isinstance(call.get("result"), dict)
            and call["result"].get("outcome") == "opened"
        ]
        opened_content = "\n".join(
            str(call["result"].get("content") or "") for call in opened
        )
        items.append(
            ObligationSpanCaseDecision(
                query_id=query_id,
                opened_spans=len(opened),
                opened_docids=tuple(str(call["docid"]) for call in opened),
                answer_bearing_span=(
                    answer_coverage(correct_answer, opened_content).all_atoms_present
                    if opened
                    else False
                ),
                baseline_judge_correct=baseline_by_id[query_id],
                candidate_judge_correct=candidate_by_id[query_id],
            )
        )

    opened_span_cases = sum(item.opened_spans > 0 for item in items)
    answer_bearing_cases = sum(item.answer_bearing_span for item in items)
    baseline_correct = sum(item.baseline_judge_correct for item in items)
    candidate_correct = sum(item.candidate_judge_correct for item in items)
    delta = candidate_correct - baseline_correct
    acceptance = registration.acceptance
    schema_complete = int(candidate_summary["schema_complete"])
    generation_failures = int(candidate_summary["failed"]) + int(
        candidate_summary["budget_exhausted"]
    )
    parse_failures = int(candidate_judge["parse_failures"])
    request_failures = int(candidate_judge["request_failures"])
    provider_cost = float(candidate_summary["total_cost_usd"])
    gates = (
        _gate("opened_span_cases", opened_span_cases, "ge", acceptance.minimum_opened_span_cases),
        _gate("answer_bearing_span_cases", answer_bearing_cases, "ge", acceptance.minimum_answer_bearing_span_cases),
        _gate("candidate_judge_correct_cases", candidate_correct, "ge", acceptance.minimum_judge_correct_cases),
        _gate("judge_correct_delta", delta, "ge", acceptance.minimum_judge_correct_delta),
        _gate("schema_complete", schema_complete, "eq", acceptance.schema_complete_must_equal),
        _gate("generation_failures", generation_failures, "eq", acceptance.generation_failures_must_equal),
        _gate("judge_parse_failures", parse_failures, "eq", acceptance.judge_parse_failures_must_equal),
        _gate("judge_request_failures", request_failures, "eq", acceptance.judge_request_failures_must_equal),
        _gate("provider_cost_usd", provider_cost, "le", acceptance.maximum_provider_cost_usd),
    )
    passed = all(gate.passed for gate in gates)
    result = ObligationSpanCalibrationDecision(
        created_at=_utc_now(),
        status="succeeded" if generation_failures == 0 and request_failures == 0 else "failed",
        decision="advance_to_fresh_slice" if passed else "reject",
        registration_sha256=_sha256_file(registration_path),
        query_count=len(items),
        opened_span_cases=opened_span_cases,
        answer_bearing_span_cases=answer_bearing_cases,
        baseline_judge_correct_cases=baseline_correct,
        candidate_judge_correct_cases=candidate_correct,
        judge_correct_delta=delta,
        schema_complete=schema_complete,
        generation_failures=generation_failures,
        judge_parse_failures=parse_failures,
        judge_request_failures=request_failures,
        provider_cost_usd=provider_cost,
        items=tuple(items),
        gates=gates,
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _gate(
    gate_id: str,
    observed: int | float,
    operator: Literal["eq", "ge", "le"],
    threshold: int | float,
) -> DecisionGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return DecisionGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )
