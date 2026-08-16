from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import load_pi_browsecomp_run
from .evidence_span_oracle import ArtifactReference, answer_coverage
from .obligation_span_fresh import ObligationSpanFreshDecision


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObligationOpenAuditRegistration(StrictContract):
    schema_version: Literal["obligation-open-audit-registration-v0"] = (
        "obligation-open-audit-registration-v0"
    )
    status: Literal["posthoc_registered_after_fresh_decision"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    paired_decision: ArtifactReference
    development_gold: ArtifactReference
    baseline_run_root: str = Field(min_length=1)
    candidate_run_root: str = Field(min_length=1)
    query_ids: tuple[str, ...] = Field(min_length=1)
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    provider_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def query_ids_are_unique(self) -> "ObligationOpenAuditRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("audit query IDs must be unique")
        return self


MechanismEffect = Literal[
    "supported_gain_from_answer_bearing_gold_open",
    "gain_without_answer_bearing_gold_open",
    "regression_without_successful_open",
    "regression_after_non_answer_bearing_open",
    "stable_with_answer_bearing_gold_open",
    "stable_other",
]


class ObligationOpenAuditCase(StrictContract):
    query_id: str = Field(min_length=1)
    paired_outcome: Literal["both_correct", "both_wrong", "improvement", "regression"]
    baseline_search_calls: int = Field(ge=0)
    candidate_search_calls: int = Field(ge=0)
    search_call_delta: int
    candidate_open_attempts: int = Field(ge=0)
    candidate_successful_opens: int = Field(ge=0)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    baseline_retrieved_gold_docids: tuple[str, ...]
    candidate_retrieved_gold_docids: tuple[str, ...]
    candidate_opened_gold_docids: tuple[str, ...]
    answer_bearing_gold_open: bool
    opened_answer_coverage: float = Field(ge=0, le=1)
    mechanism_effect: MechanismEffect


class ObligationOpenAuditResult(StrictContract):
    schema_version: Literal["obligation-open-audit-v0"] = "obligation-open-audit-v0"
    created_at: str = Field(min_length=1)
    status: Literal["posthoc_diagnostic_not_effectiveness_result"] = (
        "posthoc_diagnostic_not_effectiveness_result"
    )
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(gt=0)
    improvement_cases: int = Field(ge=0)
    regression_cases: int = Field(ge=0)
    successful_open_cases: int = Field(ge=0)
    successful_open_calls: int = Field(ge=0)
    opened_gold_cases: int = Field(ge=0)
    answer_bearing_gold_open_cases: int = Field(ge=0)
    supported_improvement_cases: int = Field(ge=0)
    improvements_without_answer_bearing_gold_open: int = Field(ge=0)
    regressions_with_successful_open: int = Field(ge=0)
    regressions_without_successful_open: int = Field(ge=0)
    total_search_call_delta: int
    provider_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    cases: tuple[ObligationOpenAuditCase, ...] = Field(min_length=1)
    diagnosis: Literal[
        "mechanism_signal_confounded_requires_no_change_repeats",
        "opening_itself_is_linked_to_regressions",
        "no_supported_span_opening_signal",
    ]
    next_action: Literal[
        "preregister_no_change_paired_repeats",
        "freeze_opening_and_diagnose_selectivity",
        "freeze_opening_and_return_to_retrieval",
    ]
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def aggregates_match_cases(self) -> "ObligationOpenAuditResult":
        expected = {
            "query_count": len(self.cases),
            "improvement_cases": sum(
                case.paired_outcome == "improvement" for case in self.cases
            ),
            "regression_cases": sum(
                case.paired_outcome == "regression" for case in self.cases
            ),
            "successful_open_cases": sum(
                case.candidate_successful_opens > 0 for case in self.cases
            ),
            "successful_open_calls": sum(
                case.candidate_successful_opens for case in self.cases
            ),
            "opened_gold_cases": sum(
                bool(case.candidate_opened_gold_docids) for case in self.cases
            ),
            "answer_bearing_gold_open_cases": sum(
                case.answer_bearing_gold_open for case in self.cases
            ),
            "supported_improvement_cases": sum(
                case.mechanism_effect
                == "supported_gain_from_answer_bearing_gold_open"
                for case in self.cases
            ),
            "improvements_without_answer_bearing_gold_open": sum(
                case.mechanism_effect == "gain_without_answer_bearing_gold_open"
                for case in self.cases
            ),
            "regressions_with_successful_open": sum(
                case.paired_outcome == "regression"
                and case.candidate_successful_opens > 0
                for case in self.cases
            ),
            "regressions_without_successful_open": sum(
                case.mechanism_effect == "regression_without_successful_open"
                for case in self.cases
            ),
            "total_search_call_delta": sum(
                case.search_call_delta for case in self.cases
            ),
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"{field_name} does not match audit cases")
        return self


def classify_mechanism_effect(
    *,
    paired_outcome: str,
    successful_opens: int,
    answer_bearing_gold_open: bool,
) -> MechanismEffect:
    if answer_bearing_gold_open and successful_opens == 0:
        raise ValueError("answer-bearing open requires a successful open")
    if paired_outcome == "improvement":
        return (
            "supported_gain_from_answer_bearing_gold_open"
            if answer_bearing_gold_open
            else "gain_without_answer_bearing_gold_open"
        )
    if paired_outcome == "regression":
        return (
            "regression_without_successful_open"
            if successful_opens == 0
            else "regression_after_non_answer_bearing_open"
        )
    if answer_bearing_gold_open:
        return "stable_with_answer_bearing_gold_open"
    return "stable_other"


def load_obligation_open_audit_registration(
    path: Path,
) -> ObligationOpenAuditRegistration:
    registration = ObligationOpenAuditRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    for artifact in (
        registration.paired_decision,
        registration.development_gold,
        *registration.frozen_artifacts,
    ):
        artifact_path = (root / artifact.path).resolve()
        if not artifact_path.is_relative_to(root) or not artifact_path.is_file():
            raise ValueError(f"audit artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"audit artifact hash changed: {artifact.path}")
    for run_root in (registration.baseline_run_root, registration.candidate_run_root):
        resolved = (root / run_root).resolve()
        if not resolved.is_relative_to((root / "runs").resolve()) or not resolved.is_dir():
            raise ValueError("audit run root is missing or escapes ignored runs/")
    return registration


def run_obligation_open_audit(
    *, registration_path: Path, output_path: Path
) -> ObligationOpenAuditResult:
    registration = load_obligation_open_audit_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("audit output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("audit output already exists")
    decision = ObligationSpanFreshDecision.model_validate_json(
        (root / registration.paired_decision.path).read_text(encoding="utf-8")
    )
    if tuple(case.query_id for case in decision.cases) != registration.query_ids:
        raise ValueError("audit query order differs from paired decision")
    gold_payload = json.loads(
        (root / registration.development_gold.path).read_text(encoding="utf-8")
    )
    gold_by_id = {str(row["query_id"]): row for row in gold_payload["rows"]}
    cases: list[ObligationOpenAuditCase] = []
    for paired in decision.cases:
        query_id = paired.query_id
        if query_id not in gold_by_id:
            raise ValueError(f"audit gold row is missing: {query_id}")
        gold = gold_by_id[query_id]
        gold_docids = tuple(str(docid) for docid in gold["gold_docids"])
        baseline_run = load_pi_browsecomp_run(
            root / registration.baseline_run_root / query_id / "run.json"
        )
        candidate_run = load_pi_browsecomp_run(
            root / registration.candidate_run_root / query_id / "run.json"
        )
        baseline_retrieved = _retrieved_gold_docids(baseline_run, gold_docids)
        candidate_retrieved = _retrieved_gold_docids(candidate_run, gold_docids)
        successful = tuple(
            call
            for call in candidate_run.evidence_open_calls
            if call.outcome == "ok"
            and call.result is not None
            and call.result.outcome == "opened"
        )
        opened_gold = tuple(
            dict.fromkeys(call.docid for call in successful if call.docid in gold_docids)
        )
        opened_gold_text = "\n".join(
            call.result.content
            for call in successful
            if call.docid in gold_docids
            and call.result is not None
            and call.result.content is not None
        )
        coverage = answer_coverage(str(gold["answer"]), opened_gold_text)
        effect = classify_mechanism_effect(
            paired_outcome=paired.paired_outcome,
            successful_opens=len(successful),
            answer_bearing_gold_open=coverage.all_atoms_present,
        )
        cases.append(
            ObligationOpenAuditCase(
                query_id=query_id,
                paired_outcome=paired.paired_outcome,
                baseline_search_calls=len(baseline_run.search_calls),
                candidate_search_calls=len(candidate_run.search_calls),
                search_call_delta=(
                    len(candidate_run.search_calls) - len(baseline_run.search_calls)
                ),
                candidate_open_attempts=len(candidate_run.evidence_open_calls),
                candidate_successful_opens=len(successful),
                gold_docids=gold_docids,
                baseline_retrieved_gold_docids=baseline_retrieved,
                candidate_retrieved_gold_docids=candidate_retrieved,
                candidate_opened_gold_docids=opened_gold,
                answer_bearing_gold_open=coverage.all_atoms_present,
                opened_answer_coverage=coverage.coverage,
                mechanism_effect=effect,
            )
        )
    supported = sum(
        case.mechanism_effect == "supported_gain_from_answer_bearing_gold_open"
        for case in cases
    )
    regression_opens = sum(
        case.paired_outcome == "regression" and case.candidate_successful_opens > 0
        for case in cases
    )
    if supported and regression_opens == 0:
        diagnosis = "mechanism_signal_confounded_requires_no_change_repeats"
        next_action = "preregister_no_change_paired_repeats"
    elif regression_opens:
        diagnosis = "opening_itself_is_linked_to_regressions"
        next_action = "freeze_opening_and_diagnose_selectivity"
    else:
        diagnosis = "no_supported_span_opening_signal"
        next_action = "freeze_opening_and_return_to_retrieval"
    result = ObligationOpenAuditResult(
        created_at=_utc_now(),
        registration_sha256=_sha256_file(registration_path),
        query_count=len(cases),
        improvement_cases=sum(case.paired_outcome == "improvement" for case in cases),
        regression_cases=sum(case.paired_outcome == "regression" for case in cases),
        successful_open_cases=sum(case.candidate_successful_opens > 0 for case in cases),
        successful_open_calls=sum(case.candidate_successful_opens for case in cases),
        opened_gold_cases=sum(bool(case.candidate_opened_gold_docids) for case in cases),
        answer_bearing_gold_open_cases=sum(case.answer_bearing_gold_open for case in cases),
        supported_improvement_cases=supported,
        improvements_without_answer_bearing_gold_open=sum(
            case.mechanism_effect == "gain_without_answer_bearing_gold_open"
            for case in cases
        ),
        regressions_with_successful_open=regression_opens,
        regressions_without_successful_open=sum(
            case.mechanism_effect == "regression_without_successful_open"
            for case in cases
        ),
        total_search_call_delta=sum(case.search_call_delta for case in cases),
        cases=tuple(cases),
        diagnosis=diagnosis,
        next_action=next_action,
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _retrieved_gold_docids(run: object, gold_docids: tuple[str, ...]) -> tuple[str, ...]:
    retrieved = {
        result.docid
        for call in run.search_calls
        for result in call.results
    }
    return tuple(docid for docid in gold_docids if docid in retrieved)
