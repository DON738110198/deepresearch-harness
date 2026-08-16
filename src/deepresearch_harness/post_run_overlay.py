from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch_harness.browsecomp_evaluation import normalize_exact_answer
from deepresearch_harness.contracts import Usage
from deepresearch_harness.document_target_oracle import (
    ObligationTargetSlate,
    select_obligation_target_slate,
)
from deepresearch_harness.evidence_span_oracle import (
    ArtifactReference,
    SelectedEvidenceSpan,
    select_answer_obligation_span,
)
from deepresearch_harness.providers import LLMProvider
from deepresearch_harness.screening_judge import (
    create_judge_prompt,
    parse_judge_response,
)


JudgeFunction = Callable[[str], tuple[str, dict[str, object]]]
DocumentLoader = Callable[[str], str | None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _text_sha256(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OverlayProviderContract(StrictContract):
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    maximum_calls: int = Field(ge=1, le=16)
    maximum_output_tokens_per_call: int = Field(ge=64, le=2_000)
    maximum_estimated_cost_usd: float = Field(gt=0)


class OverlayJudgeContract(StrictContract):
    manifest: ArtifactReference
    calibration: ArtifactReference
    base_url: str = Field(min_length=1)
    served_model_name: str = Field(min_length=1)
    maximum_calls: int = Field(ge=1, le=16)


class OverlayAcceptance(StrictContract):
    minimum_literal_supported_replacements: int = Field(ge=1)
    minimum_normalized_exact_matches: int = Field(ge=1)
    minimum_judge_correct: int = Field(ge=1)
    maximum_proposal_parse_failures: Literal[0] = 0
    maximum_unsupported_replacements: Literal[0] = 0
    maximum_provider_cost_usd: float = Field(gt=0)


class PostRunOverlayRegistration(StrictContract):
    schema_version: Literal["monotonic-post-run-overlay-registration-v0"] = (
        "monotonic-post-run-overlay-registration-v0"
    )
    status: Literal["outcome_selected_development_calibration"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    baseline_run_root: str = Field(min_length=1)
    baseline_judge: ArtifactReference
    gold_slice: ArtifactReference
    target_oracle: ArtifactReference
    span_oracle: ArtifactReference
    prepared_reference: ArtifactReference | None = None
    query_ids: tuple[str, ...] = Field(min_length=1)
    document_index_path: str = Field(min_length=1)
    target_selector_id: Literal["obligation_channel_slate_v0"] = (
        "obligation_channel_slate_v0"
    )
    maximum_target_search_calls: int = Field(ge=1, le=8)
    target_slots_per_channel: int = Field(ge=1, le=3)
    maximum_targets_per_case: int = Field(ge=1, le=16)
    span_selector_id: Literal["answer_obligation_window_v2"] = (
        "answer_obligation_window_v2"
    )
    maximum_span_characters: int = Field(ge=200, le=4_000)
    prompt_variant: Literal["baseline_visible_v0", "evidence_only_v1"] = (
        "baseline_visible_v0"
    )
    provider: OverlayProviderContract
    judge: OverlayJudgeContract
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: OverlayAcceptance
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def budgets_match_cases(self) -> "PostRunOverlayRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("overlay query IDs must be unique")
        query_count = len(self.query_ids)
        if self.provider.maximum_calls != query_count:
            raise ValueError("overlay provider calls must equal the case count")
        if self.judge.maximum_calls != query_count:
            raise ValueError("overlay Judge calls must equal the case count")
        possible_targets = (
            self.maximum_target_search_calls * self.target_slots_per_channel * 2
        )
        if self.maximum_targets_per_case < possible_targets:
            raise ValueError("overlay target cap is lower than selector maximum")
        acceptance = self.acceptance
        for value in (
            acceptance.minimum_literal_supported_replacements,
            acceptance.minimum_normalized_exact_matches,
            acceptance.minimum_judge_correct,
        ):
            if value > query_count:
                raise ValueError("overlay acceptance gate exceeds the case count")
        if (
            acceptance.maximum_provider_cost_usd
            != self.provider.maximum_estimated_cost_usd
        ):
            raise ValueError("overlay cost gates differ")
        return self


class PreparedOverlaySpan(StrictContract):
    docid: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection: SelectedEvidenceSpan


class PreparedOverlayCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    baseline_answer: str = Field(min_length=1)
    baseline_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_slate: ObligationTargetSlate
    spans: tuple[PreparedOverlaySpan, ...] = Field(min_length=1)


class PreparedOverlayBatch(StrictContract):
    schema_version: Literal["monotonic-post-run-overlay-prepared-v0"] = (
        "monotonic-post-run-overlay-prepared-v0"
    )
    created_at: str = Field(min_length=1)
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_calls: Literal[0] = 0
    new_search_calls: Literal[0] = 0
    document_open_calls: int = Field(ge=0)
    judge_calls: Literal[0] = 0
    items: tuple[PreparedOverlayCase, ...] = Field(min_length=1)


class RepairProposal(StrictContract):
    action: Literal["keep", "replace"]
    short_answer: str | None = None
    support_docids: tuple[str, ...] = ()
    evidence_quote: str | None = None

    @model_validator(mode="after")
    def action_fields_match(self) -> "RepairProposal":
        if self.action == "keep":
            if self.short_answer or self.support_docids or self.evidence_quote:
                raise ValueError("keep proposal must not carry replacement fields")
            return self
        if not self.short_answer or not self.short_answer.strip():
            raise ValueError("replace proposal requires a short answer")
        if not self.support_docids or len(self.support_docids) != len(
            set(self.support_docids)
        ):
            raise ValueError("replace proposal requires unique support docids")
        if not self.evidence_quote or not self.evidence_quote.strip():
            raise ValueError("replace proposal requires an evidence quote")
        return self


class LiteralSupportDecision(StrictContract):
    applied: bool
    reason: Literal[
        "provider_keep",
        "proposal_parse_failure",
        "unknown_support_docid",
        "answer_not_literal_in_cited_span",
        "quote_not_literal_in_cited_span",
        "literal_supported_replacement",
    ]
    supporting_docids: tuple[str, ...]
    candidate_answer: str = Field(min_length=1)
    candidate_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OverlayProviderCaseResult(StrictContract):
    schema_version: Literal["monotonic-post-run-overlay-provider-case-v0"] = (
        "monotonic-post-run-overlay-provider-case-v0"
    )
    created_at: str = Field(min_length=1)
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_id: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_response: str
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal: RepairProposal | None
    proposal_parse_error: str | None
    support_decision: LiteralSupportDecision
    provider_model: str = Field(min_length=1)
    provider_latency_ms: int = Field(ge=0)
    provider_usage: Usage


class OverlayJudgeCaseResult(StrictContract):
    schema_version: Literal["monotonic-post-run-overlay-judge-case-v0"] = (
        "monotonic-post-run-overlay-judge-case-v0"
    )
    created_at: str = Field(min_length=1)
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_id: str = Field(min_length=1)
    candidate_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_response: str
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correct: bool | None
    confidence: float | None = Field(default=None, ge=0, le=100)
    parse_error: bool
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class OverlayCaseResult(StrictContract):
    query_id: str = Field(min_length=1)
    applied: bool
    support_reason: str = Field(min_length=1)
    supporting_docids: tuple[str, ...]
    candidate_exact_answer: str | None
    correct_answer: str = Field(min_length=1)
    normalized_exact_match: bool
    judge_correct: bool | None
    proposal_parse_error: bool
    provider_usage: Usage


class OverlayGate(StrictContract):
    gate_id: str = Field(min_length=1)
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class PostRunOverlayResult(StrictContract):
    schema_version: Literal["monotonic-post-run-overlay-result-v0"] = (
        "monotonic-post-run-overlay-result-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded"] = "succeeded"
    decision: Literal["pass", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    literal_supported_replacements: int = Field(ge=0)
    unsupported_replacements: int = Field(ge=0)
    proposal_parse_failures: int = Field(ge=0)
    normalized_exact_matches: int = Field(ge=0)
    judge_correct: int = Field(ge=0)
    judge_parse_failures: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    provider_input_tokens: int = Field(ge=0)
    provider_output_tokens: int = Field(ge=0)
    provider_estimated_cost_usd: float = Field(ge=0)
    new_search_calls: Literal[0] = 0
    document_open_calls: int = Field(ge=0)
    judge_calls: int = Field(ge=0)
    items: tuple[OverlayCaseResult, ...]
    gates: tuple[OverlayGate, ...]
    next_action: Literal[
        "preregister_fresh_paired_overlay_test",
        "freeze_overlay_and_rediagnose",
    ]
    claim_boundary: str = Field(min_length=1)


def load_post_run_overlay_registration(path: Path) -> PostRunOverlayRegistration:
    registration = PostRunOverlayRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            registration.baseline_judge,
            registration.gold_slice,
            registration.target_oracle,
            registration.span_oracle,
            *(
                (registration.prepared_reference,)
                if registration.prepared_reference is not None
                else ()
            ),
            registration.judge.manifest,
            registration.judge.calibration,
            *registration.frozen_artifacts,
        ),
    )
    for relative, label in (
        (registration.baseline_run_root, "baseline run root"),
        (registration.document_index_path, "document index"),
    ):
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            raise ValueError(f"overlay {label} is missing or escapes the repository")
    target_oracle = json.loads(
        (root / registration.target_oracle.path).read_text(encoding="utf-8")
    )
    span_oracle = json.loads(
        (root / registration.span_oracle.path).read_text(encoding="utf-8")
    )
    if target_oracle.get("decision") != "pass" or span_oracle.get("decision") != "pass":
        raise ValueError("overlay prerequisites did not pass")
    return registration


def prepare_overlay_batch(
    *,
    registration_path: Path,
    document_loader: DocumentLoader,
) -> PreparedOverlayBatch:
    registration = load_post_run_overlay_registration(registration_path)
    root = registration_path.resolve().parents[2]
    baseline_judge = json.loads(
        (root / registration.baseline_judge.path).read_text(encoding="utf-8")
    )
    baseline_judge_by_id = {
        str(row["query_id"]): row for row in baseline_judge["observations"]
    }
    items: list[PreparedOverlayCase] = []
    for query_id in registration.query_ids:
        judge_row = baseline_judge_by_id.get(query_id)
        if judge_row is None or judge_row.get("correct") is not False:
            raise ValueError(f"overlay case is not a frozen baseline Judge failure: {query_id}")
        run_path = root / registration.baseline_run_root / query_id / "run.json"
        if not run_path.is_file():
            raise ValueError(f"overlay baseline run is missing: {query_id}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        baseline_answer = str(run.get("answer_text", "")).strip()
        if not baseline_answer:
            raise ValueError(f"overlay baseline answer is blank: {query_id}")
        slate = select_obligation_target_slate(
            run,
            maximum_search_calls=registration.maximum_target_search_calls,
            slots_per_channel=registration.target_slots_per_channel,
        )
        if len(slate.targets) > registration.maximum_targets_per_case:
            raise ValueError(f"overlay target slate exceeds its cap: {query_id}")
        spans: list[PreparedOverlaySpan] = []
        for target in slate.targets:
            contents = document_loader(target.docid)
            if contents is None or not contents.strip():
                raise ValueError(f"overlay target document is missing: {target.docid}")
            selected = select_answer_obligation_span(
                contents,
                slate.question,
                maximum_span_characters=registration.maximum_span_characters,
                selector_id=registration.span_selector_id,
            )
            spans.append(
                PreparedOverlaySpan(
                    docid=target.docid,
                    document_sha256=_text_sha256(contents),
                    selection=selected,
                )
            )
        items.append(
            PreparedOverlayCase(
                query_id=query_id,
                question=slate.question,
                baseline_answer=baseline_answer,
                baseline_answer_sha256=_text_sha256(baseline_answer),
                target_slate=slate,
                spans=tuple(spans),
            )
        )
    prepared = PreparedOverlayBatch(
        created_at=_utc_now(),
        registration_sha256=_sha256_file(registration_path),
        document_open_calls=sum(len(item.spans) for item in items),
        items=tuple(items),
    )
    if registration.prepared_reference is not None:
        reference = PreparedOverlayBatch.model_validate_json(
            (root / registration.prepared_reference.path).read_text(encoding="utf-8")
        )
        if [item.model_dump(mode="json") for item in prepared.items] != [
            item.model_dump(mode="json") for item in reference.items
        ]:
            raise ValueError("overlay prepared cases differ from the frozen reference")
    return prepared


def build_overlay_prompt(
    case: PreparedOverlayCase,
    *,
    prompt_variant: Literal["baseline_visible_v0", "evidence_only_v1"] = (
        "baseline_visible_v0"
    ),
) -> str:
    payload = {
        "task": (
            "Decide whether the baseline short answer should be replaced using only "
            "the supplied evidence spans. Return JSON only."
        ),
        "question": case.question,
        "evidence_spans": [
            {"docid": span.docid, "content": span.selection.content}
            for span in case.spans
        ],
        "rules": [
            "Use action=replace only when the requested short answer is explicitly stated in a supplied span.",
            "short_answer must be the minimal answer requested, copied verbatim from the cited span.",
            "evidence_quote must be a verbatim quote from one cited span and contain short_answer.",
            "Every support_docid must name a supplied span that literally contains both short_answer and evidence_quote.",
            "Otherwise use action=keep and set short_answer=null, support_docids=[], evidence_quote=null.",
            "Do not use outside knowledge or infer a value that is not explicit in the spans.",
        ],
        "json_schema": {
            "action": "keep | replace",
            "short_answer": "string | null",
            "support_docids": ["docid"],
            "evidence_quote": "string | null",
        },
    }
    if prompt_variant == "baseline_visible_v0":
        payload["baseline_response"] = case.baseline_answer
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def apply_literal_support_gate(
    *,
    baseline_answer: str,
    spans: Sequence[PreparedOverlaySpan],
    proposal: RepairProposal | None,
    proposal_parse_failed: bool = False,
) -> LiteralSupportDecision:
    if proposal_parse_failed or proposal is None:
        return _keep_decision(baseline_answer, "proposal_parse_failure")
    if proposal.action == "keep":
        return _keep_decision(baseline_answer, "provider_keep")
    by_docid = {span.docid: span.selection.content for span in spans}
    if any(docid not in by_docid for docid in proposal.support_docids):
        return _keep_decision(baseline_answer, "unknown_support_docid")
    assert proposal.short_answer is not None
    assert proposal.evidence_quote is not None
    answer_needle = _literal_normalize(proposal.short_answer)
    quote_needle = _literal_normalize(proposal.evidence_quote)
    answer_support = tuple(
        docid
        for docid in proposal.support_docids
        if answer_needle in _literal_normalize(by_docid[docid])
    )
    if len(answer_support) != len(proposal.support_docids):
        return _keep_decision(baseline_answer, "answer_not_literal_in_cited_span")
    quote_support = tuple(
        docid
        for docid in proposal.support_docids
        if quote_needle in _literal_normalize(by_docid[docid])
    )
    if len(quote_support) != len(proposal.support_docids) or (
        answer_needle not in quote_needle
    ):
        return _keep_decision(baseline_answer, "quote_not_literal_in_cited_span")
    citations = " ".join(f"[{docid}]" for docid in proposal.support_docids)
    quote = " ".join(proposal.evidence_quote.split())
    candidate = (
        f'Explanation: The bounded post-run evidence check found the answer '
        f'explicitly in the opened span: "{quote}" {citations}\n'
        f"Exact Answer: {proposal.short_answer.strip()}\n"
        "Confidence: 95%"
    )
    return LiteralSupportDecision(
        applied=True,
        reason="literal_supported_replacement",
        supporting_docids=proposal.support_docids,
        candidate_answer=candidate,
        candidate_answer_sha256=_text_sha256(candidate),
    )


def run_post_run_overlay_calibration(
    *,
    registration_path: Path,
    output_dir: Path,
    provider: LLMProvider,
    document_loader: DocumentLoader,
    judge: JudgeFunction,
    resume: bool = False,
) -> PostRunOverlayResult:
    registration = load_post_run_overlay_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_dir.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("overlay output must stay under ignored runs/")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        if not resume:
            raise ValueError("overlay output already exists; use resume")
        return PostRunOverlayResult.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
    if output_dir.exists() and not resume:
        raise ValueError("overlay output directory already exists; use resume")
    if provider.model != registration.provider.model:
        raise ValueError("overlay provider model differs from registration")
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_dir = output_dir / "provider_results"
    judge_dir = output_dir / "judge_results"
    failures_dir = output_dir / "failures"
    provider_dir.mkdir(exist_ok=True)
    judge_dir.mkdir(exist_ok=True)
    failures_dir.mkdir(exist_ok=True)
    registration_sha256 = _sha256_file(registration_path)

    prepared_path = output_dir / "prepared.json"
    if prepared_path.exists():
        prepared = PreparedOverlayBatch.model_validate_json(
            prepared_path.read_text(encoding="utf-8")
        )
        if prepared.registration_sha256 != registration_sha256:
            raise ValueError("prepared overlay targets another registration")
    else:
        prepared = prepare_overlay_batch(
            registration_path=registration_path,
            document_loader=document_loader,
        )
        _atomic_json(prepared_path, prepared.model_dump(mode="json"))

    provider_results: dict[str, OverlayProviderCaseResult] = {}
    spent_cost = 0.0
    provider_calls = 0
    for case in prepared.items:
        result_path = provider_dir / f"{case.query_id}.json"
        if result_path.exists():
            result = OverlayProviderCaseResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            _validate_provider_case(
                result,
                case,
                registration_sha256,
                registration.prompt_variant,
            )
            provider_results[case.query_id] = result
            spent_cost += result.provider_usage.estimated_cost_usd
            provider_calls += 1
            continue
        if provider_calls >= registration.provider.maximum_calls:
            raise ValueError("overlay provider call budget exhausted")
        if spent_cost >= registration.provider.maximum_estimated_cost_usd:
            raise ValueError("overlay provider cost budget exhausted")
        prompt = build_overlay_prompt(
            case,
            prompt_variant=registration.prompt_variant,
        )
        try:
            completion = provider.complete(
                stage="post_run_overlay",
                prompt=prompt,
                json_output=True,
                max_output_tokens=registration.provider.maximum_output_tokens_per_call,
            )
        except Exception as error:
            _atomic_json(
                failures_dir / f"{case.query_id}.json",
                {
                    "created_at": _utc_now(),
                    "registration_sha256": registration_sha256,
                    "query_id": case.query_id,
                    "prompt_sha256": _text_sha256(prompt),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry_policy": "manual_resume_only",
                },
            )
            raise
        proposal: RepairProposal | None = None
        parse_error: str | None = None
        try:
            proposal = RepairProposal.model_validate(json.loads(completion.text))
        except Exception as error:
            parse_error = f"{type(error).__name__}:{error}"
        support = apply_literal_support_gate(
            baseline_answer=case.baseline_answer,
            spans=case.spans,
            proposal=proposal,
            proposal_parse_failed=parse_error is not None,
        )
        result = OverlayProviderCaseResult(
            created_at=_utc_now(),
            registration_sha256=registration_sha256,
            query_id=case.query_id,
            prompt_sha256=_text_sha256(prompt),
            raw_response=completion.text,
            raw_response_sha256=_text_sha256(completion.text),
            proposal=proposal,
            proposal_parse_error=parse_error,
            support_decision=support,
            provider_model=provider.model,
            provider_latency_ms=completion.latency_ms,
            provider_usage=completion.usage,
        )
        _atomic_json(result_path, result.model_dump(mode="json"))
        provider_results[case.query_id] = result
        spent_cost += completion.usage.estimated_cost_usd
        provider_calls += 1
        if spent_cost > registration.provider.maximum_estimated_cost_usd:
            raise ValueError("overlay provider cost exceeded the registered cap")

    gold = json.loads((root / registration.gold_slice.path).read_text(encoding="utf-8"))
    gold_by_id = {str(row["query_id"]): row for row in gold["rows"]}
    judge_results: dict[str, OverlayJudgeCaseResult] = {}
    for case in prepared.items:
        provider_result = provider_results[case.query_id]
        candidate = provider_result.support_decision.candidate_answer
        result_path = judge_dir / f"{case.query_id}.json"
        if result_path.exists():
            result = OverlayJudgeCaseResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            if (
                result.registration_sha256 != registration_sha256
                or result.candidate_answer_sha256 != _text_sha256(candidate)
            ):
                raise ValueError("resumed overlay Judge result drifted")
            judge_results[case.query_id] = result
            continue
        if len(judge_results) >= registration.judge.maximum_calls:
            raise ValueError("overlay Judge call budget exhausted")
        reference = gold_by_id.get(case.query_id)
        if reference is None:
            raise ValueError(f"overlay gold case is missing: {case.query_id}")
        prompt = create_judge_prompt(
            case.question,
            candidate,
            str(reference["answer"]),
        )
        raw, usage = judge(prompt)
        parsed = parse_judge_response(raw)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(
            usage.get("total_tokens", prompt_tokens + completion_tokens)
        )
        result = OverlayJudgeCaseResult(
            created_at=_utc_now(),
            registration_sha256=registration_sha256,
            query_id=case.query_id,
            candidate_answer_sha256=_text_sha256(candidate),
            raw_response=raw,
            raw_response_sha256=_text_sha256(raw),
            correct=(
                parsed["correct"] if isinstance(parsed["correct"], bool) else None
            ),
            confidence=(
                float(parsed["confidence"])
                if isinstance(parsed["confidence"], (int, float))
                else None
            ),
            parse_error=bool(parsed["parse_error"]),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        _atomic_json(result_path, result.model_dump(mode="json"))
        judge_results[case.query_id] = result

    items: list[OverlayCaseResult] = []
    for case in prepared.items:
        provider_result = provider_results[case.query_id]
        judge_result = judge_results[case.query_id]
        reference = gold_by_id[case.query_id]
        candidate = provider_result.support_decision.candidate_answer
        exact = _extract_exact_answer(candidate)
        correct_answer = str(reference["answer"])
        items.append(
            OverlayCaseResult(
                query_id=case.query_id,
                applied=provider_result.support_decision.applied,
                support_reason=provider_result.support_decision.reason,
                supporting_docids=provider_result.support_decision.supporting_docids,
                candidate_exact_answer=exact,
                correct_answer=correct_answer,
                normalized_exact_match=(
                    exact is not None
                    and normalize_exact_answer(exact)
                    == normalize_exact_answer(correct_answer)
                ),
                judge_correct=judge_result.correct,
                proposal_parse_error=provider_result.proposal_parse_error is not None,
                provider_usage=provider_result.provider_usage,
            )
        )
    result = _summarize_overlay(
        registration=registration,
        registration_sha256=registration_sha256,
        prepared=prepared,
        items=tuple(items),
        judge_results=judge_results,
    )
    _atomic_json(summary_path, result.model_dump(mode="json"))
    return result


def _summarize_overlay(
    *,
    registration: PostRunOverlayRegistration,
    registration_sha256: str,
    prepared: PreparedOverlayBatch,
    items: tuple[OverlayCaseResult, ...],
    judge_results: dict[str, OverlayJudgeCaseResult],
) -> PostRunOverlayResult:
    replacements = sum(item.applied for item in items)
    unsupported = sum(
        item.support_reason
        not in {"literal_supported_replacement", "provider_keep", "proposal_parse_failure"}
        for item in items
    )
    parse_failures = sum(item.proposal_parse_error for item in items)
    exact = sum(item.normalized_exact_match for item in items)
    judge_correct = sum(item.judge_correct is True for item in items)
    judge_parse_failures = sum(row.parse_error for row in judge_results.values())
    usage = [item.provider_usage for item in items]
    input_tokens = sum(row.input_tokens for row in usage)
    output_tokens = sum(row.output_tokens for row in usage)
    cost = sum(row.estimated_cost_usd for row in usage)
    acceptance = registration.acceptance
    gates = (
        _gate(
            "literal_supported_replacements",
            replacements,
            "ge",
            acceptance.minimum_literal_supported_replacements,
        ),
        _gate(
            "normalized_exact_matches",
            exact,
            "ge",
            acceptance.minimum_normalized_exact_matches,
        ),
        _gate(
            "judge_correct",
            judge_correct,
            "ge",
            acceptance.minimum_judge_correct,
        ),
        _gate(
            "proposal_parse_failures",
            parse_failures,
            "eq",
            acceptance.maximum_proposal_parse_failures,
        ),
        _gate(
            "unsupported_replacements",
            unsupported,
            "eq",
            acceptance.maximum_unsupported_replacements,
        ),
        _gate(
            "provider_estimated_cost_usd",
            cost,
            "le",
            acceptance.maximum_provider_cost_usd,
        ),
    )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    return PostRunOverlayResult(
        created_at=_utc_now(),
        decision=decision,
        registration_sha256=registration_sha256,
        query_count=len(items),
        literal_supported_replacements=replacements,
        unsupported_replacements=unsupported,
        proposal_parse_failures=parse_failures,
        normalized_exact_matches=exact,
        judge_correct=judge_correct,
        judge_parse_failures=judge_parse_failures,
        provider_calls=len(items),
        provider_input_tokens=input_tokens,
        provider_output_tokens=output_tokens,
        provider_estimated_cost_usd=cost,
        document_open_calls=prepared.document_open_calls,
        judge_calls=len(judge_results),
        items=items,
        gates=gates,
        next_action=(
            "preregister_fresh_paired_overlay_test"
            if decision == "pass"
            else "freeze_overlay_and_rediagnose"
        ),
        claim_boundary=registration.claim_boundary,
    )


def _gate(
    gate_id: str,
    observed: int | float,
    operator: Literal["eq", "ge", "le"],
    threshold: int | float,
) -> OverlayGate:
    passed = {
        "eq": observed == threshold,
        "ge": observed >= threshold,
        "le": observed <= threshold,
    }[operator]
    return OverlayGate(
        gate_id=gate_id,
        observed=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
    )


def _literal_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _keep_decision(
    baseline_answer: str,
    reason: Literal[
        "provider_keep",
        "proposal_parse_failure",
        "unknown_support_docid",
        "answer_not_literal_in_cited_span",
        "quote_not_literal_in_cited_span",
    ],
) -> LiteralSupportDecision:
    return LiteralSupportDecision(
        applied=False,
        reason=reason,
        supporting_docids=(),
        candidate_answer=baseline_answer,
        candidate_answer_sha256=_text_sha256(baseline_answer),
    )


def _extract_exact_answer(answer_text: str) -> str | None:
    match = re.search(
        r"(?:^|\n)\s*(?:\*\*)?Exact Answer(?:\*\*)?\s*:\s*(.+?)(?=\n|$)",
        answer_text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match and match.group(1).strip() else None


def _validate_provider_case(
    result: OverlayProviderCaseResult,
    case: PreparedOverlayCase,
    registration_sha256: str,
    prompt_variant: Literal["baseline_visible_v0", "evidence_only_v1"],
) -> None:
    if (
        result.registration_sha256 != registration_sha256
        or result.query_id != case.query_id
        or result.prompt_sha256
        != _text_sha256(build_overlay_prompt(case, prompt_variant=prompt_variant))
    ):
        raise ValueError("resumed overlay provider result drifted")


def _validate_artifacts(root: Path, artifacts: Sequence[ArtifactReference]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
