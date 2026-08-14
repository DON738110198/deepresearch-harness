from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_judge import (
    OfficialJudgeComparison,
    validate_official_judge_batch,
)
from .browsecomp_plus import normalized_text_file_sha256


GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip().replace(
    "extracted from the [response].\n",
    "extracted from the [response]. \n",
    1,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScreeningEngine(StrictContract):
    name: Literal["vllm_openai_compatible_server"]
    version: str = Field(min_length=1)
    host: Literal["127.0.0.1"]
    served_model_name: str = Field(min_length=1)
    tensor_parallel_size: Literal[1]
    gpu_id: int = Field(ge=0)
    gpu_model: str = Field(min_length=1)
    gpu_memory_mib: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int = Field(gt=4096)
    max_num_seqs: int = Field(gt=0)
    enable_prefix_caching: bool


class ScreeningModel(StrictContract):
    model: Literal["Qwen/Qwen3-32B-AWQ"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    quantization: Literal["awq_int4"]
    base_model: Literal["Qwen/Qwen3-32B"]


class ScreeningInference(StrictContract):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    enable_thinking: Literal[False]


class ScreeningPrompt(StrictContract):
    source_repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    evaluator_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grader_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScreeningCalibration(StrictContract):
    reference_track: Literal["official_qwen3_32b_bf16_two_gpu"]
    reference_evaluations: int = Field(gt=0)
    reference_batch_manifest_path: str = Field(min_length=1)
    reference_batch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_comparison_path: str = Field(min_length=1)
    reference_comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_execution_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_label_agreement_percent: float = Field(ge=0, le=100)
    minimum_cohens_kappa: float = Field(ge=-1, le=1)
    maximum_parse_failures: int = Field(ge=0)
    maximum_absolute_pooled_accuracy_delta_pp: float = Field(ge=0)
    require_paired_variant_delta_sign_match: bool


class ScreeningJudgeManifest(StrictContract):
    schema_version: Literal["browsecomp-plus-screening-judge-v0"]
    status: Literal["planned_not_run"]
    purpose: str = Field(min_length=1)
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine: ScreeningEngine
    judge: ScreeningModel
    prompt: ScreeningPrompt
    inference: ScreeningInference
    calibration: ScreeningCalibration
    acceptance: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


class ScreeningObservation(StrictContract):
    batch_id: str
    trial_id: str
    execution_order: Literal["baseline_first", "candidate_first"]
    variant: Literal["baseline", "candidate"]
    query_id: str
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correct: bool | None
    confidence: float | None = Field(default=None, ge=0, le=100)
    parse_error: bool
    latency_ms: int = Field(ge=0)
    result_path: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScreeningJudgeResult(StrictContract):
    schema_version: Literal["browsecomp-plus-screening-judge-result-v0"]
    created_at: str
    status: Literal["succeeded", "failed"]
    batch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_model: Literal["Qwen/Qwen3-32B-AWQ"]
    judge_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    served_model_name: str
    inference: ScreeningInference
    evaluations: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    request_failures: int = Field(ge=0)
    request_errors: list[str]
    correct: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)
    observations: list[ScreeningObservation]

    @model_validator(mode="after")
    def counts_match(self) -> "ScreeningJudgeResult":
        if self.evaluations != len(self.observations):
            raise ValueError("screening evaluation count differs from observations")
        if self.parse_failures != sum(row.parse_error for row in self.observations):
            raise ValueError("screening parse-failure count differs")
        if self.correct != sum(row.correct is True for row in self.observations):
            raise ValueError("screening correct count differs")
        if self.request_failures != len(self.request_errors):
            raise ValueError("screening request-failure count differs")
        return self


def load_screening_manifest(path: Path) -> ScreeningJudgeManifest:
    return ScreeningJudgeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def parse_judge_response(text: str) -> dict[str, object]:
    result: dict[str, object] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not text:
        result["parse_error"] = True
        return result
    answer = _first_match(
        [
            r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
        ],
        text,
        re.DOTALL,
    )
    reasoning = _first_match(
        [
            r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
        ],
        text,
        re.DOTALL,
    )
    correct = _first_match(
        [
            r"\*\*correct:\*\*\s*(yes|no)",
            r"\*\*correct\*\*:\s*(yes|no)",
            r"correct:\s*(yes|no)",
        ],
        text,
    )
    confidence = _first_match(
        [
            r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
            r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
            r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
        ],
        text,
    )
    result["extracted_final_answer"] = answer
    result["reasoning"] = reasoning
    if correct is not None:
        result["correct"] = correct.lower() == "yes"
    else:
        result["parse_error"] = True
    if confidence is not None:
        result["confidence"] = min(float(confidence), 100.0)
    return result


class VllmChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        served_model_name: str,
        inference: ScreeningInference,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("screening judge server must be loopback HTTP")
        self.base_url = base_url.rstrip("/")
        self.served_model_name = served_model_name
        self.inference = inference
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def model_ids(self) -> list[str]:
        payload = self._request("GET", "/models")
        return [str(row["id"]) for row in payload.get("data", [])]

    def judge(self, prompt: str) -> tuple[str, dict[str, object]]:
        body = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.inference.temperature,
            "top_p": self.inference.top_p,
            "top_k": self.inference.top_k,
            "max_tokens": self.inference.max_output_tokens,
            "chat_template_kwargs": {
                "enable_thinking": self.inference.enable_thinking
            },
        }
        payload = self._request("POST", "/chat/completions", body)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("vLLM response has no choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("vLLM response has no text content")
        usage = payload.get("usage")
        return content, usage if isinstance(usage, dict) else {}

    def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        raw = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=raw,
            method=method,
            headers={"content-type": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("vLLM response must be an object")
                return payload
            except (OSError, ValueError, urllib.error.HTTPError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise RuntimeError(f"vLLM request failed: {last_error}") from last_error


def run_screening_judge(
    *,
    screening_manifest_path: Path,
    batch_manifest_path: Path,
    output_dir: Path,
    base_url: str,
    concurrency: int = 16,
    timeout_seconds: float = 600,
    retries: int = 2,
) -> ScreeningJudgeResult:
    if concurrency < 1:
        raise ValueError("screening concurrency must be positive")
    if output_dir.exists():
        raise ValueError("screening output directory must not exist")
    screening = load_screening_manifest(screening_manifest_path)
    batch_bytes = batch_manifest_path.read_bytes()
    batch = validate_official_judge_batch(batch_manifest_path)
    if sha256(batch_bytes).hexdigest() != (
        screening.calibration.reference_batch_manifest_sha256
    ):
        raise ValueError("screening calibration batch hash differs from registration")
    if batch.target_manifest_sha256 != screening.target_manifest_sha256:
        raise ValueError("screening judge and batch target different benchmarks")
    if batch.repository_commit != screening.prompt.source_repository_commit:
        raise ValueError("screening prompt repository revision differs from batch")
    if batch.evaluator_script_sha256 != screening.prompt.evaluator_script_sha256:
        raise ValueError("screening evaluator script hash differs from batch")
    if sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest() != (
        screening.prompt.grader_template_sha256
    ):
        raise ValueError("screening grader template differs from registration")
    if batch.input_count != screening.calibration.reference_evaluations:
        raise ValueError("screening calibration evaluation count differs")
    if batch.inference.model_dump() != screening.inference.model_dump():
        raise ValueError("screening sampling contract differs from official evaluator")

    client = VllmChatClient(
        base_url=base_url,
        served_model_name=screening.engine.served_model_name,
        inference=screening.inference,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    if screening.engine.served_model_name not in client.model_ids():
        raise ValueError("vLLM server does not expose the registered model name")

    ground_truth = _load_ground_truth(
        _resolve_batch_path(batch_manifest_path.parent, batch.ground_truth_path)
    )
    output_dir.mkdir(parents=True)
    results_dir = output_dir / "results"
    results_dir.mkdir()
    started = time.perf_counter()
    observations: list[ScreeningObservation] = []
    request_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _evaluate_item,
                item=item,
                batch_root=batch_manifest_path.parent,
                ground_truth=ground_truth,
                results_dir=results_dir,
                client=client,
            ): item
            for item in batch.items
        }
        for future in as_completed(futures):
            try:
                observations.append(future.result())
            except Exception as error:
                item = futures[future]
                request_errors.append(
                    f"{item.batch_id}:{type(error).__name__}:{error}"
                )
    observations.sort(key=lambda row: row.batch_id)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    request_errors.sort()
    status = "succeeded" if not request_errors else "failed"
    result = ScreeningJudgeResult(
        schema_version="browsecomp-plus-screening-judge-result-v0",
        created_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        batch_manifest_sha256=sha256(batch_bytes).hexdigest(),
        screening_manifest_sha256=normalized_text_file_sha256(
            screening_manifest_path
        ),
        judge_model=screening.judge.model,
        judge_revision=screening.judge.revision,
        served_model_name=screening.engine.served_model_name,
        inference=screening.inference,
        evaluations=len(observations),
        parse_failures=sum(row.parse_error for row in observations),
        request_failures=len(request_errors),
        request_errors=request_errors,
        correct=sum(row.correct is True for row in observations),
        elapsed_ms=elapsed_ms,
        observations=observations,
    )
    _atomic_json(output_dir / "screening_result.json", result.model_dump(mode="json"))
    if result.status != "succeeded":
        raise RuntimeError(
            f"screening judge had {result.request_failures} request failures"
        )
    return result


def calibrate_screening_judge(
    *,
    screening_manifest_path: Path,
    screening_result_path: Path,
    official_comparison_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise ValueError("screening calibration output already exists")
    manifest = load_screening_manifest(screening_manifest_path)
    result_bytes = screening_result_path.read_bytes()
    result = ScreeningJudgeResult.model_validate_json(result_bytes)
    comparison_bytes = official_comparison_path.read_bytes()
    comparison = OfficialJudgeComparison.model_validate_json(comparison_bytes)
    if sha256(comparison_bytes).hexdigest() != (
        manifest.calibration.reference_comparison_sha256
    ):
        raise ValueError("official comparison differs from calibration registration")
    if comparison.batch_manifest_sha256 != (
        manifest.calibration.reference_batch_manifest_sha256
    ):
        raise ValueError("official comparison targets another calibration batch")
    if comparison.execution_result_sha256 != (
        manifest.calibration.reference_execution_result_sha256
    ):
        raise ValueError("official execution differs from calibration registration")
    if comparison.evaluations != manifest.calibration.reference_evaluations:
        raise ValueError("official comparison evaluation count differs")
    if result.status != "succeeded" or result.request_failures:
        raise ValueError("screening execution did not succeed")
    if result.screening_manifest_sha256 != normalized_text_file_sha256(
        screening_manifest_path
    ):
        raise ValueError("screening result is not bound to its manifest")
    if (
        result.judge_model != manifest.judge.model
        or result.judge_revision != manifest.judge.revision
        or result.served_model_name != manifest.engine.served_model_name
        or result.inference != manifest.inference
    ):
        raise ValueError("screening result changes the registered judge contract")
    if result.batch_manifest_sha256 != (
        manifest.calibration.reference_batch_manifest_sha256
    ):
        raise ValueError("screening result targets another calibration batch")

    official_rows = {
        (row.trial_id, row.variant, row.query_id): row
        for row in comparison.observations
    }
    screened_rows = {
        (row.trial_id, row.variant, row.query_id): row
        for row in result.observations
        if row.correct is not None
    }
    if set(official_rows) != set(screened_rows):
        raise ValueError("screening and official label grids differ")
    if any(
        official_rows[key].prediction_sha256
        != screened_rows[key].prediction_sha256
        for key in official_rows
    ):
        raise ValueError("screening and official predictions differ")
    official = {key: row.correct for key, row in official_rows.items()}
    screened = {key: row.correct for key, row in screened_rows.items()}
    pairs = [(official[key], screened[key]) for key in sorted(official)]
    agreement = sum(left == right for left, right in pairs)
    agreement_percent = 100 * agreement / len(pairs)
    kappa = _cohens_kappa(pairs)
    official_accuracy = 100 * sum(left for left, _ in pairs) / len(pairs)
    screening_accuracy = 100 * sum(right for _, right in pairs) / len(pairs)
    absolute_accuracy_delta = abs(screening_accuracy - official_accuracy)

    def variant_accuracy(labels: dict, variant: str) -> float:
        selected = [value for key, value in labels.items() if key[1] == variant]
        return 100 * sum(value is True for value in selected) / len(selected)

    official_variant_delta = variant_accuracy(
        official, "candidate"
    ) - variant_accuracy(official, "baseline")
    screening_variant_delta = variant_accuracy(
        screened, "candidate"
    ) - variant_accuracy(screened, "baseline")
    sign_match = _sign(official_variant_delta) == _sign(screening_variant_delta)
    gates = {
        "label_agreement": agreement_percent
        >= manifest.calibration.minimum_label_agreement_percent,
        "cohens_kappa": kappa >= manifest.calibration.minimum_cohens_kappa,
        "parse_failures": result.parse_failures
        <= manifest.calibration.maximum_parse_failures,
        "pooled_accuracy_delta": absolute_accuracy_delta
        <= manifest.calibration.maximum_absolute_pooled_accuracy_delta_pp,
        "paired_variant_delta_sign": (
            sign_match
            if manifest.calibration.require_paired_variant_delta_sign_match
            else True
        ),
    }
    passed = all(gates.values())
    payload: dict[str, object] = {
        "schema_version": "browsecomp-plus-screening-judge-calibration-v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_for_development_screening" if passed else "rejected",
        "official_status": "reference_bf16_development_slice",
        "screening_status": "non_official_awq_development_screening",
        "screening_manifest_sha256": normalized_text_file_sha256(
            screening_manifest_path
        ),
        "screening_result_sha256": sha256(result_bytes).hexdigest(),
        "official_comparison_sha256": sha256(comparison_bytes).hexdigest(),
        "evaluations": len(pairs),
        "label_agreement_percent": round(agreement_percent, 6),
        "cohens_kappa": round(kappa, 6),
        "parse_failures": result.parse_failures,
        "official_pooled_accuracy_percent": round(official_accuracy, 6),
        "screening_pooled_accuracy_percent": round(screening_accuracy, 6),
        "absolute_pooled_accuracy_delta_pp": round(absolute_accuracy_delta, 6),
        "official_paired_variant_delta_pp": round(official_variant_delta, 6),
        "screening_paired_variant_delta_pp": round(screening_variant_delta, 6),
        "paired_variant_delta_sign_match": sign_match,
        "gates": gates,
        "claim_boundary": manifest.claim_boundary,
    }
    _atomic_json(output_path, payload)
    return payload


def _evaluate_item(*, item, batch_root, ground_truth, results_dir, client):
    run_path = _resolve_batch_path(batch_root, item.staged_input_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "completed" or run.get("query_id") != item.query_id:
        raise ValueError("screening input is not a completed matching run")
    result_parts = run.get("result")
    if not isinstance(result_parts, list) or not result_parts:
        raise ValueError("screening input has no result")
    final = result_parts[-1]
    response = final.get("output") if final.get("type") == "output_text" else None
    if not isinstance(response, str) or not response:
        raise ValueError("screening input has no final response")
    truth = ground_truth[item.query_id]
    prompt = create_judge_prompt(truth["question"], response, truth["answer"])
    started = time.perf_counter()
    judge_text, usage = client.judge(prompt)
    latency_ms = round((time.perf_counter() - started) * 1000)
    parsed = parse_judge_response(judge_text)
    result_payload = {
        "schema_version": "browsecomp-plus-screening-judge-item-v0",
        "batch_id": item.batch_id,
        "query_id": item.query_id,
        "response": response,
        "correct_answer": truth["answer"],
        "judge_prompt": prompt,
        "judge_response": judge_text,
        "judge_result": parsed,
        "usage": usage,
        "latency_ms": latency_ms,
    }
    result_path = results_dir / f"{item.batch_id}_eval.json"
    _atomic_json(result_path, result_payload)
    return ScreeningObservation(
        batch_id=item.batch_id,
        trial_id=item.trial_id,
        execution_order=item.execution_order,
        variant=item.variant,
        query_id=item.query_id,
        prediction_sha256=item.prediction_sha256,
        correct=parsed["correct"],
        confidence=parsed["confidence"],
        parse_error=bool(parsed["parse_error"]),
        latency_ms=latency_ms,
        result_path=result_path.relative_to(results_dir.parent).as_posix(),
        result_sha256=sha256(result_path.read_bytes()).hexdigest(),
    )


def _load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[str(row["query_id"])] = {
            "question": str(row["query"]),
            "answer": str(row["answer"]),
        }
    return result


def _resolve_batch_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("screening batch path must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("screening batch artifact is missing")
    return resolved


def _first_match(
    patterns: list[str], text: str, extra_flags: re.RegexFlag = re.NOFLAG
) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | extra_flags)
        if match:
            return match.group(1).strip()
    return None


def _cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_yes = sum(left for left, _ in pairs) / len(pairs)
    right_yes = sum(right for _, right in pairs) / len(pairs)
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
