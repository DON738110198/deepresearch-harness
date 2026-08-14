import json
import threading
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from deepresearch_harness.screening_judge import (
    GRADER_TEMPLATE,
    ScreeningInference,
    ScreeningJudgeResult,
    ScreeningObservation,
    VllmChatClient,
    calibrate_screening_judge,
    load_screening_manifest,
    parse_judge_response,
)


ROOT = Path(__file__).resolve().parents[1]
SCREENING_MANIFEST = (
    ROOT / "benchmarks" / "browsecomp_plus_v0" / "screening_judge_v0.json"
)


def test_screening_manifest_pins_single_gpu_non_official_contract() -> None:
    manifest = load_screening_manifest(SCREENING_MANIFEST)

    assert manifest.engine.gpu_id == 5
    assert manifest.engine.tensor_parallel_size == 1
    assert manifest.engine.host == "127.0.0.1"
    assert manifest.judge.model == "Qwen/Qwen3-32B-AWQ"
    assert manifest.inference.enable_thinking is False
    assert sha256(GRADER_TEMPLATE.encode("utf-8")).hexdigest() == (
        manifest.prompt.grader_template_sha256
    )
    assert "not official" in manifest.claim_boundary.lower()


def test_official_judge_response_parser_accepts_bold_and_plain_fields() -> None:
    parsed = parse_judge_response(
        "**extracted_final_answer:** Alpha\n"
        "**reasoning:** It matches.\n"
        "**correct:** yes\n"
        "confidence: 87%"
    )

    assert parsed == {
        "extracted_final_answer": "Alpha",
        "reasoning": "It matches.",
        "correct": True,
        "confidence": 87.0,
        "parse_error": False,
    }
    assert parse_judge_response("unstructured")["parse_error"] is True


def test_vllm_client_uses_loopback_chat_completions_and_registered_sampling() -> None:
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/v1/models"
            self._respond({"data": [{"id": "screening-model"}]})

        def do_POST(self) -> None:
            assert self.path == "/v1/chat/completions"
            length = int(self.headers["content-length"])
            requests.append(json.loads(self.rfile.read(length)))
            self._respond(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "correct: no\nconfidence: 41%"
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )

        def _respond(self, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = VllmChatClient(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            served_model_name="screening-model",
            inference=ScreeningInference(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_output_tokens=4096,
                enable_thinking=False,
            ),
            timeout_seconds=5,
            retries=0,
        )
        assert client.model_ids() == ["screening-model"]
        text, usage = client.judge("grade this")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert text == "correct: no\nconfidence: 41%"
    assert usage["completion_tokens"] == 5
    assert requests == [
        {
            "model": "screening-model",
            "messages": [{"role": "user", "content": "grade this"}],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]


def test_calibration_accepts_identical_awq_and_bf16_labels(tmp_path: Path) -> None:
    inference = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_output_tokens": 4096,
        "enable_thinking": False,
    }
    labels = {
        ("trial-01", "baseline", "q1"): False,
        ("trial-01", "candidate", "q1"): True,
        ("trial-02", "baseline", "q1"): False,
        ("trial-02", "candidate", "q1"): True,
        ("trial-03", "baseline", "q1"): True,
        ("trial-03", "candidate", "q1"): True,
    }
    official = {
        "schema_version": "browsecomp-plus-official-judge-comparison-v0",
        "created_at": "2026-08-14T00:00:00+00:00",
        "status": "official_evaluator_development_slice",
        "leaderboard_status": "not_submitted",
        "batch_manifest_sha256": "b" * 64,
        "execution_registration_sha256": "c" * 64,
        "execution_result_sha256": "d" * 64,
        "judge_model": "Qwen/Qwen3-32B",
        "judge_revision": "e" * 40,
        "inference": inference,
        "trial_count": 3,
        "queries_per_trial": 1,
        "evaluations": 6,
        "parse_failures": 0,
        "baseline": {
            "variant": "baseline",
            "correct": 1,
            "evaluations": 3,
            "pooled_accuracy_percent": 33.333333,
            "trial_accuracy_percent": _distribution(33.333333, 0, 100),
        },
        "candidate": {
            "variant": "candidate",
            "correct": 3,
            "evaluations": 3,
            "pooled_accuracy_percent": 100.0,
            "trial_accuracy_percent": _distribution(100.0, 100, 100),
        },
        "paired": {
            "comparisons": 3,
            "candidate_wins": 2,
            "baseline_wins": 0,
            "ties": 1,
        },
        "trials": [
            _trial("trial-01", "baseline_first", 0, 1),
            _trial("trial-02", "candidate_first", 0, 1),
            _trial("trial-03", "baseline_first", 1, 1),
        ],
        "observations": [
            {
                "trial_id": trial,
                "execution_order": (
                    "candidate_first" if trial == "trial-02" else "baseline_first"
                ),
                "variant": variant,
                "query_id": query_id,
                "correct": correct,
                "confidence": 80.0,
                "prediction_sha256": str(index + 1) * 64,
                "result_path": f"result-{index}.json",
                "result_sha256": "f" * 64,
            }
            for index, ((trial, variant, query_id), correct) in enumerate(labels.items())
        ],
    }
    comparison_path = tmp_path / "official.json"
    comparison_path.write_text(json.dumps(official), encoding="utf-8")
    comparison_hash = sha256(comparison_path.read_bytes()).hexdigest()
    manifest = _screening_manifest(comparison_hash)
    manifest_path = tmp_path / "screening.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()

    observations = [
        ScreeningObservation(
            batch_id=f"{trial}-{variant}__{query_id}",
            trial_id=trial,
            execution_order=(
                "candidate_first" if trial == "trial-02" else "baseline_first"
            ),
            variant=variant,
            query_id=query_id,
            prediction_sha256=str(index + 1) * 64,
            correct=correct,
            confidence=80,
            parse_error=False,
            latency_ms=10,
            result_path=f"results/{index}.json",
            result_sha256="a" * 64,
        )
        for index, ((trial, variant, query_id), correct) in enumerate(labels.items())
    ]
    screening_result = ScreeningJudgeResult(
        schema_version="browsecomp-plus-screening-judge-result-v0",
        created_at="2026-08-14T00:01:00+00:00",
        status="succeeded",
        batch_manifest_sha256="b" * 64,
        screening_manifest_sha256=manifest_hash,
        judge_model="Qwen/Qwen3-32B-AWQ",
        judge_revision="a" * 40,
        served_model_name="qwen3-32b-awq-screening",
        inference=ScreeningInference.model_validate(inference),
        evaluations=6,
        parse_failures=0,
        request_failures=0,
        request_errors=[],
        correct=4,
        elapsed_ms=100,
        observations=observations,
    )
    result_path = tmp_path / "screening-result.json"
    result_path.write_text(screening_result.model_dump_json(), encoding="utf-8")

    calibration = calibrate_screening_judge(
        screening_manifest_path=manifest_path,
        screening_result_path=result_path,
        official_comparison_path=comparison_path,
        output_path=tmp_path / "calibration.json",
    )

    assert calibration["status"] == "accepted_for_development_screening"
    assert calibration["label_agreement_percent"] == 100.0
    assert calibration["cohens_kappa"] == 1.0
    assert calibration["absolute_pooled_accuracy_delta_pp"] == 0.0
    assert calibration["paired_variant_delta_sign_match"] is True


def _distribution(mean: float, minimum: float, maximum: float) -> dict:
    return {
        "trials": 3,
        "mean": mean,
        "sample_stddev": 0.0,
        "minimum": minimum,
        "maximum": maximum,
    }


def _trial(
    trial_id: str, execution_order: str, baseline_correct: int, candidate_correct: int
) -> dict:
    return {
        "trial_id": trial_id,
        "execution_order": execution_order,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "query_count": 1,
        "baseline_accuracy_percent": baseline_correct * 100.0,
        "candidate_accuracy_percent": candidate_correct * 100.0,
    }


def _screening_manifest(comparison_hash: str) -> dict:
    return {
        "schema_version": "browsecomp-plus-screening-judge-v0",
        "status": "planned_not_run",
        "purpose": "test screening only; not official",
        "target_manifest_sha256": "9" * 64,
        "engine": {
            "name": "vllm_openai_compatible_server",
            "version": "0.9.0.1",
            "host": "127.0.0.1",
            "served_model_name": "qwen3-32b-awq-screening",
            "tensor_parallel_size": 1,
            "gpu_id": 5,
            "gpu_model": "NVIDIA RTX A6000",
            "gpu_memory_mib": 49140,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 16384,
            "max_num_seqs": 32,
            "enable_prefix_caching": True,
        },
        "judge": {
            "model": "Qwen/Qwen3-32B-AWQ",
            "revision": "a" * 40,
            "quantization": "awq_int4",
            "base_model": "Qwen/Qwen3-32B",
        },
        "prompt": {
            "source_repository_commit": "0" * 40,
            "evaluator_script_sha256": "1" * 64,
            "grader_template_sha256": sha256(
                GRADER_TEMPLATE.encode("utf-8")
            ).hexdigest(),
        },
        "inference": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 4096,
            "enable_thinking": False,
        },
        "calibration": {
            "reference_track": "official_qwen3_32b_bf16_two_gpu",
            "reference_evaluations": 6,
            "reference_batch_manifest_path": "runs/batch.json",
            "reference_batch_manifest_sha256": "b" * 64,
            "reference_comparison_path": "runs/comparison.json",
            "reference_comparison_sha256": comparison_hash,
            "reference_execution_result_sha256": "d" * 64,
            "minimum_label_agreement_percent": 90.0,
            "minimum_cohens_kappa": 0.75,
            "maximum_parse_failures": 0,
            "maximum_absolute_pooled_accuracy_delta_pp": 3.0,
            "require_paired_variant_delta_sign_match": True,
        },
        "acceptance": "test acceptance",
        "claim_boundary": "not official",
    }
