from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.screening_judge import (
    load_screening_manifest,
    run_screening_judge,
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str], *, environment: dict[str, str] | None = None) -> str:
    process = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n"
            f"{process.stderr}"
        )
    return process.stdout


def gpu_check(gpu_id: int) -> dict[str, object]:
    rows = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    devices = {}
    for line in rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise ValueError("unexpected nvidia-smi device output")
        devices[int(parts[0])] = {
            "index": int(parts[0]),
            "uuid": parts[1],
            "name": parts[2],
            "memory_used_mib": int(parts[3]),
            "memory_total_mib": int(parts[4]),
            "utilization_percent": int(parts[5]),
            "compute_pids": [],
        }
    applications = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    by_uuid = {row["uuid"]: row for row in devices.values()}
    for line in applications:
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] in by_uuid:
            by_uuid[parts[0]]["compute_pids"].append(int(parts[1]))
    selected = devices.get(gpu_id)
    if selected is None:
        raise ValueError(f"selected GPU does not exist: {gpu_id}")
    if (
        selected["memory_used_mib"] > 512
        or selected["utilization_percent"] > 5
        or selected["compute_pids"]
    ):
        raise ValueError(f"selected GPU is occupied: {gpu_id}")
    return {"captured_at": utc_now(), "device": selected}


def runtime_versions(python: Path, gpu_id: int) -> dict[str, object]:
    marker = "__SCREENING_RUNTIME__"
    snippet = (
        "import json,sys,torch,transformers,vllm;"
        f"print('{marker}'+json.dumps({{'python':sys.version,"
        "'torch':torch.__version__,'transformers':transformers.__version__,"
        "'vllm':vllm.__version__,'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count()}))"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    lines = run_text([str(python), "-c", snippet], environment=environment).splitlines()
    records = [line for line in lines if line.startswith(marker)]
    if len(records) != 1:
        raise ValueError("screening runtime did not emit one version record")
    payload = json.loads(records[0][len(marker) :])
    if payload.get("cuda_available") is not True or payload.get(
        "cuda_device_count"
    ) != 1:
        raise ValueError("screening runtime must see exactly one CUDA device")
    return payload


def build_server_command(
    *,
    python: Path,
    model_dir: Path,
    served_model_name: str,
    port: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    enable_prefix_caching: bool,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_dir),
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--quantization",
        "awq",
        "--dtype",
        "half",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--generation-config",
        "vllm",
    ]
    if enable_prefix_caching:
        command.append("--enable-prefix-caching")
    return command


def wait_for_service(base_url: str, process: subprocess.Popen, timeout: float) -> int:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"{base_url}/models", timeout=5
            ) as response:
                if response.status == 200:
                    return round((time.perf_counter() - started) * 1000)
        except (OSError, urllib.error.HTTPError):
            time.sleep(2)
    raise TimeoutError("vLLM server did not become ready")


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the non-official single-GPU BrowseComp-Plus screening judge."
    )
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--idle-check-interval-seconds", type=float, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    if args.idle_checks < 3:
        parser.error("--idle-checks must be at least 3")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    python = args.python.resolve()
    model_dir = args.model_dir.resolve()
    if not python.is_file():
        parser.error("--python must be an existing interpreter")
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        parser.error("--model-dir must contain config.json")

    screening_path = args.screening_manifest.resolve()
    batch_path = args.batch_manifest.resolve()
    screening = load_screening_manifest(screening_path)
    if args.gpu_id != screening.engine.gpu_id:
        parser.error("--gpu-id differs from the registered screening GPU")
    if model_dir.name != screening.judge.revision:
        parser.error("--model-dir must be the revision-named Hugging Face snapshot")

    checks = []
    for index in range(args.idle_checks):
        checks.append(gpu_check(args.gpu_id))
        if index + 1 < args.idle_checks:
            time.sleep(args.idle_check_interval_seconds)
    runtime = runtime_versions(python, args.gpu_id)
    if runtime["vllm"] != screening.engine.version:
        parser.error("runtime vLLM version differs from screening registration")

    command = build_server_command(
        python=python,
        model_dir=model_dir,
        served_model_name=screening.engine.served_model_name,
        port=args.port,
        gpu_memory_utilization=screening.engine.gpu_memory_utilization,
        max_model_len=screening.engine.max_model_len,
        max_num_seqs=screening.engine.max_num_seqs,
        enable_prefix_caching=screening.engine.enable_prefix_caching,
    )
    args.output_dir.mkdir(parents=True)
    stdout_path = args.output_dir / "vllm.stdout.log"
    stderr_path = args.output_dir / "vllm.stderr.log"
    registration = {
        "schema_version": "browsecomp-plus-screening-judge-execution-v0",
        "status": "registered_pre_inference",
        "registered_at": utc_now(),
        "screening_manifest_sha256": sha256_file(screening_path),
        "batch_manifest_sha256": sha256_file(batch_path),
        "judge_model": screening.judge.model,
        "judge_revision": screening.judge.revision,
        "model_dir": str(model_dir),
        "gpu_id": args.gpu_id,
        "gpu_checks": checks,
        "runtime": runtime,
        "server_command": command,
        "server_host": screening.engine.host,
        "server_port": args.port,
        "concurrency": args.concurrency,
    }
    registration_path = args.output_dir / "execution_registration.json"
    atomic_json(registration_path, registration)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    process = None
    error = None
    started_at = utc_now()
    started = time.perf_counter()
    service_ready_ms = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                env=environment,
                text=True,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            base_url = f"http://127.0.0.1:{args.port}/v1"
            service_ready_ms = wait_for_service(
                base_url, process, args.startup_timeout_seconds
            )
            run_screening_judge(
                screening_manifest_path=screening_path,
                batch_manifest_path=batch_path,
                output_dir=args.output_dir / "evaluation",
                base_url=base_url,
                concurrency=args.concurrency,
                timeout_seconds=args.request_timeout_seconds,
            )
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        if process is not None:
            stop_process_group(process)

    result_path = args.output_dir / "evaluation" / "screening_result.json"
    result = {
        "schema_version": "browsecomp-plus-screening-judge-execution-result-v0",
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "succeeded" if error is None else "failed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "service_ready_ms": service_ready_ms,
        "registration_sha256": sha256_file(registration_path),
        "screening_result": (
            {
                "path": result_path.relative_to(args.output_dir).as_posix(),
                "sha256": sha256_file(result_path),
            }
            if result_path.is_file()
            else None
        ),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "error": error,
    }
    atomic_json(args.output_dir / "execution_result.json", result)
    print(f"status={result['status']}")
    print(f"execution_result={args.output_dir / 'execution_result.json'}")
    if error:
        print(f"screening_error={error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
