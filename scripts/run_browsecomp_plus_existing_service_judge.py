from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.screening_judge import (
    load_screening_manifest,
    run_screening_judge,
    validate_service_registration,
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hash-bound judge calibration batch through an existing loopback "
            "vLLM service without reloading its model."
        )
    )
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--service-registration", type=Path, required=True)
    parser.add_argument("--asset-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    judge_path = args.judge_manifest.resolve()
    batch_path = args.batch_manifest.resolve()
    service_path = args.service_registration.resolve()
    verification_path = args.asset_verification.resolve()
    judge = load_screening_manifest(judge_path)
    service, verification = validate_service_registration(
        manifest=judge,
        registration_path=service_path,
        asset_verification_path=verification_path,
    )
    args.output_dir.mkdir(parents=True)
    registration = {
        "schema_version": "browsecomp-plus-existing-service-judge-execution-v0",
        "status": "registered_pre_inference",
        "registered_at": utc_now(),
        "judge_manifest_sha256": normalized_text_file_sha256(judge_path),
        "batch_manifest_sha256": sha256_file(batch_path),
        "service_registration_sha256": sha256_file(service_path),
        "asset_verification_sha256": sha256_file(verification_path),
        "judge_model": judge.judge.model,
        "judge_revision": judge.judge.revision,
        "served_model_name": judge.engine.served_model_name,
        "model_path": service["model_path"],
        "gpu_ids": service["gpu_indices"],
        "service_pid": service["pid"],
        "inference": judge.inference.model_dump(mode="json"),
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "asset_file_count": verification["file_count"],
    }
    registration_path = args.output_dir / "execution_registration.json"
    atomic_json(registration_path, registration)

    started_at = utc_now()
    started = time.perf_counter()
    error = None
    result_path = args.output_dir / "evaluation" / "screening_result.json"
    try:
        run_screening_judge(
            screening_manifest_path=judge_path,
            batch_manifest_path=batch_path,
            output_dir=args.output_dir / "evaluation",
            base_url=args.base_url,
            concurrency=args.concurrency,
            timeout_seconds=args.request_timeout_seconds,
        )
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"

    result = {
        "schema_version": "browsecomp-plus-existing-service-judge-result-v0",
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "succeeded" if error is None else "failed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "registration_sha256": sha256_file(registration_path),
        "judge_result": (
            {
                "path": result_path.relative_to(args.output_dir).as_posix(),
                "sha256": sha256_file(result_path),
            }
            if result_path.is_file()
            else None
        ),
        "error": error,
    }
    atomic_json(args.output_dir / "execution_result.json", result)
    print(f"status={result['status']}")
    print(f"execution_result={args.output_dir / 'execution_result.json'}")
    if error is not None:
        print(f"judge_error={error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
