from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def resolve_relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"missing or unsafe artifact: {value}")
    return resolved


def validate_batch(batch_manifest_path: Path) -> tuple[dict, dict, Path, Path]:
    raw = batch_manifest_path.read_bytes()
    batch = json.loads(raw)
    if (
        batch.get("schema_version") != "browsecomp-plus-official-judge-batch-v0"
        or batch.get("status") != "prepared_not_run"
    ):
        raise ValueError("unsupported official judge batch")
    batch_root = batch_manifest_path.resolve().parent
    items = batch.get("items")
    if not isinstance(items, list) or len(items) != batch.get("input_count"):
        raise ValueError("official judge batch item count mismatch")
    for item in items:
        path = resolve_relative(batch_root, item["staged_input_path"])
        if sha256_file(path) != item["staged_input_sha256"]:
            raise ValueError("official judge staged input hash mismatch")
    ground_truth = resolve_relative(batch_root, batch["ground_truth_path"])
    if sha256_file(ground_truth) != batch["ground_truth_sha256"]:
        raise ValueError("official judge ground-truth hash mismatch")

    audit_by_role = {row["role"]: row for row in batch.get("audit_artifacts", [])}
    evaluator_ref = audit_by_role.get("official-evaluator")
    assets_ref = audit_by_role.get("judge-assets")
    if evaluator_ref is None or assets_ref is None:
        raise ValueError("official judge batch lacks evaluator audit artifacts")
    evaluator_path = resolve_relative(batch_root, evaluator_ref["path"])
    assets_path = resolve_relative(batch_root, assets_ref["path"])
    if sha256_file(evaluator_path) != evaluator_ref["sha256"]:
        raise ValueError("official evaluator audit hash mismatch")
    if sha256_file(assets_path) != assets_ref["sha256"]:
        raise ValueError("official judge assets audit hash mismatch")
    if sha256_file(evaluator_path) != batch["official_evaluator_manifest_sha256"]:
        raise ValueError("official evaluator batch hash mismatch")
    if sha256_file(assets_path) != batch["judge_assets_manifest_sha256"]:
        raise ValueError("official judge asset-manifest batch hash mismatch")
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    expected = {
        "repository_commit": batch["repository_commit"],
        "evaluator_script": batch["evaluator_script"],
        "evaluator_script_sha256": batch["evaluator_script_sha256"],
        "uv_lock_sha256": batch["uv_lock_sha256"],
        "judge_assets_sha256": batch["judge_assets_manifest_sha256"],
    }
    for field, value in expected.items():
        if evaluator.get(field) != value:
            raise ValueError(f"official evaluator audit changes {field}")
    if (
        evaluator.get("judge", {}).get("name") != batch["judge_model"]
        or evaluator.get("judge", {}).get("revision") != batch["judge_revision"]
        or evaluator.get("inference") != batch["inference"]
    ):
        raise ValueError("official evaluator judge contract differs from batch")
    return batch, evaluator, batch_root, ground_truth


def validate_repository(repository: Path, batch: dict) -> None:
    head = run_text(["git", "rev-parse", "HEAD"], cwd=repository).strip()
    if head != batch["repository_commit"]:
        raise ValueError("official evaluator repository commit mismatch")
    if run_text(["git", "status", "--porcelain"], cwd=repository).strip():
        raise ValueError("official evaluator repository is dirty")
    evaluator_script = repository / batch["evaluator_script"]
    if sha256_file(evaluator_script) != batch["evaluator_script_sha256"]:
        raise ValueError("official evaluator script hash mismatch")
    if sha256_file(repository / "uv.lock") != batch["uv_lock_sha256"]:
        raise ValueError("official evaluator uv.lock hash mismatch")


def validate_model_assets(
    verification_path: Path, model_dir: Path, batch: dict
) -> str:
    verification_hash = sha256_file(verification_path)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    files = verification.get("files")
    if (
        verification.get("schema_version")
        != "browsecomp-plus-official-judge-asset-verification-v0"
        or verification.get("passed") is not True
        or verification.get("model") != batch["judge_model"]
        or verification.get("revision") != batch["judge_revision"]
        or verification.get("manifest_sha256")
        != batch["judge_assets_manifest_sha256"]
        or not isinstance(files, list)
        or verification.get("matched") != verification.get("file_count")
        or verification.get("file_count") != len(files)
    ):
        raise ValueError("official judge asset verification is invalid")
    if Path(verification.get("model_dir", "")).resolve() != model_dir.resolve():
        raise ValueError("official judge asset verification targets another model path")
    protected_directories = {model_dir.resolve()}
    for row in files:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row.get("matches") is not True:
            raise ValueError("official judge asset verification contains an unsafe row")
        path = (model_dir / relative).resolve()
        if not path.is_relative_to(model_dir.resolve()) or not path.is_file():
            raise ValueError("official judge verified model asset is missing")
        if path.stat().st_size != row["expected_size"]:
            raise ValueError("official judge verified model asset size changed")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("official judge model assets must be write-protected after hashing")
        protected_directories.add(path.parent)
    for directory in protected_directories:
        if directory.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("official judge model directories must be write-protected")
    return verification_hash


def runtime_versions(python: Path) -> dict:
    marker = "__DEEPRESEARCH_RUNTIME__"
    snippet = (
        "import json,sys,torch,transformers,vllm;"
        f"print('{marker}'+json.dumps({{'python':sys.version,'torch':torch.__version__,"
        "'transformers':transformers.__version__,'vllm':vllm.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count()}))"
    )
    stdout = run_text([str(python), "-c", snippet])
    runtime_lines = [line for line in stdout.splitlines() if line.startswith(marker)]
    if len(runtime_lines) != 1:
        raise ValueError("official judge runtime did not emit one version record")
    payload = json.loads(runtime_lines[0][len(marker) :])
    if payload.get("cuda_available") is not True or payload.get("cuda_device_count", 0) < 2:
        raise ValueError("official judge runtime cannot see two CUDA devices")
    return payload


def gpu_check(gpu_ids: tuple[int, int]) -> dict:
    gpu_rows = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    by_index: dict[int, dict] = {}
    for line in gpu_rows:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError("unexpected nvidia-smi GPU output")
        index = int(parts[0])
        by_index[index] = {
            "index": index,
            "uuid": parts[1],
            "memory_used_mib": int(parts[2]),
            "memory_total_mib": int(parts[3]),
            "utilization_percent": int(parts[4]),
            "compute_pids": [],
        }
    app_rows = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    by_uuid = {row["uuid"]: row for row in by_index.values()}
    for line in app_rows:
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] in by_uuid:
            by_uuid[parts[0]]["compute_pids"].append(int(parts[1]))
    selected = []
    for gpu_id in gpu_ids:
        row = by_index.get(gpu_id)
        if row is None:
            raise ValueError(f"selected GPU does not exist: {gpu_id}")
        if (
            row["memory_used_mib"] > 512
            or row["utilization_percent"] > 5
            or row["compute_pids"]
        ):
            raise ValueError(f"selected GPU is occupied: {gpu_id}")
        selected.append(row)
    return {"captured_at": utc_now(), "devices": selected}


def build_command(
    *,
    python: Path,
    repository: Path,
    batch: dict,
    batch_root: Path,
    ground_truth: Path,
    model_dir: Path,
    eval_dir: Path,
) -> list[str]:
    inference = batch["inference"]
    return [
        str(python),
        str((repository / batch["evaluator_script"]).resolve()),
        "--input_dir",
        str((batch_root / "inputs").resolve()),
        "--ground_truth",
        str(ground_truth.resolve()),
        "--eval_dir",
        str(eval_dir.resolve()),
        "--model",
        str(model_dir.resolve()),
        "--temperature",
        str(inference["temperature"]),
        "--top_p",
        str(inference["top_p"]),
        "--top_k",
        str(inference["top_k"]),
        "--max_output_tokens",
        str(inference["max_output_tokens"]),
        "--batch_size",
        str(batch["input_count"]),
        "--tensor_parallel_size",
        "2",
        "--qrel_evidence",
        str((repository / "topics-qrels/qrel_evidence.txt").resolve()),
    ]


def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result.stdout


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed launcher for the pinned BrowseComp-Plus Qwen3 judge."
    )
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--official-repository", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs=2, required=True)
    parser.add_argument(
        "--disable-nccl-p2p",
        action="store_true",
        help=(
            "Set NCCL_P2P_DISABLE=1 for machines whose peer transport hangs "
            "during communicator initialization. The override is audited."
        ),
    )
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--idle-check-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.idle_checks < 3:
        parser.error("--idle-checks must be at least 3")
    if args.idle_check_interval_seconds < 1:
        parser.error("--idle-check-interval-seconds must be at least 1")
    gpu_ids = tuple(args.gpu_ids)
    if len(set(gpu_ids)) != 2:
        parser.error("--gpu-ids must contain two distinct IDs")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    python_path = Path(os.path.abspath(args.python))
    if not python_path.is_file():
        parser.error("--python must point to the official environment interpreter")

    batch_manifest_path = args.batch_manifest.resolve()
    asset_verification_path = args.asset_verification.resolve()
    batch, evaluator, batch_root, ground_truth = validate_batch(batch_manifest_path)
    validate_repository(args.official_repository.resolve(), batch)
    asset_verification_hash = validate_model_assets(
        asset_verification_path, args.model_dir.resolve(), batch
    )
    gpu_check(gpu_ids)
    runtime = runtime_versions(python_path)
    checks = []
    for index in range(args.idle_checks):
        checks.append(gpu_check(gpu_ids))
        if index + 1 < args.idle_checks:
            time.sleep(args.idle_check_interval_seconds)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    launcher_copy = args.output_dir / "audit" / Path(__file__).name
    atomic_bytes(launcher_copy, Path(__file__).read_bytes())
    verification_copy = args.output_dir / "audit" / "judge_asset_verification.json"
    atomic_bytes(verification_copy, asset_verification_path.read_bytes())
    eval_dir = args.output_dir / "evals"
    command = build_command(
        python=python_path,
        repository=args.official_repository.resolve(),
        batch=batch,
        batch_root=batch_root,
        ground_truth=ground_truth,
        model_dir=args.model_dir.resolve(),
        eval_dir=eval_dir,
    )
    environment_overrides = (
        {"NCCL_P2P_DISABLE": "1"} if args.disable_nccl_p2p else {}
    )
    registration = {
        "schema_version": "browsecomp-plus-official-judge-execution-v0",
        "registered_at": utc_now(),
        "status": "registered_pre_inference",
        "batch_manifest_sha256": sha256_file(batch_manifest_path),
        "official_evaluator_manifest_sha256": batch[
            "official_evaluator_manifest_sha256"
        ],
        "launcher_path": launcher_copy.relative_to(args.output_dir).as_posix(),
        "launcher_sha256": sha256_file(launcher_copy),
        "asset_verification_path": verification_copy.relative_to(
            args.output_dir
        ).as_posix(),
        "asset_verification_sha256": asset_verification_hash,
        "judge_assets_manifest_sha256": batch["judge_assets_manifest_sha256"],
        "repository_commit": batch["repository_commit"],
        "repository_clean": True,
        "evaluator_script_sha256": batch["evaluator_script_sha256"],
        "uv_lock_sha256": batch["uv_lock_sha256"],
        "judge_model": batch["judge_model"],
        "judge_revision": batch["judge_revision"],
        "model_dir": str(args.model_dir.resolve()),
        "runtime": runtime,
        "gpu_ids": list(gpu_ids),
        "gpu_checks": checks,
        "inference": evaluator["inference"],
        "batch_size": batch["input_count"],
        "tensor_parallel_size": 2,
        "environment_overrides": environment_overrides,
        "command": command,
    }
    registration_path = args.output_dir / "execution_registration.json"
    atomic_json(registration_path, registration)

    stdout_path = args.output_dir / "evaluator.stdout.log"
    stderr_path = args.output_dir / "evaluator.stderr.log"
    started_at = utc_now()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    environment.update(environment_overrides)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            command,
            cwd=args.official_repository.resolve(),
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    output_files = [
        {
            "path": path.relative_to(args.output_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(eval_dir.rglob("*"))
        if path.is_file()
    ]
    result = {
        "schema_version": "browsecomp-plus-official-judge-execution-result-v0",
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "succeeded" if process.returncode == 0 else "failed",
        "exit_code": process.returncode,
        "registration_sha256": sha256_file(registration_path),
        "batch_manifest_sha256": sha256_file(batch_manifest_path),
        "stdout": {
            "path": stdout_path.relative_to(args.output_dir).as_posix(),
            "sha256": sha256_file(stdout_path),
        },
        "stderr": {
            "path": stderr_path.relative_to(args.output_dir).as_posix(),
            "sha256": sha256_file(stderr_path),
        },
        "output_files": output_files,
    }
    atomic_json(args.output_dir / "execution_result.json", result)
    print(f"status={result['status']}")
    print(f"exit_code={process.returncode}")
    print(f"registration={registration_path}")
    print(f"execution_result={args.output_dir / 'execution_result.json'}")
    return process.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"preflight_error={error}", file=sys.stderr)
        raise SystemExit(2)
