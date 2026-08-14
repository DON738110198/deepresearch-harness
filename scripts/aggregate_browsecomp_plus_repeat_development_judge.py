from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.repeat_development_judge import (
    aggregate_repeat_development_judge,
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
            "Recover only the hash-bound aggregation of already completed repeat "
            "Judge results; this command makes no model requests."
        )
    )
    parser.add_argument("--repeat-experiment", type=Path, required=True)
    parser.add_argument("--repeat-comparison", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--source-execution-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")

    source_execution_path = args.source_execution_result.resolve()
    source_execution = json.loads(source_execution_path.read_text(encoding="utf-8"))
    if (
        not isinstance(source_execution, dict)
        or source_execution.get("status") != "failed"
        or source_execution.get("comparison") is not None
    ):
        parser.error("source execution must be a failed pre-comparison run")
    result_paths = sorted(
        args.results_root.resolve().glob("*/development_judge_result.json")
    )
    if len(result_paths) != 6:
        parser.error("--results-root must contain exactly six variant results")

    args.output_dir.mkdir(parents=True)
    registration = {
        "schema_version": (
            "browsecomp-plus-repeat-development-judge-aggregation-recovery-v0"
        ),
        "status": "registered_pre_aggregation_recovery",
        "registered_at": utc_now(),
        "metric_status": "calibrated_development_diagnostic_not_official",
        "provider_calls": 0,
        "source_execution_result_sha256": sha256_file(source_execution_path),
        "source_trial_results": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
            }
            for path in result_paths
        ],
        "repeat_experiment_sha256": sha256_file(
            args.repeat_experiment.resolve()
        ),
        "repeat_comparison_sha256": sha256_file(
            args.repeat_comparison.resolve()
        ),
        "judge_manifest_sha256": sha256_file(args.judge_manifest.resolve()),
        "calibration_sha256": sha256_file(args.calibration_result.resolve()),
    }
    registration_path = args.output_dir / "execution_registration.json"
    atomic_json(registration_path, registration)

    started = time.perf_counter()
    comparison_path = args.output_dir / "repeat_development_judge_comparison.json"
    error = None
    comparison = None
    try:
        comparison = aggregate_repeat_development_judge(
            repeat_experiment_path=args.repeat_experiment.resolve(),
            repeat_comparison_path=args.repeat_comparison.resolve(),
            target_manifest_path=args.target_manifest.resolve(),
            judge_manifest_path=args.judge_manifest.resolve(),
            calibration_result_path=args.calibration_result.resolve(),
            results_root=args.results_root.resolve(),
            output_path=comparison_path,
        )
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"

    execution = {
        "schema_version": (
            "browsecomp-plus-repeat-development-judge-aggregation-execution-v0"
        ),
        "completed_at": utc_now(),
        "status": "succeeded" if error is None else "failed",
        "provider_calls": 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "registration_sha256": sha256_file(registration_path),
        "comparison": (
            {
                "path": comparison_path.relative_to(args.output_dir).as_posix(),
                "sha256": sha256_file(comparison_path),
            }
            if comparison_path.is_file()
            else None
        ),
        "error": error,
    }
    execution_path = args.output_dir / "execution_result.json"
    atomic_json(execution_path, execution)
    print(f"status={execution['status']}")
    print("provider_calls=0")
    print(f"execution_result={execution_path}")
    if comparison is not None:
        print(f"evaluations={comparison.evaluations}")
        print(f"baseline_accuracy={comparison.baseline.pooled_accuracy_percent}")
        print(f"candidate_accuracy={comparison.candidate.pooled_accuracy_percent}")
    if error is not None:
        print(f"aggregation_error={error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
