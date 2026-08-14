from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.evidence_bandwidth_decision import (
    decide_evidence_bandwidth,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen Evidence Bandwidth confirmation gates to "
            "hash-bound repeat and calibrated-Judge artifacts."
        )
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--repeat-experiment", type=Path, required=True)
    parser.add_argument("--repeat-comparison", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--judge-calibration", type=Path, required=True)
    parser.add_argument("--judge-execution-registration", type=Path, required=True)
    parser.add_argument("--judge-execution-result", type=Path, required=True)
    parser.add_argument("--judge-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()

    decision = decide_evidence_bandwidth(
        preregistration_path=args.preregistration.resolve(),
        candidate_manifest_path=args.candidate_manifest.resolve(),
        repeat_experiment_path=args.repeat_experiment.resolve(),
        repeat_comparison_path=args.repeat_comparison.resolve(),
        target_manifest_path=args.target_manifest.resolve(),
        judge_manifest_path=args.judge_manifest.resolve(),
        judge_calibration_path=args.judge_calibration.resolve(),
        judge_execution_registration_path=(
            args.judge_execution_registration.resolve()
        ),
        judge_execution_result_path=args.judge_execution_result.resolve(),
        judge_comparison_path=args.judge_comparison.resolve(),
        output_path=args.output.resolve(),
        validate_existing=args.validate_existing,
    )
    print(f"output={args.output}")
    print(f"decision={decision.decision}")
    print(f"evidence_recall_delta_pp={decision.evidence_recall_delta_pp}")
    print(f"judge_accuracy_delta_pp={decision.judge_accuracy_delta_pp}")
    print(
        "candidate_to_baseline_total_token_ratio="
        f"{decision.candidate_to_baseline_total_token_ratio}"
    )
    print(
        "candidate_to_baseline_provider_cost_ratio="
        f"{decision.candidate_to_baseline_provider_cost_ratio}"
    )
    print(f"combined_provider_cost_usd={decision.combined_provider_cost_usd}")
    print(f"next_action={decision.next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
