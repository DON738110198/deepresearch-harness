from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.evidence_debt_experiment import (
    decide_evidence_debt_fresh_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide a fresh paired Pi v10 versus v11 comparison.")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-diagnostic", type=Path, required=True)
    parser.add_argument("--baseline-judge-execution", type=Path, required=True)
    parser.add_argument("--baseline-judge-result", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-diagnostic", type=Path, required=True)
    parser.add_argument("--candidate-judge-execution", type=Path, required=True)
    parser.add_argument("--candidate-judge-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    result = decide_evidence_debt_fresh_comparison(
        registration_path=args.registration,
        baseline_summary_path=args.baseline_summary,
        baseline_diagnostic_path=args.baseline_diagnostic,
        baseline_judge_execution_path=args.baseline_judge_execution,
        baseline_judge_result_path=args.baseline_judge_result,
        candidate_summary_path=args.candidate_summary,
        candidate_diagnostic_path=args.candidate_diagnostic,
        candidate_judge_execution_path=args.candidate_judge_execution,
        candidate_judge_result_path=args.candidate_judge_result,
        output_path=args.output,
        validate_existing=args.validate_existing,
    )
    print(f"decision={result.decision}")
    print(
        "judge_accuracy="
        f"{result.baseline_judge_accuracy_percent}% -> "
        f"{result.candidate_judge_accuracy_percent}%"
    )
    print(f"paired={result.paired_improvements} improvements/{result.paired_regressions} regressions")
    print(
        "ratios="
        f"search:{result.search_call_ratio:.3f},"
        f"tokens:{result.total_token_ratio:.3f},"
        f"cost:{result.provider_cost_ratio:.3f}"
    )
    print(f"next_action={result.next_action}")
    return 0 if result.decision == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())
