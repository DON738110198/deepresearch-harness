from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.query_aware_confirmation_decision import (
    decide_query_aware_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the frozen paired-25 Query-Aware Preview gates."
    )
    parser.add_argument("--preregistration", type=Path, required=True)
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
    result = decide_query_aware_confirmation(
        preregistration_path=args.preregistration.resolve(),
        baseline_summary_path=args.baseline_summary.resolve(),
        baseline_diagnostic_path=args.baseline_diagnostic.resolve(),
        baseline_judge_execution_path=args.baseline_judge_execution.resolve(),
        baseline_judge_result_path=args.baseline_judge_result.resolve(),
        candidate_summary_path=args.candidate_summary.resolve(),
        candidate_diagnostic_path=args.candidate_diagnostic.resolve(),
        candidate_judge_execution_path=args.candidate_judge_execution.resolve(),
        candidate_judge_result_path=args.candidate_judge_result.resolve(),
        output_path=args.output.resolve(),
        validate_existing=args.validate_existing,
    )
    print(f"output={args.output}")
    print(f"decision={result.decision}")
    print(
        "judge_accuracy="
        f"{result.baseline_judge_accuracy_percent:.2f}% -> "
        f"{result.candidate_judge_accuracy_percent:.2f}% "
        f"({result.judge_accuracy_delta_pp:+.2f} pp)"
    )
    print(
        "evidence_recall="
        f"{result.baseline_evidence_recall_percent:.2f}% -> "
        f"{result.candidate_evidence_recall_percent:.2f}% "
        f"({result.evidence_recall_delta_pp:+.2f} pp)"
    )
    print(
        "ratios="
        f"search:{result.candidate_to_baseline_search_call_ratio:.3f},"
        f"tokens:{result.candidate_to_baseline_total_token_ratio:.3f},"
        f"cost:{result.candidate_to_baseline_provider_cost_ratio:.3f}"
    )
    print(f"combined_generation_cost_usd={result.combined_generation_cost_usd:.6f}")
    print(f"next_action={result.next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
