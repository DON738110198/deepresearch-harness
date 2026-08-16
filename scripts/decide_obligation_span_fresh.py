from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.obligation_span_fresh import decide_obligation_span_fresh


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered paired fresh-slice span-opening gates."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-diagnostic", type=Path, required=True)
    parser.add_argument("--baseline-judge", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-diagnostic", type=Path, required=True)
    parser.add_argument("--candidate-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide_obligation_span_fresh(
        registration_path=args.registration,
        baseline_summary_path=args.baseline_summary,
        baseline_diagnostic_path=args.baseline_diagnostic,
        baseline_judge_path=args.baseline_judge,
        candidate_summary_path=args.candidate_summary,
        candidate_diagnostic_path=args.candidate_diagnostic,
        candidate_judge_path=args.candidate_judge,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"baseline_judge_correct={result.baseline_judge_correct}")
    print(f"candidate_judge_correct={result.candidate_judge_correct}")
    print(f"judge_accuracy_delta_pp={result.judge_accuracy_delta_pp}")
    print(f"normalized_exact_delta={result.normalized_exact_delta}")
    print(f"evidence_recall_delta_pp={result.evidence_recall_delta_pp}")
    print(f"paired_improvements={result.paired_improvements}")
    print(f"paired_regressions={result.paired_regressions}")
    print(f"total_token_ratio={result.total_token_ratio}")
    print(f"provider_cost_ratio={result.provider_cost_ratio}")
    print(f"output={args.output}")
    return 0 if result.decision == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
