from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.obligation_span_experiment import (
    decide_obligation_span_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered obligation-span calibration gates."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide_obligation_span_calibration(
        registration_path=args.registration,
        candidate_summary_path=args.candidate_summary,
        candidate_judge_path=args.candidate_judge,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"opened_span_cases={result.opened_span_cases}")
    print(f"answer_bearing_span_cases={result.answer_bearing_span_cases}")
    print(f"baseline_judge_correct_cases={result.baseline_judge_correct_cases}")
    print(f"candidate_judge_correct_cases={result.candidate_judge_correct_cases}")
    print(f"judge_correct_delta={result.judge_correct_delta}")
    print(f"provider_cost_usd={result.provider_cost_usd:.8f}")
    print(f"output={args.output}")
    return 0 if result.decision == "advance_to_fresh_slice" else 1


if __name__ == "__main__":
    raise SystemExit(main())
