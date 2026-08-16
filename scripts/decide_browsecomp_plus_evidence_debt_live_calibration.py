from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.evidence_debt_experiment import (
    decide_evidence_debt_live_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decide a registered known-outcome Evidence-Debt calibration."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-diagnostic", type=Path, required=True)
    parser.add_argument("--candidate-judge-execution", type=Path, required=True)
    parser.add_argument("--candidate-judge-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    result = decide_evidence_debt_live_calibration(
        registration_path=args.registration,
        candidate_summary_path=args.candidate_summary,
        candidate_diagnostic_path=args.candidate_diagnostic,
        candidate_judge_execution_path=args.candidate_judge_execution,
        candidate_judge_result_path=args.candidate_judge_result,
        output_path=args.output,
        validate_existing=args.validate_existing,
    )
    print(f"decision={result.decision}")
    print(f"judge_correct={result.candidate_judge_correct}/{result.query_count}")
    print(f"repair_triggered_regressions={result.repair_triggered_regressions}")
    print(f"repaired_regression_corrections={result.repaired_regression_corrections}")
    print(f"preserved_improvements={result.preserved_improvements}")
    print(f"cost_usd={result.total_cost_usd:.6f}")
    print(f"next_action={result.next_action}")
    return 0 if result.decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
