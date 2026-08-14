from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.tool_loop_governor_decision import (
    decide_tool_loop_governor_fresh5,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Tool-Loop Governor fresh-5 gates."
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-diagnostic", type=Path, required=True)
    parser.add_argument("--judge-execution-result", type=Path, required=True)
    parser.add_argument("--judge-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    result = decide_tool_loop_governor_fresh5(
        preregistration_path=args.preregistration.resolve(),
        candidate_summary_path=args.candidate_summary.resolve(),
        candidate_diagnostic_path=args.candidate_diagnostic.resolve(),
        judge_execution_result_path=args.judge_execution_result.resolve(),
        judge_result_path=args.judge_result.resolve(),
        output_path=args.output.resolve(),
        validate_existing=args.validate_existing,
    )
    print(f"output={args.output}")
    print(f"decision={result.decision}")
    print(f"judge_correct={result.candidate_judge_correct}/{result.query_count}")
    print(f"evidence_recall_delta_pp={result.evidence_recall_delta_pp}")
    print(
        "candidate_to_baseline_total_token_ratio="
        f"{result.candidate_to_baseline_total_token_ratio}"
    )
    print(
        "candidate_to_baseline_provider_cost_ratio="
        f"{result.candidate_to_baseline_provider_cost_ratio}"
    )
    print(f"next_action={result.next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
