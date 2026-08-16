from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.span_opening_resolution import (
    resolve_span_opening_valid_trials,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the two valid span-opening trials after an invalid execution."
    )
    parser.add_argument("--fresh-decision", type=Path, required=True)
    parser.add_argument("--trial2-baseline-summary", type=Path, required=True)
    parser.add_argument("--trial2-candidate-summary", type=Path, required=True)
    parser.add_argument("--trial2-baseline-diagnostic", type=Path, required=True)
    parser.add_argument("--trial2-candidate-diagnostic", type=Path, required=True)
    parser.add_argument("--trial2-baseline-judge", type=Path, required=True)
    parser.add_argument("--trial2-candidate-judge", type=Path, required=True)
    parser.add_argument("--execution-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = resolve_span_opening_valid_trials(
        fresh_decision_path=args.fresh_decision,
        trial2_baseline_summary_path=args.trial2_baseline_summary,
        trial2_candidate_summary_path=args.trial2_candidate_summary,
        trial2_baseline_diagnostic_path=args.trial2_baseline_diagnostic,
        trial2_candidate_diagnostic_path=args.trial2_candidate_diagnostic,
        trial2_baseline_judge_path=args.trial2_baseline_judge,
        trial2_candidate_judge_path=args.trial2_candidate_judge,
        execution_audit_path=args.execution_audit,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"aggregate_judge_correct_delta={result.aggregate_judge_correct_delta}")
    print(f"aggregate_normalized_exact_delta={result.aggregate_normalized_exact_delta}")
    print(f"mean_evidence_recall_delta_pp={result.mean_evidence_recall_delta_pp}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
