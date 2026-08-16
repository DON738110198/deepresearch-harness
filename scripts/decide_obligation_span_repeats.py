from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.obligation_span_repeats import (
    decide_obligation_span_repeats,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered no-change span-opening repeat gates."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide_obligation_span_repeats(
        registration_path=args.registration,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"judge_correct_delta={result.judge_correct_delta}")
    print(f"noninferior_judge_trials={result.noninferior_judge_trials}")
    print(f"normalized_exact_delta={result.normalized_exact_delta}")
    print(f"mean_evidence_recall_delta_pp={result.mean_evidence_recall_delta_pp}")
    print(f"total_token_ratio={result.total_token_ratio}")
    print(f"new_recorded_provider_cost_usd={result.new_recorded_provider_cost_usd}")
    print(f"output={args.output}")
    return 0 if result.decision == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
