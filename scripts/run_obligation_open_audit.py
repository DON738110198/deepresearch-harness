from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.obligation_open_audit import run_obligation_open_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit fresh span-opening flips without provider or search calls."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_obligation_open_audit(
        registration_path=args.registration,
        output_path=args.output,
    )
    print(f"diagnosis={result.diagnosis}")
    print(f"improvements={result.improvement_cases}")
    print(f"regressions={result.regression_cases}")
    print(f"supported_improvements={result.supported_improvement_cases}")
    print(f"regressions_with_successful_open={result.regressions_with_successful_open}")
    print(f"answer_bearing_gold_open_cases={result.answer_bearing_gold_open_cases}")
    print(f"total_search_call_delta={result.total_search_call_delta}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
