from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.baseline_stability_audit import run_baseline_stability_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Localize stable v10 failures across three valid development runs."
    )
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--diagnostic", type=Path, action="append", required=True)
    parser.add_argument("--judge", type=Path, action="append", required=True)
    parser.add_argument("--gold-slice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (len(args.summary) == len(args.diagnostic) == len(args.judge) == 3):
        parser.error("exactly three --summary, --diagnostic, and --judge paths are required")
    result = run_baseline_stability_audit(
        summary_paths=tuple(args.summary),
        diagnostic_paths=tuple(args.diagnostic),
        judge_paths=tuple(args.judge),
        gold_slice_path=args.gold_slice,
        output_path=args.output,
    )
    print(f"query_count={result.query_count}")
    print(f"category_counts={result.category_counts}")
    print(f"actionable_case_ids={','.join(result.actionable_case_ids)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
