from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.gold_evidence_causal_funnel import (
    run_gold_evidence_causal_funnel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct gold/evidence arrival confounding in frozen development traces."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_gold_evidence_causal_funnel(
        registration_path=args.registration,
        output_path=args.output,
    )
    print(f"case_count={result.case_count}")
    for category, count in sorted(result.category_counts.items()):
        print(f"{category}={count}")
    print(f"queues={result.queues.model_dump_json()}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
