from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.atomic_clue_frontier import run_atomic_clue_frontier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a preregistered deterministic atomic-clue retrieval frontier."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    result = run_atomic_clue_frontier(
        registration_path=args.registration,
        output_path=args.output,
        timeout_seconds=args.search_timeout_seconds,
    )
    print(f"decision={result.decision}")
    print(f"search_calls={result.search_calls}")
    print(f"search_failures={result.search_failures}")
    print(f"unique_documents={result.unique_documents}")
    print(f"gold_hit_cases={result.gold_hit_cases}")
    print(f"output={args.output}")
    return 0 if result.decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
