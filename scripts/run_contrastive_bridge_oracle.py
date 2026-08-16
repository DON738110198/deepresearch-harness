from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.bridge_hypothesis_probe import run_oracle_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a labeled post-hoc bridge-query oracle diagnostic.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    result = run_oracle_replay(
        spec_path=args.spec,
        output_path=args.output,
        timeout_seconds=args.search_timeout_seconds,
    )
    print(f"status={result.status}")
    print(f"queries={result.query_count}")
    print(f"search_calls={result.search_calls}")
    print(f"gold_hit_cases={result.gold_hit_cases}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
