from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.counter_candidate_replay import run_candidate_replay


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay every preserved counter-hypothesis candidate without an LLM call."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    result = run_candidate_replay(
        spec_path=args.spec,
        output_path=args.output,
        timeout_seconds=args.search_timeout_seconds,
    )
    print(f"diagnosis={result.diagnosis}")
    print(f"search_calls={result.search_calls}")
    print(f"source_selected_gold_hit_cases={result.source_selected_gold_hit_cases}")
    print(f"replay_selected_gold_hit_cases={result.replay_selected_gold_hit_cases}")
    print(f"any_candidate_gold_hit_cases={result.any_candidate_gold_hit_cases}")
    print(f"unselected_rescue_cases={result.unselected_rescue_cases}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
