from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.document_target_oracle import run_document_target_oracle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a preregistered no-new-call document-target oracle."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_document_target_oracle(
        registration_path=args.registration,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"gold_target_hit_cases={result.gold_target_hit_cases}")
    print(f"total_selected_targets={result.total_selected_targets}")
    print(
        "maximum_selected_targets_in_case="
        f"{result.maximum_selected_targets_in_case}"
    )
    print(f"provider_calls={result.provider_calls}")
    print(f"new_search_calls={result.new_search_calls}")
    print(f"document_open_calls={result.document_open_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={args.output}")
    return 0 if result.decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
