from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.development_failure_profile import (
    build_development_failure_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the preregistered 175-question failure profile."
    )
    parser.add_argument("--taxonomy-registration", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--gold-diagnostic", type=Path, required=True)
    parser.add_argument("--judge-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_development_failure_profile(
        taxonomy_registration_path=args.taxonomy_registration,
        source_summary_path=args.source_summary,
        gold_diagnostic_path=args.gold_diagnostic,
        judge_result_path=args.judge_result,
        output_path=args.output,
    )
    print(f"query_count={result.query_count}")
    print(f"judge_correct={result.judge_correct}")
    print(f"judge_wrong={result.judge_wrong}")
    print(f"schema_complete={result.schema_complete}")
    print(f"category_counts={result.category_counts}")
    print(f"wrong_category_counts={result.wrong_category_counts}")
    print(f"next_layer={result.next_layer}")
    print(f"multi_agent_status={result.multi_agent_status}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
