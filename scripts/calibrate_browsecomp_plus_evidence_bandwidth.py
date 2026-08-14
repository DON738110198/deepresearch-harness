from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.evidence_bandwidth import calibrate_evidence_bandwidth


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a top-20 evidence payload against stored top-5 payload "
            "tokens without making LLM provider calls."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--document-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = calibrate_evidence_bandwidth(
        registration_path=args.registration,
        tokenizer_dir=args.tokenizer_dir,
        document_index_path=args.document_index,
        output_path=args.output,
    )
    metrics = {
        str(row.total_snippet_token_budget): {
            "aggregate_ratio": row.aggregate_payload_token_ratio,
            "maximum_trial_ratio": row.maximum_trial_payload_token_ratio,
            "passed": row.passed,
        }
        for row in result.budgets
    }
    print(f"output={args.output}")
    print("provider_calls=0")
    print(f"decision={result.decision}")
    print(
        "selected_total_snippet_token_budget="
        f"{result.selected_total_snippet_token_budget}"
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
