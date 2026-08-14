from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.wide_selector_probe import run_wide_selector_probe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Screen fixed top-5 selectors over frozen BM25/dense top-20 rankings "
            "without making LLM provider calls."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dense-index-root", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    result = run_wide_selector_probe(
        registration_path=args.registration,
        model_dir=args.model_dir,
        dense_index_root=args.dense_index_root,
        bm25_index_path=args.bm25_index,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    metrics = {
        row.selector_id: {
            "evidence_recall_percent": row.evidence_recall_percent,
            "delta_pp": row.evidence_delta_vs_bm25_pp,
            "wins_minus_losses": row.query_wins_minus_losses,
            "passed": row.passed,
        }
        for row in result.selector_metrics
    }
    print(f"output={args.output}")
    print("provider_calls=0")
    print(f"decision={result.decision}")
    print(f"selected_selector_id={result.selected_selector_id}")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
