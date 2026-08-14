from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.dense_depth_probe import run_dense_depth_probe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen BrowseComp-Plus search queries at registered dense "
            "depths without making LLM provider calls."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = run_dense_depth_probe(
        registration_path=args.registration,
        target_manifest_path=args.target_manifest,
        model_dir=args.model_dir,
        index_root=args.index_root,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    metrics = {
        str(metric.depth): {
            "evidence_recall_percent": metric.evidence_recall_percent,
            "delta_pp": metric.evidence_delta_vs_bm25_pp,
            "no_relevant_doc_queries": metric.no_relevant_doc_queries,
        }
        for metric in result.depth_metrics
    }
    print(f"output={args.output}")
    print("provider_calls=0")
    print(f"decision={result.decision}")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
