from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.progressive_disclosure_calibration import (
    calibrate_progressive_disclosure,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate progressive disclosure from frozen saved retrieval traces."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document-index-path", type=Path, required=True)
    parser.add_argument("--snippet-tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact = calibrate_progressive_disclosure(
        registration_path=args.registration,
        output_path=args.output,
        document_index_path=args.document_index_path,
        snippet_tokenizer_dir=args.snippet_tokenizer_dir,
    )
    print(f"status={artifact.status}")
    print(f"query_count={artifact.metrics.query_count}")
    print(f"evidence_recall_delta_pp={artifact.metrics.evidence_recall_delta_pp}")
    print(
        "aggregate_search_ingress_ratio="
        f"{artifact.metrics.aggregate_search_ingress_ratio}"
    )
    print(
        "selected_total_evidence_ingress_token_budget="
        f"{artifact.selected_policy.total_evidence_ingress_token_budget}"
    )
    return 0 if artifact.status == "calibration_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
