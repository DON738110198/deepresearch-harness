from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.evidence_preview_calibration import (
    calibrate_evidence_preview,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate query-aware dense-lead previews from saved bad cases."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--gold-slice", type=Path, required=True)
    parser.add_argument("--document-index", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_evidence_preview(
        source_dir=args.source_dir.resolve(),
        gold_slice_path=args.gold_slice.resolve(),
        document_index_path=args.document_index.resolve(),
        tokenizer_dir=args.tokenizer_dir.resolve(),
        output_path=args.output.resolve(),
    )
    print(f"output={args.output}")
    print(f"status={result.status}")
    print(f"relevant_dense_hits={result.relevant_dense_hit_count}")
    print(
        "baseline_selectable_relevant_hits="
        f"{result.baseline_selectable_relevant_hits}"
    )
    print(
        "candidate_selectable_relevant_hits="
        f"{result.candidate_selectable_relevant_hits}"
    )
    print(
        "candidate_maximum_search_ingress_tokens="
        f"{result.candidate_maximum_search_ingress_tokens}"
    )
    print(f"selected_policy={result.selected_policy}")
    return 0 if result.status == "calibration_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
