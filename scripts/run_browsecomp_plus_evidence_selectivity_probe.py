from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.evidence_selectivity_probe import (
    run_evidence_selectivity_probe,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved paired traces for evidence selectivity and synthesis "
            "loss without making provider calls."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_evidence_selectivity_probe(
        registration_path=args.registration.resolve(),
        target_manifest_path=args.target_manifest.resolve(),
        output_path=args.output.resolve(),
    )
    print(f"output={args.output}")
    print(f"provider_calls={result.provider_calls}")
    print(f"paired_observations={result.paired_observations}")
    for group in result.groups:
        print(
            f"{group.paired_outcome}:n={group.observations}:"
            f"evidence_delta={group.mean_evidence_recall_delta}:"
            f"candidate_duplicate_rate="
            f"{group.candidate_mean_duplicate_result_rate_percent}:"
            f"candidate_repeated_query_rate="
            f"{group.candidate_mean_repeated_query_rate_percent}"
        )
    print(f"next_candidate={result.next_candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
