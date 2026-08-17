from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.evidence_reachability_funnel import (
    run_evidence_reachability_funnel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Localize downstream evidence failures from frozen traces."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_evidence_reachability_funnel(
        registration_path=args.registration,
        output_path=args.output,
    )
    print(f"case_count={result.case_count}")
    print(f"reference_arrived={result.reference_arrived}")
    print(f"reference_open_capable={result.reference_open_capable}")
    print(f"reference_explicitly_opened={result.reference_explicitly_opened}")
    print(f"answer_visible={result.answer_visible}")
    print(f"answer_hidden={result.answer_hidden}")
    print(
        "answer_visible_reference_uncited="
        f"{result.answer_visible_reference_uncited}"
    )
    print(
        "answer_visible_reference_cited_wrong="
        f"{result.answer_visible_reference_cited_wrong}"
    )
    print(f"next_layer={result.next_layer}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
