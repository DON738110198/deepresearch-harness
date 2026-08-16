from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.evidence_debt_audit_calibration import (
    calibrate_evidence_debt_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the zero-provider-call saved-trace Evidence Debt calibration."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    result = calibrate_evidence_debt_audit(
        registration_path=args.registration.resolve(),
        output_path=args.output.resolve(),
        validate_existing=args.validate_existing,
    )
    print(f"output={args.output}")
    print(f"decision={result.decision}")
    print(
        "regression_trigger_recall="
        f"{result.regression_trigger_count}/{result.regression_count}"
    )
    print(
        "improvement_false_trigger_count="
        f"{result.improvement_false_trigger_count}/{result.improvement_count}"
    )
    print(f"provider_calls={result.provider_calls}")
    print(f"search_calls={result.search_calls}")
    print(f"next_action={result.next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
