from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.repeat_execution_audit import run_repeat_execution_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a repeat arm whose search transport failed before scoring."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_repeat_execution_audit(
        registration_path=args.registration,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"transport_failure_calls={result.transport_failure_calls}")
    print(f"successful_search_calls={result.successful_search_calls}")
    print(f"recorded_invalid_arm_provider_cost_usd={result.recorded_invalid_arm_provider_cost_usd}")
    print(f"output={args.output}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
