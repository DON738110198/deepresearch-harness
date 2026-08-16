from __future__ import annotations

import argparse
import os
from pathlib import Path

from deepresearch_harness.development_profile import preflight_development_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight for the frozen 175-question profile."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-existing-for-resume", action="store_true")
    args = parser.parse_args()

    result = preflight_development_profile(
        registration_path=args.registration,
        output_path=args.output,
        environment=os.environ,
        allow_existing_for_resume=args.allow_existing_for_resume,
    )
    print(f"route_status={result.route_status}")
    print(f"query_count={result.query_count}")
    print(f"model={result.model}")
    print(f"retriever_id={result.retriever_id}")
    print(f"health_url={result.health_url}")
    print(f"provider_key_env={result.provider_key_env}")
    print(f"provider_key_present={str(result.provider_key_present).lower()}")
    print(f"output_mode={result.output_mode}")
    print(f"provider_calls={result.provider_calls}")
    print(f"sealed_holdout_accessed={str(result.sealed_holdout_accessed).lower()}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

