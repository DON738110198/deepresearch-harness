from __future__ import annotations

import argparse
import os
from pathlib import Path

from deepresearch_harness.provider_balance import check_deepseek_balance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check DeepSeek balance without making a model inference call."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_deepseek_balance(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        output_path=args.output,
    )
    print(f"provider_reported_available={str(result.provider_reported_available).lower()}")
    print(f"positive_total_balance={str(result.positive_total_balance).lower()}")
    print(f"resume_allowed={str(result.resume_allowed).lower()}")
    print(f"model_inference_calls={result.model_inference_calls}")
    print(f"output={args.output}")
    return 0 if result.resume_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
