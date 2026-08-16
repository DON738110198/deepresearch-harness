from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.tool_health_validation import validate_tool_health_boundary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the fail-closed search dependency boundary without LLM calls."
    )
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate_tool_health_boundary(
        root=root,
        node_executable=args.node,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"checks={len(result.checks)}")
    print(f"provider_calls={result.provider_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={args.output}")
    return 0 if result.decision == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
