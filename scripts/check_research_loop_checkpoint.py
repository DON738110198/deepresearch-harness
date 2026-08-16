from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.research_loop import load_research_loop_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a bounded Deep Research experiment-loop checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--require-pause-ready", action="store_true")
    args = parser.parse_args()

    checkpoint = load_research_loop_checkpoint(args.checkpoint.resolve())
    audit = checkpoint.pause_audit()
    print(f"loop_id={audit.loop_id}")
    print(f"status={checkpoint.status}")
    print(f"ready_to_pause={str(audit.ready_to_pause).lower()}")
    print(f"blockers={','.join(audit.blockers)}")
    print(f"next_action={audit.next_action}")
    if args.require_pause_ready and not audit.ready_to_pause:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
