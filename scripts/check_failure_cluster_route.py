from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.research_loop import load_failure_cluster_route


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a failure-cluster stop route and its rejected checkpoints."
    )
    parser.add_argument("route", type=Path)
    args = parser.parse_args()

    route = load_failure_cluster_route(args.route)
    audit = route.audit()
    print(f"cluster_id={audit.cluster_id}")
    print(f"status={audit.status}")
    print(f"selection_allowed={str(audit.selection_allowed).lower()}")
    print(f"same_cluster_rejections={audit.same_cluster_rejections}")
    print(f"prior_analogue_rejections={audit.prior_analogue_rejections}")
    print(f"next_action={audit.next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

