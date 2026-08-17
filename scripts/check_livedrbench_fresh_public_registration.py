from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.livedrbench_fresh_public import (
    validate_fresh_public_dataset,
    validate_fresh_public_registration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the static LiveDRBench fresh-public registration."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument(
        "--verify-pinned-dataset",
        action="store_true",
        help="Fetch public pinned rows and verify their response hash and categories.",
    )
    args = parser.parse_args()
    registration = validate_fresh_public_registration(args.registration)
    print(f"benchmark_id={registration.benchmark_id}")
    print("selected_task_keys=" + ",".join(map(str, registration.selected_task_keys)))
    print(f"selected_task_keys_sha256={registration.selected_task_keys_sha256}")
    print(f"candidate_status={registration.candidate.status}")
    print(f"comparison_mode={registration.comparison_mode}")
    print(
        "candidate_search_budget="
        f"{registration.budget.candidate_max_search_credits_per_task} credits,"
        f"${registration.budget.candidate_max_search_cost_usd_per_task:.3f}/task"
    )
    print("provider_calls_before_generation=0")
    if args.verify_pinned_dataset:
        tasks = validate_fresh_public_dataset(registration)
        print("pinned_dataset_verified_keys=" + ",".join(str(task.key) for task in tasks))
    else:
        print("pinned_dataset_verification=skipped_offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
