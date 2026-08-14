from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.browsecomp_plus import (
    DevelopmentQueryArtifact,
    load_development_queries,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a deterministic gold-free hash slice from development queries."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="Development-query artifact whose IDs must not be selected; repeatable.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    source = load_development_queries(args.source)
    excluded_ids = {
        row.query_id
        for path in args.exclude
        for row in load_development_queries(path).queries
    }
    selected = sorted(
        (row for row in source.queries if row.query_id not in excluded_ids),
        key=lambda row: sha256(
            f"{args.prefix}:{row.query_id}".encode("utf-8")
        ).hexdigest(),
    )[: args.limit]
    if len(selected) != args.limit:
        raise ValueError("source query artifact is smaller than the requested slice")
    canonical = "\n".join(
        f"{row.query_id}\t{row.question}" for row in selected
    ).encode("utf-8")
    artifact = DevelopmentQueryArtifact(
        target_manifest_sha256=source.target_manifest_sha256,
        query_partitions_sha256=source.query_partitions_sha256,
        query_count=len(selected),
        queries_sha256=sha256(canonical).hexdigest(),
        queries=selected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"query_count={artifact.query_count}")
    print(f"queries_sha256={artifact.queries_sha256}")
    print(f"excluded_query_count={len(excluded_ids)}")
    print("query_ids=" + ",".join(row.query_id for row in artifact.queries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
