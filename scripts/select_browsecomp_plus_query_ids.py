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
        description="Project explicit development query IDs without opening gold labels."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-id", action="append", required=True)
    args = parser.parse_args()

    if len(args.query_id) != len(set(args.query_id)):
        raise ValueError("query IDs must be unique")
    source = load_development_queries(args.source)
    by_id = {row.query_id: row for row in source.queries}
    missing = [query_id for query_id in args.query_id if query_id not in by_id]
    if missing:
        raise ValueError(f"query IDs are absent from the source: {','.join(missing)}")
    selected = [by_id[query_id] for query_id in args.query_id]
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
    if args.output.exists():
        raise ValueError("output query artifact already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"query_count={artifact.query_count}")
    print(f"queries_sha256={artifact.queries_sha256}")
    print("query_ids=" + ",".join(row.query_id for row in artifact.queries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
