from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def verify_assets(manifest_path: Path, model_dir: Path, *, workers: int = 1) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "browsecomp-plus-official-judge-assets-v0":
        raise ValueError("unsupported judge asset manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("judge asset manifest has no files")

    seen: set[str] = set()
    for expected in files:
        relative = expected["path"]
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe or duplicate judge asset path: {relative}")
        seen.add(relative)
    if workers < 1:
        raise ValueError("judge asset verifier workers must be positive")
    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as executor:
        results = list(
            executor.map(lambda expected: _verify_file(expected, model_dir), files)
        )

    return {
        "schema_version": "browsecomp-plus-official-judge-asset-verification-v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": manifest["model"],
        "revision": manifest["revision"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_dir": str(model_dir.resolve()),
        "file_count": len(results),
        "matched": sum(row["matches"] for row in results),
        "passed": all(row["matches"] for row in results),
        "files": results,
    }


def _verify_file(expected: dict, model_dir: Path) -> dict:
    relative = expected["path"]
    path = (model_dir / relative).resolve()
    if not path.is_relative_to(model_dir.resolve()):
        raise ValueError(f"judge asset escaped model directory: {relative}")
    exists = path.is_file()
    actual_size = path.stat().st_size if exists else None
    actual_hash = (
        _file_hash(path, expected["hash_algorithm"], actual_size) if exists else None
    )
    return {
        "path": relative,
        "exists": exists,
        "expected_size": expected["size"],
        "actual_size": actual_size,
        "hash_algorithm": expected["hash_algorithm"],
        "expected_hash": expected["hash"],
        "actual_hash": actual_hash,
        "matches": (
            exists
            and actual_size == expected["size"]
            and actual_hash == expected["hash"]
        ),
    }


def _file_hash(path: Path, algorithm: str, size: int) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif algorithm == "git_blob_sha1":
        digest = hashlib.sha1()
        prefix = f"blob {size}\0".encode("ascii")
    else:
        raise ValueError(f"unsupported judge asset hash algorithm: {algorithm}")
    digest.update(prefix)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a Qwen3 judge directory against the pinned HF revision."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = verify_assets(args.manifest, args.model_dir, workers=args.workers)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(
        f"passed={str(result['passed']).lower()}\n"
        f"matched={result['matched']}/{result['file_count']}\n"
        f"revision={result['revision']}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
