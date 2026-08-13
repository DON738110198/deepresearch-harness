from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verify_browsecomp_plus_judge_assets import verify_assets


def test_judge_asset_verifier_checks_sha256_and_git_blob(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sha_content = b"weights"
    blob_content = b'{"model":"fixture"}\n'
    (model_dir / "weights.bin").write_bytes(sha_content)
    (model_dir / "config.json").write_bytes(blob_content)
    blob_hash = hashlib.sha1(
        f"blob {len(blob_content)}\0".encode("ascii") + blob_content
    ).hexdigest()
    manifest = {
        "schema_version": "browsecomp-plus-official-judge-assets-v0",
        "model": "Qwen/Qwen3-32B",
        "revision": "a" * 40,
        "files": [
            {
                "path": "weights.bin",
                "size": len(sha_content),
                "hash_algorithm": "sha256",
                "hash": hashlib.sha256(sha_content).hexdigest(),
            },
            {
                "path": "config.json",
                "size": len(blob_content),
                "hash_algorithm": "git_blob_sha1",
                "hash": blob_hash,
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_assets(manifest_path, model_dir)["passed"] is True
    (model_dir / "config.json").write_text("tampered", encoding="utf-8")
    result = verify_assets(manifest_path, model_dir)
    assert result["passed"] is False
    assert result["matched"] == 1
