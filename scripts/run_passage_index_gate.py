from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from deepresearch_harness.passage_index_gate import (
    PassageExportManifest,
    PassageIndexBuildManifest,
    export_passage_corpus,
    index_file_digests,
    load_passage_build_manifest,
    load_passage_export_manifest,
    load_passage_index_registration,
    run_passage_index_audit,
    sha256_file,
)
from deepresearch_harness.evidence_span_oracle import ArtifactReference


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and audit the frozen passage-level BM25 representation."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 32:
        raise ValueError("index threads must be between 1 and 32")

    registration_path = args.registration.resolve()
    output_path = args.output.resolve()
    registration = load_passage_index_registration(registration_path)
    root = registration_path.parents[2]
    if not output_path.is_relative_to((root / "runs").resolve()):
        raise ValueError("passage-index output must stay under ignored runs/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_manifest_path = output_path.parent / "export-manifest.json"
    build_manifest_path = output_path.parent / "index-build-manifest.json"

    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as error:
        raise RuntimeError("install the BrowseComp-Plus dependencies") from error

    source_index_path = (root / registration.source_document_index.path).resolve()
    corpus_dir = (root / registration.passage_corpus_path).resolve()
    passage_index_path = (root / registration.passage_index_path).resolve()
    source_searcher = LuceneSearcher(str(source_index_path))
    if source_searcher.num_docs != registration.source_document_index.document_count:
        raise ValueError("source Lucene document count differs from registration")

    export = _ensure_export(
        root=root,
        registration_path=registration_path,
        export_manifest_path=export_manifest_path,
        corpus_dir=corpus_dir,
        source_searcher=source_searcher,
    )
    build = _ensure_index(
        root=root,
        registration_path=registration_path,
        export_manifest_path=export_manifest_path,
        build_manifest_path=build_manifest_path,
        corpus_dir=corpus_dir,
        passage_index_path=passage_index_path,
        expected_passages=export.passage_count,
        threads=args.threads,
        log_dir=output_path.parent,
        searcher_type=LuceneSearcher,
    )

    source_searcher.set_bm25(
        registration.retrieval.bm25_k1, registration.retrieval.bm25_b
    )
    passage_searcher = LuceneSearcher(str(passage_index_path))
    passage_searcher.set_bm25(
        registration.retrieval.bm25_k1, registration.retrieval.bm25_b
    )
    if passage_searcher.num_docs != build.index_document_count:
        raise ValueError("opened passage index document count differs from build manifest")

    def full_search(query: str, limit: int) -> tuple[str, ...]:
        return tuple(str(hit.docid) for hit in source_searcher.search(query, k=limit))

    def passage_search(query: str, limit: int) -> tuple[str, ...]:
        return tuple(str(hit.docid) for hit in passage_searcher.search(query, k=limit))

    result = run_passage_index_audit(
        registration_path=registration_path,
        export_manifest_path=export_manifest_path,
        build_manifest_path=build_manifest_path,
        output_path=output_path,
        full_document_search=full_search,
        passage_search=passage_search,
        passage_document_exists=lambda docid: passage_searcher.doc(docid) is not None,
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"generated_query_count={result.generated_query_count}")
    print(
        "full_document_gold_hit_cases_at20="
        f"{result.full_document_gold_hit_cases_at20}"
    )
    print(f"passage_gold_hit_cases_at20={result.passage_gold_hit_cases_at20}")
    print(f"passage_wins={result.passage_wins}")
    print(f"passage_losses={result.passage_losses}")
    print(f"source_document_count={result.source_document_count}")
    print(f"passage_count={result.passage_count}")
    print(
        "development_gold_document_coverage_ratio="
        f"{result.development_gold_document_coverage_ratio:.6f}"
    )
    print(f"provider_calls={result.provider_calls}")
    print(f"online_search_calls={result.online_search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={output_path}")
    return 0


def _ensure_export(
    *,
    root: Path,
    registration_path: Path,
    export_manifest_path: Path,
    corpus_dir: Path,
    source_searcher: object,
) -> PassageExportManifest:
    registration = load_passage_index_registration(registration_path)
    corpus_path = corpus_dir / "collection.jsonl"
    partial_path = corpus_dir / "collection.jsonl.partial"
    if export_manifest_path.exists() or corpus_path.exists():
        if not export_manifest_path.is_file() or not corpus_path.is_file():
            raise ValueError("passage export is partial; preserve it and inspect before resume")
        return load_passage_export_manifest(
            registration_path=registration_path, manifest_path=export_manifest_path
        )
    if partial_path.exists():
        raise ValueError("passage export partial file already exists; refusing to overwrite")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    def documents():
        for internal_docid in range(source_searcher.num_docs):
            document = source_searcher.doc(internal_docid)
            if document is None:
                raise ValueError(f"source Lucene document is missing: {internal_docid}")
            raw = json.loads(document.raw())
            docid = str(raw.get("id", ""))
            contents = raw.get("contents")
            if docid != str(document.docid()):
                raise ValueError(f"source raw/docid mismatch: {internal_docid}")
            if not isinstance(contents, str) or not contents.strip():
                raise ValueError(f"source document has blank contents: {docid}")
            yield docid, contents

    stats = export_passage_corpus(
        documents=documents(),
        chunking=registration.chunking,
        output_path=partial_path,
    )
    if stats.source_document_count != registration.source_document_index.document_count:
        raise ValueError("passage export source count differs from registration")
    partial_path.replace(corpus_path)
    manifest = PassageExportManifest(
        created_at=_utc_now(),
        registration_sha256=sha256_file(registration_path),
        corpus_file=corpus_path.relative_to(root).as_posix(),
        chunking=registration.chunking,
        **stats.model_dump(),
    )
    export_manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _ensure_index(
    *,
    root: Path,
    registration_path: Path,
    export_manifest_path: Path,
    build_manifest_path: Path,
    corpus_dir: Path,
    passage_index_path: Path,
    expected_passages: int,
    threads: int,
    log_dir: Path,
    searcher_type: type,
) -> PassageIndexBuildManifest:
    registration = load_passage_index_registration(registration_path)
    if build_manifest_path.exists() or passage_index_path.exists():
        if not build_manifest_path.is_file() or not passage_index_path.is_dir():
            raise ValueError("passage index is partial; preserve it and inspect before resume")
        return load_passage_build_manifest(
            registration_path=registration_path,
            export_manifest_path=export_manifest_path,
            build_manifest_path=build_manifest_path,
        )
    partial_index = passage_index_path.with_name(passage_index_path.name + ".partial")
    if partial_index.exists():
        recovered = searcher_type(str(partial_index))
        recovered_document_count = recovered.num_docs
        if recovered_document_count != expected_passages:
            raise ValueError(
                "passage index partial directory is incomplete; preserving it without retry"
            )
        recovery_path = log_dir / "index-build.recovery.json"
        if recovery_path.exists():
            raise ValueError("passage index recovery artifact already exists")
        stdout_path = log_dir / "index-build.stdout.log"
        recovery = {
            "schema_version": "passage-index-build-recovery-v0",
            "created_at": _utc_now(),
            "reason": "parent_log_decode_failure_after_child_completion",
            "partial_index_path": partial_index.relative_to(root).as_posix(),
            "expected_document_count": expected_passages,
            "observed_document_count": recovered_document_count,
            "stdout_log": (
                {
                    "path": stdout_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(stdout_path),
                }
                if stdout_path.is_file()
                else None
            ),
            "stderr_log_missing_after_decode_failure": True,
            "rebuild_performed": False,
        }
        recovery_path.write_text(
            json.dumps(recovery, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if hasattr(recovered.object, "close"):
            recovered.object.close()
        recovered = None
        index_files = [path for path in partial_index.iterdir() if path.is_file()]
        build_latency_ms = max(
            0.0,
            (
                max(path.stat().st_mtime for path in index_files)
                - partial_index.stat().st_ctime
            )
            * 1000,
        )
        partial_index.replace(passage_index_path)
        return _write_build_manifest(
            root=root,
            registration_path=registration_path,
            export_manifest_path=export_manifest_path,
            build_manifest_path=build_manifest_path,
            passage_index_path=passage_index_path,
            expected_passages=expected_passages,
            searcher_type=searcher_type,
            command=_index_command(corpus_dir, partial_index, threads),
            build_latency_ms=build_latency_ms,
            completion_mode="recovered_completed_partial",
            recovery_path=recovery_path,
        )
    partial_index.parent.mkdir(parents=True, exist_ok=True)
    command = _index_command(corpus_dir, partial_index, threads)
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        env=os.environ.copy(),
        text=False,
        capture_output=True,
        check=False,
    )
    build_latency_ms = (perf_counter() - started) * 1000
    stdout_path = log_dir / "index-build.stdout.log"
    stderr_path = log_dir / "index-build.stderr.log"
    stdout_path.write_bytes(completed.stdout or b"")
    stderr_path.write_bytes(completed.stderr or b"")
    if completed.returncode != 0:
        raise RuntimeError(
            f"passage index build failed with exit {completed.returncode}; see {stderr_path}"
        )
    partial_index.replace(passage_index_path)
    return _write_build_manifest(
        root=root,
        registration_path=registration_path,
        export_manifest_path=export_manifest_path,
        build_manifest_path=build_manifest_path,
        passage_index_path=passage_index_path,
        expected_passages=expected_passages,
        searcher_type=searcher_type,
        command=command,
        build_latency_ms=build_latency_ms,
        completion_mode="direct",
        recovery_path=None,
    )


def _write_build_manifest(
    *,
    root: Path,
    registration_path: Path,
    export_manifest_path: Path,
    build_manifest_path: Path,
    passage_index_path: Path,
    expected_passages: int,
    searcher_type: type,
    command: tuple[str, ...],
    build_latency_ms: float,
    completion_mode: str,
    recovery_path: Path | None,
) -> PassageIndexBuildManifest:
    registration = load_passage_index_registration(registration_path)
    searcher = searcher_type(str(passage_index_path))
    if searcher.num_docs != expected_passages:
        raise ValueError("passage index count differs from exported passage count")
    manifest = PassageIndexBuildManifest(
        created_at=_utc_now(),
        registration_sha256=sha256_file(registration_path),
        export_manifest=ArtifactReference(
            path=export_manifest_path.relative_to(root).as_posix(),
            sha256=sha256_file(export_manifest_path),
        ),
        index_path=registration.passage_index_path,
        index_document_count=searcher.num_docs,
        index_files=index_file_digests(passage_index_path),
        pyserini_version=version("pyserini"),
        java_version=_java_version(),
        index_command=command,
        build_latency_ms=build_latency_ms,
        completion_mode=completion_mode,
        recovery_artifact=(
            ArtifactReference(
                path=recovery_path.relative_to(root).as_posix(),
                sha256=sha256_file(recovery_path),
            )
            if recovery_path is not None
            else None
        ),
    )
    build_manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _index_command(
    corpus_dir: Path, partial_index: Path, threads: int
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(corpus_dir),
        "--index",
        str(partial_index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    )


def _java_version() -> str:
    completed = subprocess.run(
        [str(Path(os.environ["JAVA_HOME"]) / "bin" / "java"), "-version"],
        text=True,
        capture_output=True,
        check=False,
    )
    text = (completed.stderr or completed.stdout).splitlines()
    return text[0].strip() if text else f"unknown on {platform.platform()}"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
