from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.passage_index_gate import (
    PassageChunking,
    PassageExportManifest,
    PassageIndexBuildManifest,
    collapse_passage_hits,
    export_passage_corpus,
    index_file_digests,
    load_passage_index_registration,
    run_passage_index_audit,
    split_document_into_passages,
)
from deepresearch_harness.evidence_span_oracle import ArtifactReference


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _chunking() -> PassageChunking:
    return PassageChunking(
        tokenizer_id="whitespace_v0",
        passage_tokens=256,
        overlap_tokens=64,
        minimum_passage_tokens=1,
        header_policy="repeat_frontmatter_v0",
        maximum_header_tokens=64,
        passage_id_format="{docid}::p{ordinal:05d}",
    )


def test_split_passages_repeats_frontmatter_and_preserves_overlap() -> None:
    body = " ".join(f"token-{index}" for index in range(300))
    records = split_document_into_passages(
        "doc-1", f"---\ntitle: Frozen title\ndate: 2026-08-16\n---\n{body}", _chunking()
    )

    assert [record.id for record in records] == ["doc-1::p00000", "doc-1::p00001"]
    assert all("title: Frozen title" in record.contents for record in records)
    assert records[0].body_token_count == 256
    assert records[1].body_token_count == 108
    assert "token-192" in records[1].contents


def test_export_is_deterministic_and_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    documents = [("1", "alpha beta gamma"), ("2", "delta epsilon")]

    first_stats = export_passage_corpus(
        documents=documents, chunking=_chunking(), output_path=first
    )
    second_stats = export_passage_corpus(
        documents=documents, chunking=_chunking(), output_path=second
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_stats.corpus_sha256 == second_stats.corpus_sha256
    assert first_stats.source_document_count == 2
    assert first_stats.passage_count == 2

    with pytest.raises(ValueError, match="duplicate passage source docid"):
        export_passage_corpus(
            documents=[("1", "alpha"), ("1", "beta")],
            chunking=_chunking(),
            output_path=tmp_path / "duplicate.jsonl",
        )


def test_collapse_passage_hits_keeps_parent_rank_and_cap() -> None:
    collapsed = collapse_passage_hits(
        ["a::p00002", "a::p00000", "b::p00001", "c::p00000"],
        maximum_documents=2,
    )

    assert collapsed == ("a", "b")
    with pytest.raises(ValueError, match="invalid passage id"):
        collapse_passage_hits(["not-a-passage"], maximum_documents=2)


def test_passage_gate_accepts_registered_recall_without_provider_calls(
    tmp_path: Path,
) -> None:
    registration_path, export_path, build_path = _fixture(tmp_path)
    registration = load_passage_index_registration(registration_path)
    assert registration.query_ids == ("q1", "q2")

    output = tmp_path / "runs" / "result" / "audit.json"
    full = {"alpha query": ("d1",), "beta query": ("d2",)}
    passages = {
        "alpha query": ("gold-1::p00000", "d1::p00000"),
        "beta query": ("d2::p00000",),
    }
    result = run_passage_index_audit(
        registration_path=registration_path,
        export_manifest_path=export_path,
        build_manifest_path=build_path,
        output_path=output,
        full_document_search=lambda query, limit: full[query][:limit],
        passage_search=lambda query, limit: passages[query][:limit],
        passage_document_exists=lambda docid: docid
        in {"gold-1::p00000", "gold-2::p00000"},
    )

    assert result.decision == "passage_index_candidate"
    assert result.full_document_gold_hit_cases_at20 == 0
    assert result.passage_gold_hit_cases_at20 == 1
    assert result.passage_wins == 1
    assert result.development_gold_document_coverage_ratio == 1.0
    assert result.provider_calls == result.online_search_calls == result.judge_calls == 0
    assert all(gate.passed for gate in result.gates)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    benchmark_dir = tmp_path / "benchmarks" / "browsecomp_plus_v0"
    runs = tmp_path / "runs"
    source_index = runs / "external" / "bm25"
    source_index.mkdir(parents=True)
    source_segment = source_index / "segments_1"
    source_segment.write_bytes(b"source-index")

    lexical_path = runs / "prior" / "lexical.json"
    _write_json(
        lexical_path,
        {
            "decision": "index_representation_diagnosis_required",
            "generated_query_top20_cases": 0,
            "items": [{"query_id": "q1"}, {"query_id": "q2"}],
        },
    )
    stability_path = runs / "prior" / "stability.json"
    _write_json(
        stability_path,
        {
            "cases": [
                {"query_id": "q1", "category": "persistent_retrieval_miss"},
                {"query_id": "q2", "category": "persistent_retrieval_miss"},
            ]
        },
    )
    gold_path = runs / "prior" / "gold.json"
    _write_json(
        gold_path,
        {
            "rows": [
                {"query_id": "q1", "gold_docids": ["gold-1"]},
                {"query_id": "q2", "gold_docids": ["gold-2"]},
            ]
        },
    )
    baseline_root = runs / "baseline"
    _write_json(
        baseline_root / "q1" / "run.json",
        {"search_calls": [{"outcome": "ok", "query": "alpha query"}]},
    )
    _write_json(
        baseline_root / "q2" / "run.json",
        {"search_calls": [{"outcome": "ok", "query": "beta query"}]},
    )

    registration_path = benchmark_dir / "registration.json"
    _write_json(
        registration_path,
        {
            "schema_version": "passage-index-representation-registration-v0",
            "status": "posthoc_registered_failure_cluster",
            "registered_at": "2026-08-16T00:00:00+00:00",
            "purpose": "fixture passage gate",
            "prerequisites": {
                "lexical_rank_result": {
                    "path": lexical_path.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(lexical_path),
                },
                "stability_audit": {
                    "path": stability_path.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(stability_path),
                },
                "gold_slice": {
                    "path": gold_path.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(gold_path),
                },
            },
            "baseline_run_root": baseline_root.relative_to(tmp_path).as_posix(),
            "query_ids": ["q1", "q2"],
            "source_document_index": {
                "path": source_index.relative_to(tmp_path).as_posix(),
                "document_count": 2,
                "files": [
                    {
                        "name": source_segment.name,
                        "bytes": source_segment.stat().st_size,
                        "sha256": _hash(source_segment),
                    }
                ],
            },
            "passage_corpus_path": "runs/external/passages-corpus",
            "passage_index_path": "runs/external/passages-index",
            "chunking": _chunking().model_dump(mode="json"),
            "retrieval": {
                "query_source": "trial1_recorded_successful_search_calls",
                "analyzer_id": "pyserini_default_english_v1.2.0",
                "bm25_k1": 0.9,
                "bm25_b": 0.4,
                "maximum_passage_hits_per_query": 200,
                "maximum_unique_documents_per_query": 20,
            },
            "acceptance": {
                "minimum_passage_generated_query_gold_doc_recall_at20_cases": 1,
                "required_source_document_coverage_ratio": 1.0,
                "required_development_gold_document_coverage_ratio": 1.0,
            },
            "budgets": {
                "maximum_provider_calls": 0,
                "maximum_online_search_calls": 0,
                "maximum_judge_calls": 0,
            },
            "sealed_holdout_access": "forbidden",
            "claim_boundary": "fixture only",
        },
    )

    corpus_path = runs / "external" / "passages-corpus" / "collection.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        '{"id":"gold-1::p00000","contents":"alpha"}\n'
        '{"id":"gold-2::p00000","contents":"beta"}\n',
        encoding="utf-8",
    )
    export_path = runs / "result" / "export.json"
    export = PassageExportManifest(
        created_at="2026-08-16T00:00:00+00:00",
        registration_sha256=_hash(registration_path),
        corpus_file=corpus_path.relative_to(tmp_path).as_posix(),
        chunking=_chunking(),
        source_document_count=2,
        source_documents_with_passages=2,
        passage_count=2,
        corpus_sha256=_hash(corpus_path),
        source_docids_sha256="0" * 64,
        passage_ids_sha256="1" * 64,
        export_latency_ms=1,
    )
    _write_json(export_path, export.model_dump(mode="json"))

    passage_index = runs / "external" / "passages-index"
    passage_index.mkdir(parents=True)
    passage_segment = passage_index / "segments_1"
    passage_segment.write_bytes(b"passage-index")
    build_path = runs / "result" / "build.json"
    build = PassageIndexBuildManifest(
        created_at="2026-08-16T00:00:00+00:00",
        registration_sha256=_hash(registration_path),
        export_manifest=ArtifactReference(
            path=export_path.relative_to(tmp_path).as_posix(), sha256=_hash(export_path)
        ),
        index_path=passage_index.relative_to(tmp_path).as_posix(),
        index_document_count=2,
        index_files=index_file_digests(passage_index),
        pyserini_version="1.2.0",
        java_version="fixture",
        index_command=("fixture",),
        build_latency_ms=1,
    )
    _write_json(build_path, build.model_dump(mode="json"))
    return registration_path, export_path, build_path
