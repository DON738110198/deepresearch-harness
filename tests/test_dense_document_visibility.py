from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.dense_document_visibility import (
    load_dense_document_visibility_registration,
    run_dense_document_visibility_audit,
)


class WhitespaceTokenizer:
    truncation_side = "right"

    def __call__(
        self,
        text: str,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[str]]:
        del padding, return_attention_mask, return_token_type_ids, add_special_tokens
        tokens = text.split()
        if truncation:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}

    def decode(
        self,
        token_ids: list[str],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(token_ids)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_visibility_audit_routes_away_from_truncation_without_model_calls(
    tmp_path: Path,
) -> None:
    registration_path, documents = _fixture(tmp_path)
    output = tmp_path / "runs" / "audit" / "result.json"

    result = run_dense_document_visibility_audit(
        registration_path=registration_path,
        output_path=output,
        tokenizer=WhitespaceTokenizer(),
        tokenizer_runtime_version="fixture-transformers",
        document_loader=documents.get,
    )

    assert result.decision == "reject_head_truncation_hypothesis"
    assert result.next_action == "preregister_raw_question_dense_rank_alignment"
    assert result.visible_cases == 1
    assert result.hidden_cases == 1
    assert result.visible_documents == 1
    assert result.truncated_documents == 1
    assert result.provider_calls == 0
    assert result.embedding_model_calls == 0
    assert result.search_calls == 0
    assert result.judge_calls == 0
    assert result.gpu_calls == 0
    assert result.sealed_holdout_accessed is False
    assert output.is_file()


def test_registration_rejects_query_limit_misread_as_document_limit(
    tmp_path: Path,
) -> None:
    registration_path, _ = _fixture(tmp_path)
    payload = json.loads(registration_path.read_text(encoding="utf-8"))
    payload["index_build_provenance"]["passage_max_length"] = 512
    _write_json(registration_path, payload)

    with pytest.raises(ValueError, match="4096-token document recipe"):
        load_dense_document_visibility_registration(registration_path)


def test_registration_rejects_unpinned_tokenizer_file(tmp_path: Path) -> None:
    registration_path, _ = _fixture(tmp_path)
    tokenizer_json = tmp_path / "runs" / "model" / "tokenizer.json"
    tokenizer_json.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer file hash changed"):
        load_dense_document_visibility_registration(registration_path)


def test_visibility_audit_refuses_missing_gold_document(tmp_path: Path) -> None:
    registration_path, documents = _fixture(tmp_path)
    documents.pop("doc-visible")

    with pytest.raises(ValueError, match="gold document is missing"):
        run_dense_document_visibility_audit(
            registration_path=registration_path,
            output_path=tmp_path / "runs" / "audit" / "result.json",
            tokenizer=WhitespaceTokenizer(),
            tokenizer_runtime_version="fixture-transformers",
            document_loader=documents.get,
        )


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    benchmark = tmp_path / "benchmarks" / "browsecomp_plus_v0"
    runs = tmp_path / "runs"
    benchmark.mkdir(parents=True)
    runs.mkdir()

    index = runs / "index"
    index.mkdir()
    segment = index / "segments_1"
    segment.write_bytes(b"fixture-index")
    source_index_registration = benchmark / "source-index.json"
    _write_json(
        source_index_registration,
        {
            "source_document_index": {
                "path": index.relative_to(tmp_path).as_posix(),
                "document_count": 2,
                "files": [
                    {
                        "name": segment.name,
                        "bytes": segment.stat().st_size,
                        "sha256": _hash(segment),
                    }
                ],
            }
        },
    )

    gold = runs / "gold.json"
    _write_json(
        gold,
        {
            "rows": [
                {
                    "query_id": "q-visible",
                    "question": "Which visible code?",
                    "answer": "VISIBLE-CODE",
                    "gold_docids": ["doc-visible"],
                },
                {
                    "query_id": "q-hidden",
                    "question": "Which hidden code?",
                    "answer": "HIDDEN-CODE",
                    "gold_docids": ["doc-hidden"],
                },
            ]
        },
    )
    answerability = runs / "answerability.json"
    _write_json(
        answerability,
        {
            "decision": "retrieval_layer_confirmed",
            "items": [
                {"query_id": "q-visible", "gold_docids": ["doc-visible"]},
                {"query_id": "q-hidden", "gold_docids": ["doc-hidden"]},
            ],
        },
    )
    dense_rank = runs / "dense-rank.json"
    _write_json(
        dense_rank,
        {
            "decision": "freeze_dense_channel",
            "items": [
                {"query_id": "q-visible", "gold_docids": ["doc-visible"]},
                {"query_id": "q-hidden", "gold_docids": ["doc-hidden"]},
            ],
        },
    )

    model_dir = runs / "model"
    model_dir.mkdir()
    tokenizer_files = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"tokenizer-config",
    }
    for name, content in tokenizer_files.items():
        (model_dir / name).write_bytes(content)

    retriever_manifest = benchmark / "retriever.json"
    _write_json(
        retriever_manifest,
        {
            "schema_version": "browsecomp-plus-retriever-candidates-v0",
            "target_manifest_sha256": "0" * 64,
            "candidates": [
                {
                    "candidate_id": "fixture-dense",
                    "kind": "dense_faiss",
                    "model": {
                        "name": "fixture/model",
                        "revision": "1" * 40,
                        "model_file": "model.safetensors",
                        "model_file_sha256": "2" * 64,
                    },
                    "index": {
                        "dataset": "fixture/index",
                        "revision": "3" * 40,
                        "subdirectory": "dense",
                        "shards": [{"filename": "corpus.pkl", "sha256": "4" * 64}],
                    },
                    "query_prefix": "Query:",
                    "pooling": "eos",
                    "normalize": True,
                    "max_length": 512,
                    "top_k": 5,
                }
            ],
            "replay": {
                "source_queries": "frozen_agent_search_calls",
                "baseline": "stored_bm25_top5",
                "candidate_depth": 5,
                "fusion": "reciprocal_rank_fusion",
                "fusion_k": 60,
                "fused_depth": 5,
                "official_metric_status": "diagnostic_not_official",
            },
        },
    )

    registration = benchmark / "registration.json"
    _write_json(
        registration,
        {
            "schema_version": "dense-document-visibility-registration-v0",
            "status": "posthoc_registered_failure_cluster",
            "registered_at": "2026-08-16T00:00:00+00:00",
            "purpose": "fixture visibility audit",
            "prerequisites": {
                "corpus_answerability_result": {
                    "path": answerability.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(answerability),
                },
                "dense_rank_result": {
                    "path": dense_rank.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(dense_rank),
                },
                "gold_slice": {
                    "path": gold.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(gold),
                },
                "source_index_registration": {
                    "path": source_index_registration.relative_to(tmp_path).as_posix(),
                    "sha256": _hash(source_index_registration),
                },
            },
            "query_ids": ["q-visible", "q-hidden"],
            "document_index_path": index.relative_to(tmp_path).as_posix(),
            "retriever_manifest": {
                "path": retriever_manifest.relative_to(tmp_path).as_posix(),
                "normalized_sha256": normalized_text_file_sha256(retriever_manifest),
            },
            "candidate_id": "fixture-dense",
            "model_directory": model_dir.relative_to(tmp_path).as_posix(),
            "tokenizer_files_sha256": {
                name: _hash(model_dir / name) for name in tokenizer_files
            },
            "index_build_provenance": {
                "prebuilt_index_repository_url": "https://example.test/index",
                "prebuilt_index_revision": "5" * 40,
                "prebuilt_metadata_binding": "absent",
                "reproduction_recipe_repository_url": "https://example.test/source",
                "reproduction_recipe_commit": "6" * 40,
                "reproduction_recipe_path": "recipe.md",
                "reproduction_recipe_sha256": "7" * 64,
                "tevatron_commit": "8" * 40,
                "tevatron_collator_sha256": "9" * 64,
                "tevatron_dataset_sha256": "a" * 64,
                "query_max_length": 512,
                "passage_max_length": 4096,
                "passage_prefix": "",
                "append_eos_token": False,
                "add_special_tokens": True,
                "truncation_side": "right",
                "content_policy": "text.strip()",
            },
            "diagnostic": {
                "answer_match_policy": "normalized_literal_atoms_v0",
                "document_visible_policy": "all answer atoms survive",
                "case_visible_policy": "at least one gold document is visible",
                "maximum_tokens": 4096,
            },
            "acceptance": {
                "minimum_visible_cases_to_reject_head_truncation_hypothesis": 1
            },
            "budgets": {
                "maximum_provider_calls": 0,
                "maximum_online_search_calls": 0,
                "maximum_judge_calls": 0,
                "maximum_gpu_calls": 0,
            },
            "sealed_holdout_access": "forbidden",
            "claim_boundary": "fixture diagnostic only",
        },
    )

    documents = {
        "doc-visible": "lead VISIBLE-CODE tail",
        "doc-hidden": " ".join(["noise"] * 4096 + ["HIDDEN-CODE"]),
    }
    return registration, documents
