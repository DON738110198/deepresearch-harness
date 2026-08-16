from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import normalized_text_file_sha256
from .evidence_span_oracle import (
    AnswerCoverage,
    ArtifactReference,
    DecisionGate,
    answer_coverage,
)
from .retrieval_replay import RetrieverCandidatesManifest, select_candidate


DocumentLoader = Callable[[str], str | None]


class Tokenizer(Protocol):
    truncation_side: str

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def decode(self, token_ids: Sequence[Any], **kwargs: Any) -> str: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizedArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VisibilityPrerequisites(StrictContract):
    corpus_answerability_result: ArtifactReference
    dense_rank_result: ArtifactReference
    gold_slice: ArtifactReference
    source_index_registration: ArtifactReference


class IndexBuildProvenance(StrictContract):
    prebuilt_index_repository_url: str = Field(min_length=1)
    prebuilt_index_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    prebuilt_metadata_binding: Literal["absent"] = "absent"
    reproduction_recipe_repository_url: str = Field(min_length=1)
    reproduction_recipe_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    reproduction_recipe_path: str = Field(min_length=1)
    reproduction_recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tevatron_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tevatron_collator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tevatron_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_max_length: int = Field(ge=1)
    passage_max_length: int = Field(ge=1)
    passage_prefix: Literal[""] = ""
    append_eos_token: Literal[False] = False
    add_special_tokens: Literal[True] = True
    truncation_side: Literal["right"] = "right"
    content_policy: Literal["text.strip()"] = "text.strip()"

    @model_validator(mode="after")
    def documented_lengths_are_exact(self) -> "IndexBuildProvenance":
        if self.query_max_length != 512 or self.passage_max_length != 4096:
            raise ValueError(
                "index provenance must preserve the 512-token query and "
                "4096-token document recipe"
            )
        return self


class VisibilityDiagnostic(StrictContract):
    answer_match_policy: Literal["normalized_literal_atoms_v0"]
    document_visible_policy: str = Field(min_length=1)
    case_visible_policy: str = Field(min_length=1)
    maximum_tokens: Literal[4096] = 4096


class VisibilityAcceptance(StrictContract):
    minimum_visible_cases_to_reject_head_truncation_hypothesis: int = Field(ge=1)


class ZeroCallBudget(StrictContract):
    maximum_provider_calls: Literal[0] = 0
    maximum_online_search_calls: Literal[0] = 0
    maximum_judge_calls: Literal[0] = 0
    maximum_gpu_calls: Literal[0] = 0


class DenseDocumentVisibilityRegistration(StrictContract):
    schema_version: Literal["dense-document-visibility-registration-v0"] = (
        "dense-document-visibility-registration-v0"
    )
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    prerequisites: VisibilityPrerequisites
    query_ids: tuple[str, ...] = Field(min_length=1)
    document_index_path: str = Field(min_length=1)
    retriever_manifest: NormalizedArtifactReference
    candidate_id: str = Field(min_length=1)
    model_directory: str = Field(min_length=1)
    tokenizer_files_sha256: dict[str, str]
    index_build_provenance: IndexBuildProvenance
    diagnostic: VisibilityDiagnostic
    acceptance: VisibilityAcceptance
    budgets: ZeroCallBudget
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def frozen_contract_is_consistent(self) -> "DenseDocumentVisibilityRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("dense-document visibility query IDs must be unique")
        if (
            self.acceptance.minimum_visible_cases_to_reject_head_truncation_hypothesis
            > len(self.query_ids)
        ):
            raise ValueError("dense-document visibility gate exceeds registered cases")
        if self.diagnostic.maximum_tokens != self.index_build_provenance.passage_max_length:
            raise ValueError("visibility window differs from the 4096-token document recipe")
        expected_files = {"config.json", "tokenizer.json", "tokenizer_config.json"}
        if set(self.tokenizer_files_sha256) != expected_files:
            raise ValueError("dense-document visibility tokenizer file set changed")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.tokenizer_files_sha256.values()
        ):
            raise ValueError("dense-document visibility tokenizer hashes are invalid")
        return self


class TokenizerRuntime(StrictContract):
    transformers_version: str = Field(min_length=1)
    tokenizer_class: str = Field(min_length=1)
    truncation_side: Literal["right"] = "right"
    maximum_tokens: Literal[4096] = 4096
    tokenizer_files_sha256: dict[str, str]


class DocumentVisibility(StrictContract):
    docid: str = Field(min_length=1)
    document_characters: int = Field(gt=0)
    encoded_token_count: int = Field(ge=1, le=4096)
    window_truncated: bool
    full_document_coverage: AnswerCoverage
    visible_window_coverage: AnswerCoverage
    answer_visible: bool


class DenseDocumentVisibilityCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    documents: tuple[DocumentVisibility, ...] = Field(min_length=1)
    visible_document_count: int = Field(ge=0)
    answer_visible: bool

    @model_validator(mode="after")
    def aggregate_matches_documents(self) -> "DenseDocumentVisibilityCase":
        observed = sum(document.answer_visible for document in self.documents)
        if self.visible_document_count != observed:
            raise ValueError("visible document count differs from document observations")
        if self.answer_visible != (observed > 0):
            raise ValueError("case visibility differs from document observations")
        if self.gold_docids != tuple(document.docid for document in self.documents):
            raise ValueError("case gold documents differ from visibility observations")
        return self


VisibilityDecision = Literal[
    "reject_head_truncation_hypothesis",
    "admit_auditable_passage_dense_candidate",
]


VisibilityNextAction = Literal[
    "preregister_raw_question_dense_rank_alignment",
    "reconstruct_auditable_passage_dense_candidate",
]


class DenseDocumentVisibilityResult(StrictContract):
    schema_version: Literal["dense-document-visibility-v0"] = (
        "dense-document-visibility-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    decision: VisibilityDecision
    provenance_status: Literal[
        "reproduction_recipe_only_prebuilt_metadata_unbound"
    ] = "reproduction_recipe_only_prebuilt_metadata_unbound"
    registration: ArtifactReference
    tokenizer_runtime: TokenizerRuntime
    query_count: int = Field(ge=1)
    gold_document_count: int = Field(ge=1)
    visible_cases: int = Field(ge=0)
    hidden_cases: int = Field(ge=0)
    visible_documents: int = Field(ge=0)
    hidden_documents: int = Field(ge=0)
    truncated_documents: int = Field(ge=0)
    document_open_calls: int = Field(ge=0)
    provider_calls: Literal[0] = 0
    embedding_model_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    gpu_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[DenseDocumentVisibilityCase, ...] = Field(min_length=1)
    gates: tuple[DecisionGate, ...] = Field(min_length=1, max_length=1)
    next_action: VisibilityNextAction
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def aggregate_matches_items(self) -> "DenseDocumentVisibilityResult":
        documents = [document for item in self.items for document in item.documents]
        observed = (
            len(self.items),
            len(documents),
            sum(item.answer_visible for item in self.items),
            sum(not item.answer_visible for item in self.items),
            sum(document.answer_visible for document in documents),
            sum(not document.answer_visible for document in documents),
            sum(document.window_truncated for document in documents),
        )
        recorded = (
            self.query_count,
            self.gold_document_count,
            self.visible_cases,
            self.hidden_cases,
            self.visible_documents,
            self.hidden_documents,
            self.truncated_documents,
        )
        if observed != recorded:
            raise ValueError("visibility result aggregates differ from item observations")
        if self.document_open_calls != self.gold_document_count:
            raise ValueError("visibility document-open count changed")
        return self


def load_dense_document_visibility_registration(
    path: Path,
) -> DenseDocumentVisibilityRegistration:
    registration = DenseDocumentVisibilityRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(root, tuple(registration.prerequisites.model_dump().values()))

    manifest_path = _require_file(root, registration.retriever_manifest.path)
    if (
        normalized_text_file_sha256(manifest_path)
        != registration.retriever_manifest.normalized_sha256
    ):
        raise ValueError("dense-document visibility retriever manifest hash changed")
    manifest = RetrieverCandidatesManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    candidate = select_candidate(manifest, registration.candidate_id)
    if candidate.max_length != registration.index_build_provenance.query_max_length:
        raise ValueError("retriever query limit differs from the registered 512-token input")

    model_directory = _require_directory(root, registration.model_directory)
    for filename, expected_hash in registration.tokenizer_files_sha256.items():
        if _sha256_file(model_directory / filename) != expected_hash:
            raise ValueError(f"tokenizer file hash changed: {filename}")

    source_registration_path = _require_file(
        root, registration.prerequisites.source_index_registration.path
    )
    source_registration = _read_object(source_registration_path)
    source_index = source_registration.get("source_document_index")
    if not isinstance(source_index, dict):
        raise ValueError("source index registration has no document index contract")
    if source_index.get("path") != registration.document_index_path:
        raise ValueError("document index path differs from source index registration")
    index_directory = _require_directory(root, registration.document_index_path)
    files = source_index.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source index registration has no file manifest")
    _validate_index_files(index_directory, files)

    answerability = _read_object(
        _require_file(root, registration.prerequisites.corpus_answerability_result.path)
    )
    if answerability.get("decision") != "retrieval_layer_confirmed":
        raise ValueError("corpus answerability prerequisite changed")
    dense_rank = _read_object(
        _require_file(root, registration.prerequisites.dense_rank_result.path)
    )
    if dense_rank.get("decision") != "freeze_dense_channel":
        raise ValueError("dense-rank prerequisite changed")
    for label, artifact in (("answerability", answerability), ("dense-rank", dense_rank)):
        if _item_ids(artifact, label=label) != registration.query_ids:
            raise ValueError(f"{label} cases differ from visibility registration")

    gold = _read_object(_require_file(root, registration.prerequisites.gold_slice.path))
    gold_ids = tuple(str(row["query_id"]) for row in _rows(gold))
    missing = [query_id for query_id in registration.query_ids if query_id not in gold_ids]
    if missing:
        raise ValueError(f"visibility gold is missing cases: {missing}")
    return registration


def run_dense_document_visibility_audit(
    *,
    registration_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    tokenizer_runtime_version: str,
    document_loader: DocumentLoader,
) -> DenseDocumentVisibilityResult:
    registration = load_dense_document_visibility_registration(registration_path)
    root = registration_path.resolve().parents[2]
    output = output_path.resolve()
    if not output.is_relative_to((root / "runs").resolve()):
        raise ValueError("dense-document visibility output must stay under ignored runs/")
    if output.exists():
        raise ValueError("dense-document visibility result already exists")
    if getattr(tokenizer, "truncation_side", None) != "right":
        raise ValueError("dense-document visibility tokenizer must right-truncate")

    gold = _read_object(_require_file(root, registration.prerequisites.gold_slice.path))
    gold_by_id = {str(row["query_id"]): row for row in _rows(gold)}
    maximum_tokens = registration.diagnostic.maximum_tokens
    items: list[DenseDocumentVisibilityCase] = []
    for query_id in registration.query_ids:
        row = gold_by_id[query_id]
        correct_answer = str(row["answer"])
        documents: list[DocumentVisibility] = []
        for raw_docid in row["gold_docids"]:
            docid = str(raw_docid)
            contents = document_loader(docid)
            if contents is None or not contents.strip():
                raise ValueError(f"gold document is missing or blank: {docid}")
            full_coverage = answer_coverage(correct_answer, contents)
            if not full_coverage.all_atoms_present:
                raise ValueError(f"gold document no longer contains the literal answer: {docid}")
            token_ids, truncated = _encode_visible_window(
                tokenizer, contents.strip(), maximum_tokens
            )
            visible_text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            visible_coverage = answer_coverage(correct_answer, visible_text)
            documents.append(
                DocumentVisibility(
                    docid=docid,
                    document_characters=len(contents),
                    encoded_token_count=len(token_ids),
                    window_truncated=truncated,
                    full_document_coverage=full_coverage,
                    visible_window_coverage=visible_coverage,
                    answer_visible=visible_coverage.all_atoms_present,
                )
            )
        visible_count = sum(document.answer_visible for document in documents)
        items.append(
            DenseDocumentVisibilityCase(
                query_id=query_id,
                question=str(row["question"]),
                correct_answer=correct_answer,
                gold_docids=tuple(str(value) for value in row["gold_docids"]),
                documents=tuple(documents),
                visible_document_count=visible_count,
                answer_visible=visible_count > 0,
            )
        )

    visible_cases = sum(item.answer_visible for item in items)
    threshold = (
        registration.acceptance.minimum_visible_cases_to_reject_head_truncation_hypothesis
    )
    passed = visible_cases >= threshold
    decision: VisibilityDecision = (
        "reject_head_truncation_hypothesis"
        if passed
        else "admit_auditable_passage_dense_candidate"
    )
    next_action: VisibilityNextAction = (
        "preregister_raw_question_dense_rank_alignment"
        if passed
        else "reconstruct_auditable_passage_dense_candidate"
    )
    documents = [document for item in items for document in item.documents]
    result = DenseDocumentVisibilityResult(
        created_at=_utc_now(),
        decision=decision,
        registration=ArtifactReference(
            path=registration_path.resolve().relative_to(root).as_posix(),
            sha256=_sha256_file(registration_path),
        ),
        tokenizer_runtime=TokenizerRuntime(
            transformers_version=tokenizer_runtime_version,
            tokenizer_class=type(tokenizer).__name__,
            tokenizer_files_sha256=registration.tokenizer_files_sha256,
        ),
        query_count=len(items),
        gold_document_count=len(documents),
        visible_cases=visible_cases,
        hidden_cases=len(items) - visible_cases,
        visible_documents=sum(document.answer_visible for document in documents),
        hidden_documents=sum(not document.answer_visible for document in documents),
        truncated_documents=sum(document.window_truncated for document in documents),
        document_open_calls=len(documents),
        items=tuple(items),
        gates=(
            DecisionGate(
                gate_id="visible_cases_under_official_4096_token_recipe",
                observed=visible_cases,
                operator="ge",
                threshold=threshold,
                passed=passed,
            ),
        ),
        next_action=next_action,
        claim_boundary=registration.claim_boundary,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise ValueError("dense-document visibility partial result already exists")
    partial.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(output)
    return result


def _encode_visible_window(
    tokenizer: Tokenizer, text: str, maximum_tokens: int
) -> tuple[list[Any], bool]:
    kwargs = {
        "padding": False,
        "truncation": True,
        "return_attention_mask": False,
        "return_token_type_ids": False,
        "add_special_tokens": True,
    }
    encoded = tokenizer(text, max_length=maximum_tokens, **kwargs)
    probe = tokenizer(text, max_length=maximum_tokens + 1, **kwargs)
    token_ids = _input_ids(encoded)
    probe_ids = _input_ids(probe)
    if not token_ids:
        raise ValueError("dense-document tokenizer returned no tokens")
    if len(token_ids) > maximum_tokens or len(probe_ids) > maximum_tokens + 1:
        raise ValueError("dense-document tokenizer exceeded the requested window")
    return token_ids, len(probe_ids) > maximum_tokens


def _input_ids(encoded: Mapping[str, Any]) -> list[Any]:
    values = encoded.get("input_ids")
    if not isinstance(values, list) or (values and isinstance(values[0], list)):
        raise ValueError("dense-document tokenizer returned invalid input_ids")
    return values


def _validate_artifacts(root: Path, raw_artifacts: Sequence[object]) -> None:
    for raw in raw_artifacts:
        artifact = ArtifactReference.model_validate(raw)
        path = _require_file(root, artifact.path)
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")


def _validate_index_files(index_directory: Path, raw_files: Sequence[object]) -> None:
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("source index file manifest is invalid")
        name = raw.get("name")
        expected_bytes = raw.get("bytes")
        expected_hash = raw.get("sha256")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("source index filename is invalid")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError("source index byte count is invalid")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("source index hash is invalid")
        path = index_directory / name
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"source index file size changed: {name}")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"source index file hash changed: {name}")


def _item_ids(value: Mapping[str, object], *, label: str) -> tuple[str, ...]:
    items = value.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{label} items are invalid")
    return tuple(str(item["query_id"]) for item in items if isinstance(item, dict))


def _rows(value: Mapping[str, object]) -> list[dict[str, object]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("visibility gold rows are invalid")
    return rows


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"dense-document visibility artifact is not an object: {path}")
    return value


def _require_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"required visibility file is missing or escapes root: {relative}")
    return path


def _require_directory(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"required visibility directory is missing or escapes root: {relative}")
    return path


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required visibility file is missing: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
