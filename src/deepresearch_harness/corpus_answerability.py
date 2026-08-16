from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepresearch_harness.evidence_span_oracle import (
    AnswerCoverage,
    ArtifactReference,
    DecisionGate,
    answer_coverage,
)


DocumentLoader = Callable[[str], str | None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusAnswerabilityAcceptance(StrictContract):
    minimum_answerable_cases: int = Field(ge=1)


class CorpusAnswerabilityRegistration(StrictContract):
    schema_version: Literal["persistent-miss-corpus-answerability-registration-v0"] = (
        "persistent-miss-corpus-answerability-registration-v0"
    )
    status: Literal["posthoc_registered_failure_cluster"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    stability_audit: ArtifactReference
    gold_slice: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    document_index_path: str = Field(min_length=1)
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: CorpusAnswerabilityAcceptance
    sealed_holdout_access: Literal["forbidden"] = "forbidden"
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def gate_is_valid(self) -> "CorpusAnswerabilityRegistration":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("corpus-answerability query IDs must be unique")
        if self.acceptance.minimum_answerable_cases > len(self.query_ids):
            raise ValueError("corpus-answerability gate exceeds registered cases")
        return self


class GoldDocumentAnswerability(StrictContract):
    docid: str = Field(min_length=1)
    missing: bool
    document_characters: int = Field(ge=0)
    literal_answer_present: bool


class CorpusAnswerabilityCase(StrictContract):
    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    gold_docids: tuple[str, ...] = Field(min_length=1)
    documents: tuple[GoldDocumentAnswerability, ...] = Field(min_length=1)
    combined_coverage: AnswerCoverage
    answerable: bool


class CorpusAnswerabilityResult(StrictContract):
    schema_version: Literal["persistent-miss-corpus-answerability-v0"] = (
        "persistent-miss-corpus-answerability-v0"
    )
    created_at: str = Field(min_length=1)
    status: Literal["succeeded"] = "succeeded"
    decision: Literal[
        "retrieval_layer_confirmed",
        "corpus_or_label_audit_required",
    ]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    answerable_cases: int = Field(ge=0)
    literal_absent_cases: int = Field(ge=0)
    cases_with_missing_gold_documents: int = Field(ge=0)
    gold_document_count: int = Field(ge=0)
    missing_gold_documents: int = Field(ge=0)
    document_open_calls: int = Field(ge=0)
    provider_calls: Literal[0] = 0
    search_calls: Literal[0] = 0
    judge_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    items: tuple[CorpusAnswerabilityCase, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "diagnose_retriever_rank_and_index_representation",
        "audit_corpus_and_answer_aliases_before_retrieval_changes",
    ]
    claim_boundary: str = Field(min_length=1)


def load_corpus_answerability_registration(
    path: Path,
) -> CorpusAnswerabilityRegistration:
    registration = CorpusAnswerabilityRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            registration.stability_audit,
            registration.gold_slice,
            *registration.frozen_artifacts,
        ),
    )
    index = (root / registration.document_index_path).resolve()
    if not index.is_relative_to(root) or not index.is_dir():
        raise ValueError("corpus-answerability index is missing or escapes root")
    stability = json.loads(
        (root / registration.stability_audit.path).read_text(encoding="utf-8")
    )
    expected_ids = {
        str(row["query_id"])
        for row in stability["cases"]
        if row["category"] == "persistent_retrieval_miss"
    }
    if set(registration.query_ids) != expected_ids:
        raise ValueError("registered cases differ from the full persistent-miss cluster")
    return registration


def run_corpus_answerability_audit(
    *,
    registration_path: Path,
    output_path: Path,
    document_loader: DocumentLoader,
) -> CorpusAnswerabilityResult:
    registration = load_corpus_answerability_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("corpus-answerability output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("corpus-answerability output already exists")
    gold = json.loads((root / registration.gold_slice.path).read_text(encoding="utf-8"))
    gold_by_id = {str(row["query_id"]): row for row in gold["rows"]}

    items: list[CorpusAnswerabilityCase] = []
    missing_documents = 0
    document_open_calls = 0
    for query_id in registration.query_ids:
        row = gold_by_id.get(query_id)
        if row is None:
            raise ValueError(f"persistent-miss case is absent from gold: {query_id}")
        correct_answer = str(row["answer"])
        documents: list[GoldDocumentAnswerability] = []
        full_texts: list[str] = []
        for raw_docid in row["gold_docids"]:
            docid = str(raw_docid)
            document_open_calls += 1
            contents = document_loader(docid)
            if contents is None or not contents.strip():
                missing_documents += 1
                documents.append(
                    GoldDocumentAnswerability(
                        docid=docid,
                        missing=True,
                        document_characters=0,
                        literal_answer_present=False,
                    )
                )
                continue
            full_texts.append(contents)
            documents.append(
                GoldDocumentAnswerability(
                    docid=docid,
                    missing=False,
                    document_characters=len(contents),
                    literal_answer_present=answer_coverage(
                        correct_answer, contents
                    ).all_atoms_present,
                )
            )
        combined = answer_coverage(correct_answer, "\n".join(full_texts))
        items.append(
            CorpusAnswerabilityCase(
                query_id=query_id,
                question=str(row["question"]),
                correct_answer=correct_answer,
                gold_docids=tuple(str(value) for value in row["gold_docids"]),
                documents=tuple(documents),
                combined_coverage=combined,
                answerable=combined.all_atoms_present,
            )
        )

    answerable_cases = sum(item.answerable for item in items)
    cases_with_missing = sum(any(doc.missing for doc in item.documents) for item in items)
    literal_absent = len(items) - answerable_cases
    gate = DecisionGate(
        gate_id="answerable_cases",
        observed=answerable_cases,
        operator="ge",
        threshold=registration.acceptance.minimum_answerable_cases,
        passed=answerable_cases >= registration.acceptance.minimum_answerable_cases,
    )
    decision: Literal[
        "retrieval_layer_confirmed", "corpus_or_label_audit_required"
    ] = (
        "retrieval_layer_confirmed"
        if gate.passed
        else "corpus_or_label_audit_required"
    )
    result = CorpusAnswerabilityResult(
        created_at=_utc_now(),
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=len(items),
        answerable_cases=answerable_cases,
        literal_absent_cases=literal_absent,
        cases_with_missing_gold_documents=cases_with_missing,
        gold_document_count=sum(len(item.documents) for item in items),
        missing_gold_documents=missing_documents,
        document_open_calls=document_open_calls,
        items=tuple(items),
        gates=(gate,),
        next_action=(
            "diagnose_retriever_rank_and_index_representation"
            if decision == "retrieval_layer_confirmed"
            else "audit_corpus_and_answer_aliases_before_retrieval_changes"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _validate_artifacts(root: Path, artifacts: Sequence[ArtifactReference]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")
