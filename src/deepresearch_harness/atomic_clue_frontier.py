from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RegisteredClueCase(StrictContract):
    query_id: str = Field(min_length=1)
    request: ArtifactReference
    queries: tuple[str, ...] = Field(min_length=1, max_length=6)


class AtomicClueAcceptance(StrictContract):
    minimum_gold_hit_cases: int = Field(ge=1)
    search_failures_must_equal: Literal[0] = 0


class AtomicClueRegistration(StrictContract):
    schema_version: Literal["atomic-clue-frontier-registration-v0"] = (
        "atomic-clue-frontier-registration-v0"
    )
    status: Literal["registered_before_retrieval"]
    registered_at: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    cases: tuple[RegisteredClueCase, ...] = Field(min_length=1)
    gold_slice: ArtifactReference
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    retriever_id: str = Field(min_length=1)
    max_search_results: int = Field(ge=1, le=100)
    splitter_id: Literal["declarative_sentence_frontier_v0"] = (
        "declarative_sentence_frontier_v0"
    )
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    acceptance: AtomicClueAcceptance
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def ids_and_gate_match(self) -> "AtomicClueRegistration":
        ids = tuple(item.query_id for item in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("atomic clue case IDs must be unique")
        if self.acceptance.minimum_gold_hit_cases > len(ids):
            raise ValueError("atomic clue gold-hit gate exceeds registered cases")
        return self


class RetrievedDocument(StrictContract):
    docid: str
    score: float
    snippet: str


class AtomicClueSearch(StrictContract):
    query_index: int = Field(ge=0)
    query: str
    documents: tuple[RetrievedDocument, ...]
    gold_hits: tuple[str, ...]
    gold_hit_ranks: tuple[int, ...]
    latency_ms: int = Field(ge=0)


class AtomicClueCaseResult(StrictContract):
    query_id: str
    status: Literal["succeeded", "failed"]
    searches: tuple[AtomicClueSearch, ...]
    unique_docids: tuple[str, ...]
    gold_hits: tuple[str, ...]
    error: str | None


class DecisionGate(StrictContract):
    gate_id: str
    observed: int | float
    operator: Literal["eq", "ge", "le"]
    threshold: int | float
    passed: bool


class AtomicClueResult(StrictContract):
    schema_version: Literal["atomic-clue-frontier-v0"] = "atomic-clue-frontier-v0"
    created_at: str
    status: Literal["succeeded", "failed"]
    decision: Literal["pass", "reject"]
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    search_failures: int = Field(ge=0)
    unique_documents: int = Field(ge=0)
    gold_hit_cases: int = Field(ge=0)
    items: tuple[AtomicClueCaseResult, ...]
    gates: tuple[DecisionGate, ...]
    next_action: Literal[
        "preregister_atomic_frontier_stage_replay",
        "use_near_miss_corpus_for_bridge_synthesis",
    ]
    claim_boundary: str


_REQUEST_PREFIXES = (
    "i want ",
    "can you ",
    "could you ",
    "how ",
    "please ",
    "what ",
    "when ",
    "where ",
    "which ",
    "who ",
)


def build_atomic_clue_queries(question: str) -> tuple[str, ...]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", question.strip())
    queries: list[str] = []
    for sentence in sentences:
        normalized = sentence.strip().strip('"').rstrip(".?!").strip()
        lowered = normalized.casefold()
        if not normalized or lowered.startswith(_REQUEST_PREFIXES):
            continue
        if len(_tokens(normalized)) < 5:
            continue
        queries.append(normalized)
    if not queries:
        raise ValueError("question produced no declarative clue queries")
    return tuple(queries[:6])


def _validate_artifacts(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")


def load_atomic_clue_registration(path: Path) -> AtomicClueRegistration:
    registration = AtomicClueRegistration.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (
            *(item.request for item in registration.cases),
            registration.gold_slice,
            *registration.frozen_artifacts,
        ),
    )
    for case in registration.cases:
        request = json.loads((root / case.request.path).read_text(encoding="utf-8"))
        computed = build_atomic_clue_queries(str(request["question"]))
        if computed != case.queries:
            raise ValueError(f"registered clue queries differ from deterministic splitter: {case.query_id}")
    return registration


def _search(
    search_url: str, run_id: str, query: str, timeout_seconds: int
) -> tuple[tuple[RetrievedDocument, ...], int]:
    request = Request(
        search_url,
        data=json.dumps({"run_id": run_id, "query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"atomic clue retrieval failed: {error}") from error
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("atomic clue retrieval response has no results array")
    documents = tuple(
        RetrievedDocument(
            docid=str(item["docid"]),
            score=float(item["score"]),
            snippet=str(item["snippet"]),
        )
        for item in results
    )
    return documents, round((time.perf_counter() - started) * 1000)


def run_atomic_clue_frontier(
    *, registration_path: Path, output_path: Path, timeout_seconds: int = 60
) -> AtomicClueResult:
    registration = load_atomic_clue_registration(registration_path)
    root = registration_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("atomic clue output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("atomic clue output already exists")
    gold = json.loads((root / registration.gold_slice.path).read_text(encoding="utf-8"))
    gold_by_id = {
        str(row["query_id"]): tuple(str(item) for item in row["gold_docids"])
        for row in gold["rows"]
    }
    items: list[AtomicClueCaseResult] = []
    search_calls = search_failures = 0
    for case in registration.cases:
        if case.query_id not in gold_by_id:
            raise ValueError(f"atomic clue case is absent from gold: {case.query_id}")
        searches: list[AtomicClueSearch] = []
        error_text: str | None = None
        for index, query in enumerate(case.queries):
            try:
                search_calls += 1
                documents, latency = _search(
                    registration.search_url,
                    f"atomic-clue-frontier-{case.query_id}-{index}",
                    query,
                    timeout_seconds,
                )
                if len(documents) > registration.max_search_results:
                    raise ValueError("atomic clue retriever exceeded the result cap")
                gold_docids = gold_by_id[case.query_id]
                hits = tuple(item.docid for item in documents if item.docid in gold_docids)
                ranks = tuple(
                    rank + 1
                    for rank, item in enumerate(documents)
                    if item.docid in gold_docids
                )
                searches.append(
                    AtomicClueSearch(
                        query_index=index,
                        query=query,
                        documents=documents,
                        gold_hits=hits,
                        gold_hit_ranks=ranks,
                        latency_ms=latency,
                    )
                )
            except Exception as error:
                search_failures += 1
                error_text = f"search_error: {type(error).__name__}: {error}"
                break
        unique_docids = tuple(
            dict.fromkeys(doc.docid for search in searches for doc in search.documents)
        )
        hits = tuple(dict.fromkeys(hit for search in searches for hit in search.gold_hits))
        items.append(
            AtomicClueCaseResult(
                query_id=case.query_id,
                status="succeeded" if error_text is None else "failed",
                searches=tuple(searches),
                unique_docids=unique_docids,
                gold_hits=hits,
                error=error_text,
            )
        )
    gold_hit_cases = sum(bool(item.gold_hits) for item in items)
    unique_documents = len(
        {docid for item in items for docid in item.unique_docids}
    )
    acceptance = registration.acceptance
    gates = (
        DecisionGate(gate_id="search_failures", observed=search_failures, operator="eq", threshold=0, passed=search_failures == 0),
        DecisionGate(gate_id="gold_hit_cases", observed=gold_hit_cases, operator="ge", threshold=acceptance.minimum_gold_hit_cases, passed=gold_hit_cases >= acceptance.minimum_gold_hit_cases),
    )
    decision: Literal["pass", "reject"] = (
        "pass" if all(gate.passed for gate in gates) else "reject"
    )
    result = AtomicClueResult(
        created_at=_utc_now(),
        status="succeeded" if search_failures == 0 else "failed",
        decision=decision,
        registration_sha256=_sha256_file(registration_path),
        query_count=len(registration.cases),
        search_calls=search_calls,
        search_failures=search_failures,
        unique_documents=unique_documents,
        gold_hit_cases=gold_hit_cases,
        items=tuple(items),
        gates=gates,
        next_action=(
            "preregister_atomic_frontier_stage_replay"
            if decision == "pass"
            else "use_near_miss_corpus_for_bridge_synthesis"
        ),
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
