from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .bm25_server import SearchResult
from .browsecomp_plus import load_browsecomp_plus_target
from .evidence_preview import (
    build_query_aware_lead_preview,
    format_query_aware_dense_lead,
)
from .evidence_span_oracle import select_answer_obligation_span
from .progressive_disclosure import (
    DisclosureSearchResult,
    DisclosureStateSnapshot,
    EvidenceCandidate,
    EvidenceTokenizer,
    OpenEvidenceResult,
    ProgressiveDisclosurePolicy,
    ProgressiveDisclosureSession,
    format_bm25_anchor,
    format_bm25_anchor_preview,
    format_dense_lead,
    format_opened_evidence,
    format_opened_obligation_span,
)
from .retrieval_replay import (
    QwenDenseReplaySearcher,
    load_retriever_candidates,
    select_candidate,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DisclosureSearchRequest(StrictContract):
    run_id: str = Field(min_length=1, max_length=512)
    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("run_id", "query")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class OpenEvidenceRequest(StrictContract):
    run_id: str = Field(min_length=1, max_length=512)
    docid: str = Field(min_length=1, max_length=512)
    obligation_query: str | None = Field(default=None, min_length=1, max_length=10_000)

    @field_validator("run_id", "docid", "obligation_query")
    @classmethod
    def values_are_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value


class DisclosureStateRequest(StrictContract):
    run_id: str = Field(min_length=1, max_length=512)

    @field_validator("run_id")
    @classmethod
    def run_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id must not be blank")
        return value


class DisclosureTrace(StrictContract):
    search_index: int = Field(ge=1)
    anchors_returned: int = Field(ge=0)
    leads_returned: int = Field(ge=0)
    within_channel_duplicate_slots: int = Field(ge=0)
    prior_context_duplicate_slots: int = Field(ge=0)
    cross_channel_duplicate_slots: int = Field(ge=0)
    new_ingress_tokens: int = Field(ge=0)
    cumulative_ingress_tokens: int = Field(ge=0)
    remaining_ingress_tokens: int = Field(ge=0)
    remaining_search_ingress_tokens: int = Field(ge=0)
    remaining_open_ingress_tokens: int = Field(ge=0)
    ingress_budget_exhausted: bool


class ProgressiveSearchResponse(StrictContract):
    results: list[SearchResult] = Field(max_length=20)
    disclosure: DisclosureTrace
    state: DisclosureStateSnapshot
    latency_ms: int = Field(ge=0)


class ProgressiveOpenResponse(StrictContract):
    result: OpenEvidenceResult
    state: DisclosureStateSnapshot
    latency_ms: int = Field(ge=0)


class ProgressiveStateResponse(StrictContract):
    state: DisclosureStateSnapshot


class ProgressiveHealthResponse(StrictContract):
    status: str = "ok"
    retriever_id: str = Field(min_length=1)
    session_count: int = Field(ge=0)
    policy: ProgressiveDisclosurePolicy


class ProgressiveDisclosureRuntime:
    def __init__(
        self,
        *,
        retriever_id: str,
        policy: ProgressiveDisclosurePolicy,
        tokenizer: EvidenceTokenizer,
        bm25_search: Callable[[str], Sequence[EvidenceCandidate]],
        dense_search: Callable[[str], Sequence[EvidenceCandidate]],
        document_loader: Callable[[str], str | None],
        obligation_document_loader: Callable[[str, str], str | None] | None = None,
        maximum_sessions: int = 1_000,
    ) -> None:
        if not retriever_id.strip():
            raise ValueError("retriever_id must not be blank")
        if maximum_sessions < 1:
            raise ValueError("maximum_sessions must be positive")
        self._retriever_id = retriever_id
        self._policy = policy
        self._tokenizer = tokenizer
        self._bm25_search = bm25_search
        self._dense_search = dense_search
        self._document_loader = document_loader
        self._obligation_document_loader = obligation_document_loader
        self._maximum_sessions = maximum_sessions
        self._sessions: dict[str, ProgressiveDisclosureSession] = {}
        self._lock = threading.Lock()

    def search(self, *, run_id: str, query: str) -> ProgressiveSearchResponse:
        started = time.perf_counter()
        with self._lock:
            session = self._get_or_create_session(run_id)
            result = session.search(
                bm25_candidates=self._bm25_search(query),
                dense_candidates=self._dense_search(query),
            )
            response = ProgressiveSearchResponse(
                results=[
                    SearchResult(docid=row.docid, score=row.score, snippet=row.content)
                    for row in result.results
                ],
                disclosure=_trace_from_result(result),
                state=session.snapshot(),
                latency_ms=round((time.perf_counter() - started) * 1_000),
            )
        return response

    def open_evidence(
        self, *, run_id: str, docid: str, obligation_query: str | None = None
    ) -> ProgressiveOpenResponse:
        started = time.perf_counter()
        with self._lock:
            session = self._get_session(run_id)
            result = session.open_evidence(
                docid, obligation_query=obligation_query
            )
            response = ProgressiveOpenResponse(
                result=result,
                state=session.snapshot(),
                latency_ms=round((time.perf_counter() - started) * 1_000),
            )
        return response

    def state(self, *, run_id: str) -> ProgressiveStateResponse:
        with self._lock:
            return ProgressiveStateResponse(state=self._get_session(run_id).snapshot())

    def health(self) -> ProgressiveHealthResponse:
        with self._lock:
            return ProgressiveHealthResponse(
                retriever_id=self._retriever_id,
                session_count=len(self._sessions),
                policy=self._policy,
            )

    def _get_or_create_session(self, run_id: str) -> ProgressiveDisclosureSession:
        session = self._sessions.get(run_id)
        if session is not None:
            return session
        if len(self._sessions) >= self._maximum_sessions:
            raise RuntimeError("maximum progressive-disclosure sessions reached")
        session = ProgressiveDisclosureSession(
            run_id=run_id,
            policy=self._policy,
            tokenizer=self._tokenizer,
            document_loader=self._document_loader,
            obligation_document_loader=self._obligation_document_loader,
        )
        self._sessions[run_id] = session
        return session

    def _get_session(self, run_id: str) -> ProgressiveDisclosureSession:
        try:
            return self._sessions[run_id]
        except KeyError as error:
            raise KeyError(f"unknown run_id: {run_id}") from error


class HuggingFaceEvidenceTokenizer:
    def __init__(self, tokenizer: Any, *, maximum_tokens: int) -> None:
        if maximum_tokens < 1:
            raise ValueError("maximum_tokens must be positive")
        self._tokenizer = tokenizer
        self._maximum_tokens = maximum_tokens

    def encode(self, text: str) -> Sequence[object]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self._maximum_tokens,
        )

    def decode(self, tokens: Sequence[object]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


def build_browsecomp_runtime(
    *,
    manifest_path: Path,
    retriever_manifest_path: Path,
    candidate_id: str,
    model_dir: Path,
    index_root: Path,
    document_index_path: Path,
    snippet_tokenizer_dir: Path,
    policy: ProgressiveDisclosurePolicy,
    dense_pool_size: int,
    lead_preview_policy: Literal["head_v0", "query_window_v0"] = "head_v0",
) -> ProgressiveDisclosureRuntime:
    try:
        from pyserini.search.lucene import LuceneSearcher
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install BrowseComp-Plus dense and BM25 dependencies"
        ) from error

    if dense_pool_size < policy.dense_lead_count or dense_pool_size > 20:
        raise ValueError("dense_pool_size must cover the lead count and be at most 20")
    target = load_browsecomp_plus_target(manifest_path)
    candidates = load_retriever_candidates(
        retriever_manifest_path, target_manifest_path=manifest_path
    )
    source_candidate = select_candidate(candidates, candidate_id)
    dense_candidate = source_candidate.model_copy(update={"top_k": dense_pool_size})
    dense = QwenDenseReplaySearcher(
        candidate=dense_candidate,
        model_dir=model_dir,
        index_root=index_root,
        batch_size=1,
    )
    documents = LuceneSearcher(str(document_index_path.resolve()))
    tokenizer = HuggingFaceEvidenceTokenizer(
        AutoTokenizer.from_pretrained(
            snippet_tokenizer_dir.resolve(),
            local_files_only=True,
            trust_remote_code=False,
        ),
        maximum_tokens=max(
            policy.anchor_token_cap,
            policy.lead_token_cap,
            policy.open_token_cap,
        ),
    )

    def load_document(docid: str) -> str | None:
        document = documents.doc(docid)
        if document is None:
            return None
        raw = json.loads(document.raw())
        contents = raw.get("contents")
        return contents if isinstance(contents, str) and contents.strip() else None

    def bm25_search(query: str) -> Sequence[EvidenceCandidate]:
        rows: list[EvidenceCandidate] = []
        for hit in documents.search(query, policy.anchor_count):
            raw = json.loads(hit.lucene_document.get("raw"))
            rows.append(
                EvidenceCandidate(
                    docid=hit.docid,
                    score=hit.score,
                    text=(
                        format_bm25_anchor(raw["contents"])
                        if policy.anchor_open_policy == "assume_full"
                        else format_bm25_anchor_preview(hit.docid, raw["contents"])
                    ),
                )
            )
        return rows

    def dense_search(query: str) -> Sequence[EvidenceCandidate]:
        rows: list[EvidenceCandidate] = []
        for hit in dense.search_many([query])[query]:
            contents = load_document(hit.docid)
            if contents is None:
                raise ValueError(f"dense hit is absent from document store: {hit.docid}")
            lead_text = (
                format_dense_lead(hit.docid, contents)
                if lead_preview_policy == "head_v0"
                else format_query_aware_dense_lead(
                    hit.docid,
                    build_query_aware_lead_preview(contents, query),
                )
            )
            rows.append(
                EvidenceCandidate(
                    docid=hit.docid,
                    score=hit.score,
                    text=lead_text,
                )
            )
        return rows

    def open_document(docid: str) -> str | None:
        contents = load_document(docid)
        if contents is None:
            return None
        return format_opened_evidence(docid, contents)

    def open_obligation_span(docid: str, obligation_query: str) -> str | None:
        contents = load_document(docid)
        if contents is None:
            return None
        selected = select_answer_obligation_span(
            contents,
            obligation_query,
            maximum_span_characters=2_000,
        )
        return format_opened_obligation_span(
            docid,
            selected.content,
            start_character=selected.start_character,
            end_character=selected.end_character,
        )

    retriever_id = (
            f"bm25-anchor-{policy.anchor_count}+{candidate_id}-lead-"
            f"{policy.dense_lead_count}-{lead_preview_policy}-"
            f"{policy.lead_token_cap}"
    )
    if policy.open_content_policy == "answer_obligation_window_v0":
        retriever_id += "-anchor-reopen-answer_obligation_window_v0"
    return ProgressiveDisclosureRuntime(
        retriever_id=retriever_id,
        policy=policy,
        tokenizer=tokenizer,
        bm25_search=bm25_search,
        dense_search=dense_search,
        document_loader=open_document,
        obligation_document_loader=(
            open_obligation_span
            if policy.open_content_policy == "answer_obligation_window_v0"
            else None
        ),
    )


def make_handler(runtime: ProgressiveDisclosureRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "deepresearch-progressive-disclosure/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(
                HTTPStatus.OK, runtime.health().model_dump(mode="json")
            )

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_body()
                if self.path == "/search":
                    request = DisclosureSearchRequest.model_validate(payload)
                    response: BaseModel = runtime.search(
                        run_id=request.run_id, query=request.query
                    )
                elif self.path == "/open":
                    request = OpenEvidenceRequest.model_validate(payload)
                    response = runtime.open_evidence(
                        run_id=request.run_id,
                        docid=request.docid,
                        obligation_query=request.obligation_query,
                    )
                elif self.path == "/state":
                    request = DisclosureStateRequest.model_validate(payload)
                    response = runtime.state(run_id=request.run_id)
                else:
                    self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
            except KeyError as error:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "unknown_run", "detail": str(error)},
                )
                return
            except (ValueError, ValidationError, json.JSONDecodeError) as error:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_request", "detail": str(error)},
                )
                return
            except Exception as error:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "request_failed", "detail": str(error)},
                )
                return
            self._write_json(HTTPStatus.OK, response.model_dump(mode="json"))

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _read_body(self) -> dict[str, object]:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > 65_536:
                raise ValueError("request body must contain 1 to 65536 bytes")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def _trace_from_result(result: DisclosureSearchResult) -> DisclosureTrace:
    return DisclosureTrace(
        search_index=result.search_index,
        anchors_returned=result.anchors_returned,
        leads_returned=result.leads_returned,
        within_channel_duplicate_slots=result.within_channel_duplicate_slots,
        prior_context_duplicate_slots=result.prior_context_duplicate_slots,
        cross_channel_duplicate_slots=result.cross_channel_duplicate_slots,
        new_ingress_tokens=result.new_ingress_tokens,
        cumulative_ingress_tokens=result.cumulative_ingress_tokens,
        remaining_ingress_tokens=result.remaining_ingress_tokens,
        remaining_search_ingress_tokens=result.remaining_search_ingress_tokens,
        remaining_open_ingress_tokens=result.remaining_open_ingress_tokens,
        ingress_budget_exhausted=result.ingress_budget_exhausted,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve BM25 anchors and progressively disclosed dense evidence."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retriever-manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--document-index-path", type=Path, required=True)
    parser.add_argument("--snippet-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=5)
    parser.add_argument("--dense-lead-count", type=int, default=15)
    parser.add_argument("--dense-pool-size", type=int, default=20)
    parser.add_argument(
        "--lead-preview-policy",
        choices=("head_v0", "query_window_v0"),
        default="head_v0",
    )
    parser.add_argument("--anchor-token-cap", type=int, default=512)
    parser.add_argument("--lead-token-cap", type=int, default=24)
    parser.add_argument("--open-token-cap", type=int, default=512)
    parser.add_argument(
        "--anchor-open-policy",
        choices=("assume_full", "reopen_with_obligation"),
        default="assume_full",
    )
    parser.add_argument(
        "--open-content-policy",
        choices=("head_v0", "answer_obligation_window_v0"),
        default="head_v0",
    )
    parser.add_argument("--maximum-open-calls", type=int, default=8)
    parser.add_argument("--total-evidence-ingress-token-budget", type=int, required=True)
    parser.add_argument("--open-evidence-ingress-token-budget", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the progressive-disclosure server must bind to loopback")
    policy = ProgressiveDisclosurePolicy(
        anchor_count=args.anchor_count,
        dense_lead_count=args.dense_lead_count,
        anchor_token_cap=args.anchor_token_cap,
        lead_token_cap=args.lead_token_cap,
        open_token_cap=args.open_token_cap,
        maximum_open_calls=args.maximum_open_calls,
        total_evidence_ingress_token_budget=(
            args.total_evidence_ingress_token_budget
        ),
        open_evidence_ingress_token_budget=(
            args.open_evidence_ingress_token_budget
        ),
        anchor_open_policy=args.anchor_open_policy,
        open_content_policy=args.open_content_policy,
    )
    runtime = build_browsecomp_runtime(
        manifest_path=args.manifest,
        retriever_manifest_path=args.retriever_manifest,
        candidate_id=args.candidate_id,
        model_dir=args.model_dir,
        index_root=args.index_root,
        document_index_path=args.document_index_path,
        snippet_tokenizer_dir=args.snippet_tokenizer_dir,
        policy=policy,
        dense_pool_size=args.dense_pool_size,
        lead_preview_policy=args.lead_preview_policy,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(
        f"progressive_disclosure_server=http://{args.host}:"
        f"{server.server_port}/search",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
