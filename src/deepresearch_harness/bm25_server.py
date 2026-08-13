from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .browsecomp_plus import BrowseCompPlusTargetManifest, load_browsecomp_plus_target


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("query")
    @classmethod
    def query_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docid: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    snippet: str = Field(min_length=1)


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[SearchResult] = Field(max_length=5)
    latency_ms: int = Field(ge=0)


class BrowseCompBM25Runtime:
    def __init__(self, *, index_path: Path, manifest: BrowseCompPlusTargetManifest) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "BM25 runtime dependencies are missing; install pyserini and transformers"
            ) from error

        contract = manifest.benchmark.standard_search
        self._top_k = contract.top_k
        self._snippet_max_tokens = contract.snippet_max_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(
            contract.snippet_tokenizer.name,
            revision=contract.snippet_tokenizer.revision,
            trust_remote_code=False,
        )
        self._searcher = LuceneSearcher(str(index_path.resolve()))
        self._lock = threading.Lock()

    def search(self, query: str) -> SearchResponse:
        started = time.perf_counter()
        with self._lock:
            hits = self._searcher.search(query, self._top_k)
            results = []
            for hit in hits:
                raw = json.loads(hit.lucene_document.get("raw"))
                snippet = truncate_with_tokenizer(
                    raw["contents"], self._tokenizer, self._snippet_max_tokens
                )
                results.append(
                    SearchResult(docid=hit.docid, score=hit.score, snippet=snippet)
                )
        return SearchResponse(
            results=results,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


def truncate_with_tokenizer(text: str, tokenizer: Any, max_tokens: int) -> str:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens + 1,
    )
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def make_handler(runtime: BrowseCompBM25Runtime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "deepresearch-bm25/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write_json(HTTPStatus.OK, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/search":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length <= 0 or length > 65_536:
                    raise ValueError("request body must contain 1 to 65536 bytes")
                request = SearchRequest.model_validate_json(self.rfile.read(length))
                response = runtime.search(request.query)
            except (ValueError, ValidationError, json.JSONDecodeError) as error:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_request", "detail": str(error)},
                )
                return
            except Exception as error:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "search_failed", "detail": str(error)},
                )
                return
            self._write_json(HTTPStatus.OK, response.model_dump(mode="json"))

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the pinned BrowseComp-Plus BM25 contract.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the benchmark search server must bind to loopback")
    if not args.index_path.is_dir():
        raise ValueError(f"index path does not exist: {args.index_path}")
    manifest = load_browsecomp_plus_target(args.manifest)
    runtime = BrowseCompBM25Runtime(index_path=args.index_path, manifest=manifest)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"bm25_server=http://{args.host}:{server.server_port}/search", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
