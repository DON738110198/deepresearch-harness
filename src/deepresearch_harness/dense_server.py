from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from .bm25_server import SearchResponse, SearchResult, make_handler, truncate_with_tokenizer
from .browsecomp_plus import load_browsecomp_plus_target
from .retrieval_replay import (
    QwenDenseReplaySearcher,
    load_retriever_candidates,
    select_candidate,
)


class BrowseCompDenseRuntime:
    def __init__(
        self,
        *,
        manifest_path: Path,
        retriever_manifest_path: Path,
        candidate_id: str,
        model_dir: Path,
        index_root: Path,
        document_index_path: Path,
        snippet_tokenizer_dir: Path | None = None,
    ) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                'install server dependencies with pip install -e ".[browsecomp-plus,browsecomp-plus-dense]"'
            ) from error

        target = load_browsecomp_plus_target(manifest_path)
        candidates = load_retriever_candidates(
            retriever_manifest_path, target_manifest_path=manifest_path
        )
        candidate = select_candidate(candidates, candidate_id)
        if candidate.top_k != target.benchmark.standard_search.top_k:
            raise ValueError("dense candidate top_k differs from the benchmark search contract")

        self._dense = QwenDenseReplaySearcher(
            candidate=candidate,
            model_dir=model_dir,
            index_root=index_root,
            batch_size=1,
        )
        tokenizer_source: str | Path = (
            snippet_tokenizer_dir
            if snippet_tokenizer_dir is not None
            else target.benchmark.standard_search.snippet_tokenizer.name
        )
        tokenizer_kwargs: dict[str, object] = {"trust_remote_code": False}
        if snippet_tokenizer_dir is None:
            tokenizer_kwargs["revision"] = (
                target.benchmark.standard_search.snippet_tokenizer.revision
            )
        else:
            tokenizer_kwargs["local_files_only"] = True
        self._snippet_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, **tokenizer_kwargs
        )
        self._snippet_max_tokens = target.benchmark.standard_search.snippet_max_tokens
        self._documents = LuceneSearcher(str(document_index_path.resolve()))
        self._lock = threading.Lock()

    def search(self, query: str) -> SearchResponse:
        started = time.perf_counter()
        with self._lock:
            hits = self._dense.search_many([query])[query]
            results: list[SearchResult] = []
            for hit in hits:
                document = self._documents.doc(hit.docid)
                if document is None:
                    raise ValueError(f"dense hit is absent from document store: {hit.docid}")
                raw = json.loads(document.raw())
                snippet = truncate_with_tokenizer(
                    raw["contents"],
                    self._snippet_tokenizer,
                    self._snippet_max_tokens,
                )
                results.append(
                    SearchResult(docid=hit.docid, score=hit.score, snippet=snippet)
                )
        return SearchResponse(
            results=results,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve a pinned dense BrowseComp-Plus retriever on loopback."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retriever-manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--document-index-path", type=Path, required=True)
    parser.add_argument("--snippet-tokenizer-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the benchmark search server must bind to loopback")
    runtime = BrowseCompDenseRuntime(
        manifest_path=args.manifest,
        retriever_manifest_path=args.retriever_manifest,
        candidate_id=args.candidate_id,
        model_dir=args.model_dir,
        index_root=args.index_root,
        document_index_path=args.document_index_path,
        snippet_tokenizer_dir=args.snippet_tokenizer_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"dense_server=http://{args.host}:{server.server_port}/search", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
