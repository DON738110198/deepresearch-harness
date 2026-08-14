from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from .bm25_server import SearchResponse, SearchResult, make_handler
from .browsecomp_plus import load_browsecomp_plus_target
from .evidence_bandwidth import allocate_waterfill_caps
from .retrieval_replay import (
    QwenDenseReplaySearcher,
    load_retriever_candidates,
    select_candidate,
)


class EvidenceBandwidthDenseRuntime:
    def __init__(
        self,
        *,
        manifest_path: Path,
        retriever_manifest_path: Path,
        candidate_id: str,
        model_dir: Path,
        index_root: Path,
        document_index_path: Path,
        snippet_tokenizer_dir: Path,
        result_count: int,
        total_snippet_token_budget: int,
        minimum_snippet_tokens_per_result: int,
    ) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "install BrowseComp-Plus dense and BM25 dependencies"
            ) from error
        target = load_browsecomp_plus_target(manifest_path)
        candidates = load_retriever_candidates(
            retriever_manifest_path, target_manifest_path=manifest_path
        )
        source_candidate = select_candidate(candidates, candidate_id)
        if result_count <= target.benchmark.standard_search.top_k or result_count > 20:
            raise ValueError("evidence bandwidth result count must be in [6, 20]")
        candidate = source_candidate.model_copy(update={"top_k": result_count})
        self._dense = QwenDenseReplaySearcher(
            candidate=candidate,
            model_dir=model_dir,
            index_root=index_root,
            batch_size=1,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            snippet_tokenizer_dir.resolve(),
            local_files_only=True,
            trust_remote_code=False,
        )
        self._documents = LuceneSearcher(str(document_index_path.resolve()))
        self._result_count = result_count
        self._total_budget = total_snippet_token_budget
        self._minimum = minimum_snippet_tokens_per_result
        if self._minimum * self._result_count > self._total_budget:
            raise ValueError("evidence bandwidth budget cannot fund every result")
        self._lock = threading.Lock()

    def search(self, query: str) -> SearchResponse:
        started = time.perf_counter()
        with self._lock:
            hits = self._dense.search_many([query])[query]
            tokenized: list[list[int]] = []
            for hit in hits:
                document = self._documents.doc(hit.docid)
                if document is None:
                    raise ValueError(
                        f"dense hit is absent from document store: {hit.docid}"
                    )
                raw = json.loads(document.raw())
                tokenized.append(
                    self._tokenizer.encode(
                        raw["contents"],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self._total_budget,
                    )
                )
            caps = allocate_waterfill_caps(
                [len(tokens) for tokens in tokenized],
                total_budget=self._total_budget,
                minimum_per_result=self._minimum,
            )
            results = [
                SearchResult(
                    docid=hit.docid,
                    score=hit.score,
                    snippet=self._tokenizer.decode(
                        tokens[:cap], skip_special_tokens=True
                    ),
                )
                for hit, tokens, cap in zip(hits, tokenized, caps, strict=True)
            ]
        return SearchResponse(
            results=results,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve token-budgeted dense BrowseComp-Plus retrieval."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retriever-manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--document-index-path", type=Path, required=True)
    parser.add_argument("--snippet-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--result-count", type=int, required=True)
    parser.add_argument("--total-snippet-token-budget", type=int, required=True)
    parser.add_argument("--minimum-snippet-tokens-per-result", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the evidence bandwidth server must bind to loopback")
    runtime = EvidenceBandwidthDenseRuntime(
        manifest_path=args.manifest,
        retriever_manifest_path=args.retriever_manifest,
        candidate_id=args.candidate_id,
        model_dir=args.model_dir,
        index_root=args.index_root,
        document_index_path=args.document_index_path,
        snippet_tokenizer_dir=args.snippet_tokenizer_dir,
        result_count=args.result_count,
        total_snippet_token_budget=args.total_snippet_token_budget,
        minimum_snippet_tokens_per_result=(
            args.minimum_snippet_tokens_per_result
        ),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(
        f"evidence_bandwidth_server=http://{args.host}:{server.server_port}/search",
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
