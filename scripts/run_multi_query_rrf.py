from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.multi_query_rrf import (
    MultiQueryRrfRegistration,
    build_multi_query_rrf_slate,
    score_multi_query_rrf_slate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and score a gold-blind multi-query RRF slate."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--slate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.slate.exists():
        registration = MultiQueryRrfRegistration.model_validate_json(
            args.registration.read_text(encoding="utf-8")
        )
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as error:
            raise RuntimeError("install the BrowseComp-Plus dependencies") from error
        root = args.registration.resolve().parents[2]
        searcher = LuceneSearcher(
            str((root / registration.retrieval.document_index_path).resolve())
        )
        searcher.set_bm25(
            registration.retrieval.bm25_k1, registration.retrieval.bm25_b
        )

        def search(query: str, limit: int) -> tuple[str, ...]:
            return tuple(str(hit.docid) for hit in searcher.search(query, k=limit))

        slate = build_multi_query_rrf_slate(
            registration_path=args.registration,
            output_path=args.slate,
            search=search,
        )
        print(f"slate_frozen_queries={slate.frozen_query_count}")
        print(f"slate={args.slate}")
    result = score_multi_query_rrf_slate(
        registration_path=args.registration,
        slate_path=args.slate,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(f"single_query_gold_hit_cases_at20={result.single_query_gold_hit_cases_at20}")
    print(f"single_query_gold_hit_cases_at100={result.single_query_gold_hit_cases_at100}")
    print(f"fused_gold_hit_cases_at20={result.fused_gold_hit_cases_at20}")
    print(f"fused_gold_hit_cases_at100={result.fused_gold_hit_cases_at100}")
    print(f"offline_bm25_queries={result.offline_bm25_queries}")
    print(f"provider_calls={result.provider_calls}")
    print(f"online_search_calls={result.online_search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
