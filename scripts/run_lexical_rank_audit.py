from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.lexical_rank_audit import (
    load_lexical_rank_registration,
    run_lexical_rank_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare raw-question and generated-query BM25 gold ranks."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration = load_lexical_rank_registration(args.registration)
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as error:
        raise RuntimeError("install the BrowseComp-Plus dependencies") from error
    root = args.registration.resolve().parents[2]
    searcher = LuceneSearcher(str((root / registration.document_index_path).resolve()))

    def search(query: str, limit: int) -> tuple[str, ...]:
        return tuple(str(hit.docid) for hit in searcher.search(query, k=limit))

    result = run_lexical_rank_audit(
        registration_path=args.registration,
        output_path=args.output,
        search=search,
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"raw_question_top5_cases={result.raw_question_top5_cases}")
    print(f"raw_question_top20_cases={result.raw_question_top20_cases}")
    print(f"raw_question_top100_cases={result.raw_question_top100_cases}")
    print(f"raw_question_top1000_cases={result.raw_question_top1000_cases}")
    print(f"generated_query_top5_cases={result.generated_query_top5_cases}")
    print(f"generated_query_top20_cases={result.generated_query_top20_cases}")
    print(f"generated_query_top100_cases={result.generated_query_top100_cases}")
    print(f"generated_query_top1000_cases={result.generated_query_top1000_cases}")
    print(f"raw_question_better_cases={result.raw_question_better_cases}")
    print(f"offline_bm25_queries={result.offline_bm25_queries}")
    print(f"provider_calls={result.provider_calls}")
    print(f"online_search_calls={result.online_search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
