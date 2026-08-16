from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.visible_pivot_bridge import (
    load_visible_pivot_registration,
    run_visible_pivot_oracle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the registered CPU-only visible-pivot bridge oracle without "
            "provider, online search, Judge, GPU, or holdout access."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        from pyserini.index.lucene import IndexReader
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as error:
        raise RuntimeError("install the BrowseComp-Plus dependencies") from error

    registration_path = args.registration.resolve()
    registration = load_visible_pivot_registration(registration_path)
    root = registration_path.parents[2]
    index_path = (root / registration.document_index_path).resolve()
    searcher = LuceneSearcher(str(index_path))
    searcher.set_bm25(registration.retrieval.bm25_k1, registration.retrieval.bm25_b)
    if searcher.num_docs != registration.document_count:
        raise ValueError("visible-pivot Lucene document count changed")
    reader = IndexReader(str(index_path))

    def load_document(docid: str) -> str:
        document = searcher.doc(docid)
        if document is None:
            raise ValueError(f"visible-pivot document is missing: {docid}")
        raw = json.loads(document.raw())
        contents = raw.get("contents")
        if not isinstance(contents, str):
            raise ValueError(f"visible-pivot document contents are invalid: {docid}")
        return contents

    result = run_visible_pivot_oracle(
        registration_path=registration_path,
        output_path=args.output.resolve(),
        analyze=reader.analyze,
        load_document=load_document,
        document_frequency=lambda term: int(reader.get_term_counts(term)[0]),
        search=lambda query, limit: tuple(
            str(hit.docid) for hit in searcher.search(query, k=limit)
        ),
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"generated_query_count={result.generated_query_count}")
    print(f"baseline_gold_hit_cases_at20={result.baseline_gold_hit_cases_at20}")
    print(
        "visible_pivot_gold_hit_cases_at20="
        f"{result.visible_pivot_gold_hit_cases_at20}"
    )
    print(f"cases_with_candidate_pivots={result.cases_with_candidate_pivots}")
    print(f"baseline_bm25_queries={result.baseline_bm25_queries}")
    print(f"pivot_bm25_queries={result.pivot_bm25_queries}")
    print(f"offline_bm25_queries={result.offline_bm25_queries}")
    print(f"provider_calls={result.provider_calls}")
    print(f"online_search_calls={result.online_search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
