from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from deepresearch_harness.gold_blind_pivot_slate import (
    GoldBlindPivotSlate,
    build_gold_blind_pivot_slate,
    load_gold_blind_pivot_registration,
    score_gold_blind_pivot_slate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a persisted no-gold visible-pivot slate, then score its fixed "
            "provenance queries without provider, online search, Judge, or GPU."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        from pyserini.index.lucene import LuceneIndexReader
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as error:
        raise RuntimeError("install the BrowseComp-Plus dependencies") from error

    registration_path = args.registration.resolve()
    registration = load_gold_blind_pivot_registration(registration_path)
    root = registration_path.parents[2]
    source = json.loads(
        (root / registration.source_registration.path).read_text(encoding="utf-8")
    )
    index_path = (root / source["document_index_path"]).resolve()
    output_dir = args.output_dir.resolve()
    slate_path = output_dir / "slate.json"
    audit_path = output_dir / "audit.json"
    if audit_path.exists():
        raise ValueError("gold-blind pivot audit already exists")

    reader = LuceneIndexReader(str(index_path))
    if slate_path.exists():
        slate = GoldBlindPivotSlate.model_validate_json(
            slate_path.read_text(encoding="utf-8")
        )
        if slate.registration.sha256 != sha256(registration_path.read_bytes()).hexdigest():
            raise ValueError("saved gold-blind pivot slate targets another registration")
    else:
        slate = build_gold_blind_pivot_slate(
            registration_path=registration_path,
            output_path=slate_path,
            analyze=reader.analyze,
            document_frequency=lambda term: int(
                reader.get_term_counts(term, analyzer=None)[0]
            ),
        )

    searcher = LuceneSearcher(str(index_path))
    searcher.set_bm25(registration.retrieval.bm25_k1, registration.retrieval.bm25_b)
    if searcher.num_docs != int(source["document_count"]):
        raise ValueError("gold-blind pivot Lucene document count changed")
    result = score_gold_blind_pivot_slate(
        registration_path=registration_path,
        slate_path=slate_path,
        output_path=audit_path,
        analyze=reader.analyze,
        search=lambda query, limit: tuple(
            str(hit.docid) for hit in searcher.search(query, k=limit)
        ),
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"selected_pivot_count={slate.selected_pivot_count}")
    print(f"selection_failures={result.selection_failures}")
    print(f"selector_gold_hit_cases_at20={result.selector_gold_hit_cases_at20}")
    print(f"retained_oracle_rescue_cases={result.retained_oracle_rescue_cases}")
    print(f"offline_bm25_queries={result.offline_bm25_queries}")
    print(f"provider_calls={result.provider_calls}")
    print(f"online_search_calls={result.online_search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"slate={slate_path}")
    print(f"output={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
