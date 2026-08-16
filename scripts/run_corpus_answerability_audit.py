from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.corpus_answerability import (
    load_corpus_answerability_registration,
    run_corpus_answerability_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit literal answerability of persistent retrieval misses."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration = load_corpus_answerability_registration(args.registration)
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as error:
        raise RuntimeError("install the BrowseComp-Plus dependencies") from error
    root = args.registration.resolve().parents[2]
    searcher = LuceneSearcher(str((root / registration.document_index_path).resolve()))

    def load_document(docid: str) -> str | None:
        document = searcher.doc(docid)
        if document is None:
            return None
        payload = json.loads(document.raw())
        contents = payload.get("contents")
        return contents if isinstance(contents, str) and contents.strip() else None

    result = run_corpus_answerability_audit(
        registration_path=args.registration,
        output_path=args.output,
        document_loader=load_document,
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"answerable_cases={result.answerable_cases}")
    print(f"literal_absent_cases={result.literal_absent_cases}")
    print(
        "cases_with_missing_gold_documents="
        f"{result.cases_with_missing_gold_documents}"
    )
    print(f"gold_document_count={result.gold_document_count}")
    print(f"missing_gold_documents={result.missing_gold_documents}")
    print(f"provider_calls={result.provider_calls}")
    print(f"search_calls={result.search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
