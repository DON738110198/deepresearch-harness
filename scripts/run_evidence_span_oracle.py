from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.evidence_span_oracle import (
    load_evidence_span_oracle_registration,
    run_evidence_span_oracle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered posthoc evidence-span availability oracle."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration = load_evidence_span_oracle_registration(args.registration)

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

    result = run_evidence_span_oracle(
        registration_path=args.registration,
        output_path=args.output,
        document_loader=load_document,
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"full_document_hit_cases={result.full_document_hit_cases}")
    print(f"head_span_hit_cases={result.head_span_hit_cases}")
    print(f"selected_span_hit_cases={result.selected_span_hit_cases}")
    print(
        "uninspected_selected_span_hit_cases="
        f"{result.uninspected_selected_span_hit_cases}"
    )
    print(f"selected_over_head_delta={result.selected_over_head_delta}")
    print(f"provider_calls={result.provider_calls}")
    print(f"search_calls={result.search_calls}")
    print(f"output={args.output}")
    return 0 if result.decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
