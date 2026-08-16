from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.dense_document_visibility import (
    run_dense_document_visibility_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit answer visibility under the pinned 4096-token dense document recipe."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import transformers
        from pyserini.search.lucene import LuceneSearcher
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install the BrowseComp-Plus dependencies") from error

    registration_payload = json.loads(args.registration.read_text(encoding="utf-8"))
    root = args.registration.resolve().parents[2]
    model_directory = (root / registration_payload["model_directory"]).resolve()
    document_index = (root / registration_payload["document_index_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.truncation_side = "right"
    searcher = LuceneSearcher(str(document_index))

    def load_document(docid: str) -> str | None:
        document = searcher.doc(docid)
        if document is None:
            return None
        payload = json.loads(document.raw())
        contents = payload.get("contents")
        return contents if isinstance(contents, str) and contents.strip() else None

    result = run_dense_document_visibility_audit(
        registration_path=args.registration,
        output_path=args.output,
        tokenizer=tokenizer,
        tokenizer_runtime_version=transformers.__version__,
        document_loader=load_document,
    )
    print(f"decision={result.decision}")
    print(f"provenance_status={result.provenance_status}")
    print(f"query_count={result.query_count}")
    print(f"gold_document_count={result.gold_document_count}")
    print(f"visible_cases={result.visible_cases}")
    print(f"hidden_cases={result.hidden_cases}")
    print(f"visible_documents={result.visible_documents}")
    print(f"truncated_documents={result.truncated_documents}")
    print(f"provider_calls={result.provider_calls}")
    print(f"embedding_model_calls={result.embedding_model_calls}")
    print(f"search_calls={result.search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"gpu_calls={result.gpu_calls}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
