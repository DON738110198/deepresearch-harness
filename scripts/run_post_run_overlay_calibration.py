from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.post_run_overlay import (
    load_post_run_overlay_registration,
    run_post_run_overlay_calibration,
)
from deepresearch_harness.providers import provider_from_config
from deepresearch_harness.screening_judge import VllmChatClient, load_screening_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the resumable known-case monotonic post-run overlay calibration."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    registration = load_post_run_overlay_registration(args.registration)
    settings = HarnessConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    provider_config = settings.provider
    if (
        provider_config.model != registration.provider.model
        or provider_config.base_url != registration.provider.base_url
        or provider_config.api_key_env != registration.provider.api_key_env
    ):
        raise ValueError("local provider config differs from overlay registration")

    root = args.registration.resolve().parents[2]
    manifest_path = root / registration.judge.manifest.path
    calibration_path = root / registration.judge.calibration.path
    manifest = load_screening_manifest(manifest_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "accepted_for_development_screening":
        raise ValueError("overlay Judge calibration was not accepted")
    if calibration.get("screening_manifest_sha256") != normalized_text_file_sha256(
        manifest_path
    ):
        raise ValueError("overlay Judge calibration targets another manifest")
    if manifest.engine.served_model_name != registration.judge.served_model_name:
        raise ValueError("overlay Judge served model differs from registration")
    judge_client = VllmChatClient(
        base_url=registration.judge.base_url,
        served_model_name=registration.judge.served_model_name,
        inference=manifest.inference,
        timeout_seconds=600,
        retries=0,
    )
    if registration.judge.served_model_name not in judge_client.model_ids():
        raise ValueError("registered overlay Judge is not available")

    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as error:
        raise RuntimeError("install the BrowseComp-Plus dependencies") from error
    searcher = LuceneSearcher(str((root / registration.document_index_path).resolve()))

    def load_document(docid: str) -> str | None:
        document = searcher.doc(docid)
        if document is None:
            return None
        payload = json.loads(document.raw())
        contents = payload.get("contents")
        return contents if isinstance(contents, str) and contents.strip() else None

    provider = provider_from_config(settings)
    result = run_post_run_overlay_calibration(
        registration_path=args.registration,
        output_dir=args.output_dir,
        provider=provider,
        document_loader=load_document,
        judge=judge_client.judge,
        resume=args.resume,
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(
        "literal_supported_replacements="
        f"{result.literal_supported_replacements}"
    )
    print(f"normalized_exact_matches={result.normalized_exact_matches}")
    print(f"judge_correct={result.judge_correct}")
    print(f"proposal_parse_failures={result.proposal_parse_failures}")
    print(f"unsupported_replacements={result.unsupported_replacements}")
    print(f"provider_calls={result.provider_calls}")
    print(f"provider_input_tokens={result.provider_input_tokens}")
    print(f"provider_output_tokens={result.provider_output_tokens}")
    print(f"provider_estimated_cost_usd={result.provider_estimated_cost_usd:.10f}")
    print(f"new_search_calls={result.new_search_calls}")
    print(f"document_open_calls={result.document_open_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={args.output_dir / 'summary.json'}")
    return 0 if result.decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
