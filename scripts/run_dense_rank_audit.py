from __future__ import annotations

import argparse
import os
from pathlib import Path

from deepresearch_harness.dense_rank_audit import (
    collect_dense_rank_cases,
    load_dense_rank_registration,
    run_dense_rank_audit,
)
from deepresearch_harness.retrieval_replay import (
    QwenDenseReplaySearcher,
    RetrieverCandidatesManifest,
    select_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pinned dense gold-document ranks for the frozen persistent-miss "
            "queries without provider, online search, Judge, GPU, or holdout access."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    registration_path = args.registration.resolve()
    output_path = args.output.resolve()
    registration = load_dense_rank_registration(registration_path)
    root = registration_path.parents[2]
    manifest_path = (root / registration.retriever_manifest.path).resolve()
    manifest = RetrieverCandidatesManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    candidate = select_candidate(manifest, registration.candidate_id)
    frozen = collect_dense_rank_cases(registration_path)
    queries = [query.query for item in frozen for query in item.queries]

    searcher = QwenDenseReplaySearcher(
        candidate=candidate,
        model_dir=(root / registration.model_directory).resolve(),
        index_root=(root / registration.index_root).resolve(),
        batch_size=registration.batch_size,
        search_depth=registration.depths[-1],
    )
    if searcher.runtime_snapshot.device != "cpu":
        raise ValueError("dense-rank audit did not honor the frozen CPU contract")
    results = searcher.search_many(queries)
    result = run_dense_rank_audit(
        registration_path=registration_path,
        candidate_results=results,
        runtime=searcher.runtime_snapshot,
        output_path=output_path,
    )
    print(f"decision={result.decision}")
    print(f"query_count={result.query_count}")
    print(f"generated_query_count={result.generated_query_count}")
    print(f"unique_dense_query_count={result.unique_dense_query_count}")
    print(f"dense_gold_hit_cases_at20={result.dense_gold_hit_cases_at20}")
    print(f"dense_gold_hit_cases_at100={result.dense_gold_hit_cases_at100}")
    print(f"dense_gold_hit_cases_at1000={result.dense_gold_hit_cases_at1000}")
    print(f"passage_gold_hit_cases_at20={result.passage_gold_hit_cases_at20}")
    print(f"load_latency_ms={result.runtime.load_latency_ms}")
    print(f"search_latency_ms={result.runtime.search_latency_ms}")
    print(f"provider_calls={result.provider_calls}")
    print(f"online_search_calls={result.online_search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
