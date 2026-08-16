from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.raw_question_dense_rank import (
    build_raw_question_dense_slate,
    collect_raw_questions,
    load_raw_question_builder_registration,
    score_raw_question_dense_slate,
)
from deepresearch_harness.retrieval_replay import (
    QwenDenseReplaySearcher,
    RetrieverCandidatesManifest,
    select_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and score the gold-blind raw-question dense-rank audit."
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--slate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.slate.exists():
        registration = load_raw_question_builder_registration(args.registration)
        root = args.registration.resolve().parents[2]
        manifest = RetrieverCandidatesManifest.model_validate_json(
            (root / registration.retriever_manifest.path).read_text(encoding="utf-8")
        )
        candidate = select_candidate(manifest, registration.candidate_id)
        searcher = QwenDenseReplaySearcher(
            candidate=candidate,
            model_dir=root / registration.model_directory,
            index_root=root / registration.index_root,
            batch_size=registration.batch_size,
            search_depth=registration.depths[-1],
        )
        questions = collect_raw_questions(args.registration)
        results = searcher.search_many(questions)
        slate = build_raw_question_dense_slate(
            registration_path=args.registration,
            output_path=args.slate,
            candidate_results=results,
            runtime=searcher.runtime_snapshot,
        )
        print(f"slate_query_count={slate.query_count}")
        print(f"dense_query_encodes={slate.dense_query_encodes}")
        print(f"search_latency_ms={slate.runtime.search_latency_ms}")
        print(f"slate={args.slate}")

    result = score_raw_question_dense_slate(
        registration_path=args.registration,
        slate_path=args.slate,
        output_path=args.output,
    )
    print(f"decision={result.decision}")
    print(
        "generated_query_gold_hit_cases_at20="
        f"{result.generated_query_gold_hit_cases_at20}"
    )
    print(
        "raw_question_gold_hit_cases_at20="
        f"{result.raw_question_gold_hit_cases_at20}"
    )
    print(
        "raw_question_gold_hit_cases_at100="
        f"{result.raw_question_gold_hit_cases_at100}"
    )
    print(
        "raw_question_gold_hit_cases_at1000="
        f"{result.raw_question_gold_hit_cases_at1000}"
    )
    print(f"raw_rank_wins={result.raw_rank_wins}")
    print(f"raw_rank_losses={result.raw_rank_losses}")
    print(f"raw_rank_ties={result.raw_rank_ties}")
    print(f"provider_calls={result.provider_calls}")
    print(f"online_search_calls={result.online_search_calls}")
    print(f"judge_calls={result.judge_calls}")
    print(f"gpu_used={str(result.gpu_used).lower()}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
