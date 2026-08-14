from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import FailureFocus, validate_suite_assets
from .batch import run_experiment_batch
from .browsecomp_plus import (
    fetch_development_queries,
    freeze_query_partitions,
    load_browsecomp_plus_target,
    load_deepseek_provider_snapshot,
    load_official_evaluator_manifest,
)
from .browsecomp_evaluation import (
    export_official_development_ground_truth,
    freeze_development_gold_slice,
    score_gold_diagnostic,
)
from .browsecomp_decision import decide_browsecomp_layer_promotion
from .browsecomp_judge import (
    aggregate_official_judge_results,
    prepare_official_judge_batch,
    validate_official_judge_batch,
)
from .browsecomp_repeats import aggregate_repeat_experiment
from .contracts import HarnessConfig
from .experiment import validate_experiment_manifest
from .pipeline import (
    BaselineResearchPipeline,
    LocalCorpusCollector,
    ObligationEvidenceDebtPipeline,
    PrimarySourceResearchPipeline,
    SearchWritePipeline,
)
from .pi_browsecomp import (
    audit_pi_failed_resume,
    export_pi_runs_for_official_evaluator,
    run_pi_unscored_smoke,
)
from .providers import FakeProvider, provider_from_config
from .public_benchmark import (
    load_livedrbench_manifest,
    run_livedrbench_pilot,
    score_livedrbench_predictions,
)
from .review import prepare_blind_review, score_blind_review
from .review_translation import translate_review_packet
from .review_workspace import render_review_workspace, validate_review_submission_file
from .retrieval_replay import (
    QwenDenseReplaySearcher,
    collect_frozen_search_queries,
    load_retriever_candidates,
    score_retrieval_replay,
    select_candidate,
)
from .web_research import live_collector_from_config
from .screening_judge import calibrate_screening_judge


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def execute(*, question: str, corpus: Path, output_dir: Path, config: Path | None, variant: str) -> int:
    settings = HarnessConfig.model_validate_json(config.read_text(encoding="utf-8")) if config else None
    provider = provider_from_config(settings)
    max_evidence = settings.run.max_evidence if settings else 6
    pipeline_class = {
        "b0": SearchWritePipeline,
        "b1": BaselineResearchPipeline,
        "b2": ObligationEvidenceDebtPipeline,
    }[variant]
    pipeline = pipeline_class(
        provider=provider,
        collector=LocalCorpusCollector(corpus),
        output_dir=output_dir,
        max_evidence=max_evidence,
        budget_limits=settings.run.budget if settings else None,
    )
    state = pipeline.run(question)
    print(f"run_id={state.run_id}")
    print(f"status={state.status.value}")
    print(f"report={state.report_path}")
    print(f"tokens={state.total_usage.input_tokens + state.total_usage.output_tokens}")
    print(f"estimated_cost_usd={state.total_usage.estimated_cost_usd:.8f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the training-free Deep Research baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run deterministic offline smoke demo.")
    demo.add_argument("--output-dir", type=Path, default=Path("runs"))
    run = subparsers.add_parser("run", help="Run against a supplied corpus and provider config.")
    run.add_argument("--question", required=True)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument("--variant", choices=("b0", "b1", "b2"), default="b1")
    research = subparsers.add_parser("research", help="Run a live web research task and write a Chinese cited report.")
    research.add_argument("--question", required=True)
    research.add_argument("--config", type=Path, required=True)
    research.add_argument("--context-file", type=Path)
    research.add_argument("--max-evidence", type=int)
    research.add_argument("--output-dir", type=Path, default=Path("runs") / "live")
    validate = subparsers.add_parser("validate-pilot", help="Validate a pilot suite and its corpus references.")
    validate.add_argument("--suite", type=Path, required=True)
    validate_experiment = subparsers.add_parser("validate-experiment", help="Validate a frozen experiment manifest.")
    validate_experiment.add_argument("--manifest", type=Path, required=True)
    run_experiment = subparsers.add_parser("run-experiment", help="Run both variants from a frozen manifest.")
    run_experiment.add_argument("--manifest", type=Path, required=True)
    run_experiment.add_argument("--config", type=Path, required=True)
    run_experiment.add_argument("--output-dir", type=Path, default=Path("runs") / "experiments")
    validate_public = subparsers.add_parser(
        "validate-public-benchmark",
        help="Validate a pinned LiveDRBench compatibility manifest.",
    )
    validate_public.add_argument("--manifest", type=Path, required=True)
    run_public = subparsers.add_parser(
        "run-public-benchmark",
        help="Run the pinned LiveDRBench compatibility pilot.",
    )
    run_public.add_argument("--manifest", type=Path, required=True)
    run_public.add_argument("--config", type=Path, required=True)
    run_public.add_argument("--output-dir", type=Path, default=Path("runs") / "public_benchmarks")
    score_public = subparsers.add_parser(
        "score-public-benchmark",
        help="Score saved LiveDRBench predictions without another model call.",
    )
    score_public.add_argument("--manifest", type=Path, required=True)
    score_public.add_argument("--predictions", type=Path, required=True)
    score_public.add_argument("--output", type=Path, required=True)
    validate_browsecomp_plus = subparsers.add_parser(
        "validate-browsecomp-plus-target",
        help="Validate pinned BrowseComp-Plus targets without reading benchmark gold.",
    )
    validate_browsecomp_plus.add_argument("--manifest", type=Path, required=True)
    freeze_browsecomp_plus = subparsers.add_parser(
        "freeze-browsecomp-plus-split",
        help="Freeze query-ID partitions from a TSV without persisting query text.",
    )
    freeze_browsecomp_plus.add_argument("--manifest", type=Path, required=True)
    freeze_browsecomp_plus.add_argument("--query-ids-tsv", type=Path, required=True)
    freeze_browsecomp_plus.add_argument("--output", type=Path, required=True)
    prepare_browsecomp_plus = subparsers.add_parser(
        "prepare-browsecomp-plus-dev-queries",
        help="Project and decrypt development questions only; never reads gold columns.",
    )
    prepare_browsecomp_plus.add_argument("--manifest", type=Path, required=True)
    prepare_browsecomp_plus.add_argument("--partitions", type=Path, required=True)
    prepare_browsecomp_plus.add_argument("--output", type=Path, required=True)
    prepare_browsecomp_plus.add_argument("--limit", type=int)
    prepare_browsecomp_plus.add_argument("--offset", type=int, default=0)
    run_pi_browsecomp = subparsers.add_parser(
        "run-pi-browsecomp-smoke",
        help="Run a gold-free development smoke through Pi and a recorded local retriever.",
    )
    run_pi_browsecomp.add_argument("--manifest", type=Path, required=True)
    run_pi_browsecomp.add_argument("--partitions", type=Path, required=True)
    run_pi_browsecomp.add_argument("--queries", type=Path, required=True)
    run_pi_browsecomp.add_argument("--output-dir", type=Path, required=True)
    run_pi_browsecomp.add_argument("--node", type=Path, required=True)
    run_pi_browsecomp.add_argument("--adapter-dir", type=Path, required=True)
    run_pi_browsecomp.add_argument("--search-url", default="http://127.0.0.1:8765/search")
    run_pi_browsecomp.add_argument(
        "--model",
        choices=["deepseek-v4-flash", "deepseek-v4-pro"],
        default="deepseek-v4-flash",
    )
    run_pi_browsecomp.add_argument(
        "--control-policy",
        choices=[
            "standard",
            "answer_reserve_v0",
            "answer_reserve_v1",
            "answer_reserve_nonthinking_v0",
            "first_tool_deadline_v0",
            "tool_bootstrap_v0",
            "rare_anchor_portfolio_v0",
            "constraint_portfolio_v1",
        ],
        default="standard",
    )
    run_pi_browsecomp.add_argument("--timeout-seconds", type=int, default=900)
    run_pi_browsecomp.add_argument("--retriever-id", default="bm25")
    run_pi_browsecomp.add_argument("--retriever-manifest", type=Path)
    run_pi_browsecomp.add_argument(
        "--resume-failed",
        action="store_true",
        help="Retry only failed queries after validating and archiving prior attempts.",
    )
    audit_pi_resume = subparsers.add_parser(
        "audit-pi-browsecomp-resume",
        help="Validate failed-only resume readiness without API calls or writes.",
    )
    audit_pi_resume.add_argument("--manifest", type=Path, required=True)
    audit_pi_resume.add_argument("--partitions", type=Path, required=True)
    audit_pi_resume.add_argument("--queries", type=Path, required=True)
    audit_pi_resume.add_argument("--output-dir", type=Path, required=True)
    audit_pi_resume.add_argument(
        "--search-url", default="http://127.0.0.1:8765/search"
    )
    audit_pi_resume.add_argument(
        "--model",
        choices=["deepseek-v4-flash", "deepseek-v4-pro"],
        default="deepseek-v4-flash",
    )
    audit_pi_resume.add_argument(
        "--control-policy",
        choices=[
            "standard",
            "answer_reserve_v0",
            "answer_reserve_v1",
            "answer_reserve_nonthinking_v0",
            "first_tool_deadline_v0",
            "tool_bootstrap_v0",
            "rare_anchor_portfolio_v0",
            "constraint_portfolio_v1",
        ],
        default="standard",
    )
    audit_pi_resume.add_argument("--retriever-id", default="bm25")
    audit_pi_resume.add_argument("--retriever-manifest", type=Path)
    export_pi_browsecomp = subparsers.add_parser(
        "export-pi-browsecomp-runs",
        help="Convert frozen Pi traces into the official evaluator input shape.",
    )
    export_pi_browsecomp.add_argument("--source-dir", type=Path, required=True)
    export_pi_browsecomp.add_argument("--output-dir", type=Path, required=True)
    prepare_browsecomp_gold = subparsers.add_parser(
        "prepare-browsecomp-plus-dev-gold",
        help="After predictions are frozen, decrypt only evaluator fields for development IDs.",
    )
    prepare_browsecomp_gold.add_argument("--manifest", type=Path, required=True)
    prepare_browsecomp_gold.add_argument("--partitions", type=Path, required=True)
    prepare_browsecomp_gold.add_argument("--source-summary", type=Path, required=True)
    prepare_browsecomp_gold.add_argument("--output", type=Path, required=True)
    score_browsecomp_diagnostic = subparsers.add_parser(
        "score-browsecomp-plus-diagnostic",
        help="Run strict exact-answer and retrieval diagnostics; not the official judge.",
    )
    score_browsecomp_diagnostic.add_argument("--source-dir", type=Path, required=True)
    score_browsecomp_diagnostic.add_argument("--gold-slice", type=Path, required=True)
    score_browsecomp_diagnostic.add_argument("--output", type=Path, required=True)
    export_browsecomp_ground_truth = subparsers.add_parser(
        "export-browsecomp-plus-official-ground-truth",
        help="Project a prediction-bound development gold slice into official JSONL shape.",
    )
    export_browsecomp_ground_truth.add_argument(
        "--gold-slice", type=Path, required=True
    )
    export_browsecomp_ground_truth.add_argument(
        "--output-dir", type=Path, required=True
    )
    replay_browsecomp_retrieval = subparsers.add_parser(
        "replay-browsecomp-plus-retrieval",
        help="Replay frozen agent queries through a pinned dense retriever and BM25+dense fusion.",
    )
    replay_browsecomp_retrieval.add_argument("--manifest", type=Path, required=True)
    replay_browsecomp_retrieval.add_argument(
        "--retriever-manifest", type=Path, required=True
    )
    replay_browsecomp_retrieval.add_argument("--source-dir", type=Path, required=True)
    replay_browsecomp_retrieval.add_argument("--gold-slice", type=Path, required=True)
    replay_browsecomp_retrieval.add_argument("--candidate-id", required=True)
    replay_browsecomp_retrieval.add_argument("--model-dir", type=Path, required=True)
    replay_browsecomp_retrieval.add_argument("--index-root", type=Path, required=True)
    replay_browsecomp_retrieval.add_argument("--output", type=Path, required=True)
    replay_browsecomp_retrieval.add_argument("--batch-size", type=int, default=8)
    aggregate_browsecomp_repeats = subparsers.add_parser(
        "aggregate-browsecomp-plus-repeats",
        help="Validate and aggregate paired v6 repeat trials without an official judge.",
    )
    aggregate_browsecomp_repeats.add_argument(
        "--experiment-manifest", type=Path, required=True
    )
    aggregate_browsecomp_repeats.add_argument(
        "--target-manifest", type=Path, required=True
    )
    aggregate_browsecomp_repeats.add_argument("--output", type=Path, required=True)
    aggregate_browsecomp_repeats.add_argument(
        "--validate-existing",
        action="store_true",
        help="Revalidate an existing comparison against every frozen source artifact.",
    )
    prepare_official_judge = subparsers.add_parser(
        "prepare-browsecomp-plus-official-judge-batch",
        help="Stage a self-contained, hash-bound official-judge batch from paired repeats.",
    )
    prepare_official_judge.add_argument(
        "--repeat-experiment", type=Path, required=True
    )
    prepare_official_judge.add_argument(
        "--repeat-comparison", type=Path, required=True
    )
    prepare_official_judge.add_argument(
        "--target-manifest", type=Path, required=True
    )
    prepare_official_judge.add_argument(
        "--official-evaluator", type=Path, required=True
    )
    prepare_official_judge.add_argument("--output-dir", type=Path, required=True)
    validate_official_judge = subparsers.add_parser(
        "validate-browsecomp-plus-official-judge-batch",
        help="Revalidate a staged official-judge batch without loading the judge model.",
    )
    validate_official_judge.add_argument("--batch-manifest", type=Path, required=True)
    aggregate_official_judge = subparsers.add_parser(
        "aggregate-browsecomp-plus-official-judge",
        help="Validate official Qwen judge artifacts and aggregate paired accuracy.",
    )
    aggregate_official_judge.add_argument(
        "--batch-manifest", type=Path, required=True
    )
    aggregate_official_judge.add_argument(
        "--execution-registration", type=Path, required=True
    )
    aggregate_official_judge.add_argument(
        "--execution-result", type=Path, required=True
    )
    aggregate_official_judge.add_argument("--output", type=Path, required=True)
    aggregate_official_judge.add_argument("--validate-existing", action="store_true")
    calibrate_screening_judge_parser = subparsers.add_parser(
        "calibrate-browsecomp-plus-screening-judge",
        help="Calibrate an OpenAI-compatible vLLM judge against pinned BF16 labels.",
    )
    calibrate_screening_judge_parser.add_argument(
        "--screening-manifest", type=Path, required=True
    )
    calibrate_screening_judge_parser.add_argument(
        "--screening-result", type=Path, required=True
    )
    calibrate_screening_judge_parser.add_argument(
        "--official-comparison", type=Path, required=True
    )
    calibrate_screening_judge_parser.add_argument(
        "--output", type=Path, required=True
    )
    calibrate_screening_judge_parser.add_argument(
        "--validate-existing", action="store_true"
    )
    decide_browsecomp_layer = subparsers.add_parser(
        "decide-browsecomp-plus-layer",
        help="Apply frozen development gates and classify official-judge bad cases.",
    )
    decide_browsecomp_layer.add_argument(
        "--repeat-experiment", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument(
        "--repeat-comparison", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument(
        "--target-manifest", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument(
        "--promotion-gates", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument(
        "--judge-batch-manifest", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument(
        "--judge-execution-registration", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument(
        "--judge-execution-result", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument(
        "--judge-comparison", type=Path, required=True
    )
    decide_browsecomp_layer.add_argument("--output", type=Path, required=True)
    decide_browsecomp_layer.add_argument("--validate-existing", action="store_true")
    prepare_review = subparsers.add_parser("prepare-review", help="Create blinded two-variant review artifacts.")
    prepare_review.add_argument("--summary", type=Path, required=True)
    prepare_review.add_argument("--suite", type=Path, required=True)
    prepare_review.add_argument("--output-dir", type=Path, required=True)
    prepare_review.add_argument("--seed", type=int, default=20260812)
    score_review = subparsers.add_parser("score-review", help="Validate, unblind, and aggregate a completed review.")
    score_review.add_argument("--packet", type=Path, required=True)
    score_review.add_argument("--annotations", type=Path, required=True)
    score_review.add_argument("--answer-key", type=Path, required=True)
    score_review.add_argument("--output", type=Path, required=True)
    render_review = subparsers.add_parser("render-review", help="Build a static human-review workspace from a blind packet.")
    render_review.add_argument("--packet", type=Path, required=True)
    render_review.add_argument("--output", type=Path, required=True)
    render_review.add_argument("--locale", choices=("en", "zh-CN"), default="en")
    render_review.add_argument("--translations", type=Path)
    translate_review = subparsers.add_parser("translate-review", help="Build an auditable Chinese reading-aid bundle.")
    translate_review.add_argument("--packet", type=Path, required=True)
    translate_review.add_argument("--config", type=Path, required=True)
    translate_review.add_argument("--output", type=Path, required=True)
    translate_review.add_argument("--max-chunk-characters", type=int, default=6000)
    translate_review.add_argument("--max-output-tokens", type=int, default=8192)
    validate_review = subparsers.add_parser("validate-review", help="Lock a complete review submission without reading the answer key.")
    validate_review.add_argument("--packet", type=Path, required=True)
    validate_review.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        corpus = project_root() / "examples" / "offline_corpus.json"
        pipeline = BaselineResearchPipeline(provider=FakeProvider(), collector=LocalCorpusCollector(corpus), output_dir=args.output_dir)
        state = pipeline.run("What evidence supports a phased rollout?")
        print(f"run_id={state.run_id}\nstatus={state.status.value}\nreport={state.report_path}")
        return 0
    if args.command == "validate-pilot":
        suite, corpus = validate_suite_assets(args.suite)
        counts = {focus.value: 0 for focus in FailureFocus}
        for task in suite.tasks:
            counts[task.failure_focus.value] += 1
        print(f"suite={suite.suite_id}\nstatus={suite.status}\ntasks={len(suite.tasks)}\ncorpus_records={len(corpus)}")
        print("failure_focus=" + ",".join(f"{name}:{count}" for name, count in counts.items()))
        return 0
    if args.command == "research":
        settings = HarnessConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if args.max_evidence is not None and args.max_evidence < 1:
            parser.error("--max-evidence must be at least 1")
        decision_context = args.context_file.read_text(encoding="utf-8") if args.context_file else None
        collector = live_collector_from_config(settings.search)
        pipeline = PrimarySourceResearchPipeline(
            provider=provider_from_config(settings),
            collector=collector,
            output_dir=args.output_dir,
            max_evidence=args.max_evidence or settings.run.max_evidence,
            budget_limits=settings.run.budget,
            report_language="Simplified Chinese",
        )
        state = pipeline.run(args.question, decision_context=decision_context)
        search_calls = sum(event.stage == "search" for event in state.trace)
        fetch_ok = sum(event.stage == "fetch" and event.outcome == "ok" for event in state.trace)
        print(
            f"run_id={state.run_id}\nstatus={state.status.value}\nreport={state.report_path}\n"
            f"state={args.output_dir / state.run_id / 'state.json'}\nsearch_calls={search_calls}\n"
            f"fetched_sources={fetch_ok}\n"
            f"tokens={state.total_usage.input_tokens + state.total_usage.output_tokens}\n"
            f"estimated_cost_usd={state.total_usage.estimated_cost_usd:.8f}"
        )
        return 0
    if args.command == "validate-experiment":
        manifest = validate_experiment_manifest(args.manifest)
        print(
            f"experiment={manifest.experiment_id}\nstatus={manifest.status}\n"
            f"budget_mode={manifest.budget_mode.value}\nmodel={manifest.provider.model}\nvariants={len(manifest.variants)}"
        )
        return 0
    if args.command == "validate-public-benchmark":
        manifest = load_livedrbench_manifest(args.manifest)
        print(
            f"benchmark={manifest.benchmark_id}\nstatus={manifest.status}\n"
            f"dataset={manifest.dataset_id}@{manifest.dataset_revision}\ntasks={len(manifest.task_keys)}\n"
            f"evaluator={manifest.evaluator}\nofficial_evaluator={manifest.official_evaluator_status}"
        )
        return 0
    if args.command == "run-public-benchmark":
        settings = HarnessConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        summary = run_livedrbench_pilot(
            manifest_path=args.manifest,
            config=settings,
            output_root=args.output_dir,
        )
        print(
            f"benchmark={summary.benchmark_id}\nstatus={summary.status}\noutput_dir={summary.output_dir}\n"
            f"completed={summary.completed}\nfailed={summary.failed}\n"
            f"structured_output_rate={summary.structured_output_rate:.4f}\n"
            f"official_shape_compatible_rate={summary.official_shape_compatible_rate:.4f}\n"
            f"macro_exact_precision={summary.macro_exact_precision:.4f}\n"
            f"macro_exact_recall={summary.macro_exact_recall:.4f}\n"
            f"macro_exact_f1={summary.macro_exact_f1:.4f}\n"
            f"tokens={summary.total_tokens}\ncost_usd={summary.total_estimated_cost_usd:.8f}\n"
            f"official_evaluator={summary.official_evaluator_status}"
        )
        return 0
    if args.command == "score-public-benchmark":
        scores = score_livedrbench_predictions(
            manifest_path=args.manifest,
            predictions_path=args.predictions,
            output_path=args.output,
        )
        print(
            f"benchmark={scores.benchmark_id}\noutput={args.output}\n"
            f"prediction_coverage_rate={scores.prediction_coverage_rate:.4f}\n"
            f"official_shape_compatible_rate={scores.official_shape_compatible_rate:.4f}\n"
            f"macro_exact_precision={scores.macro_exact_precision:.4f}\n"
            f"macro_exact_recall={scores.macro_exact_recall:.4f}\n"
            f"macro_exact_f1={scores.macro_exact_f1:.4f}\n"
            f"official_evaluator={scores.official_evaluator_status}"
        )
        return 0
    if args.command == "validate-browsecomp-plus-target":
        manifest = load_browsecomp_plus_target(args.manifest)
        evaluator_path = args.manifest.with_name("official_evaluator.json")
        evaluator = (
            load_official_evaluator_manifest(
                evaluator_path, target_manifest_path=args.manifest
            )
            if evaluator_path.is_file()
            else None
        )
        provider_snapshot_path = args.manifest.with_name(
            "deepseek_provider_snapshot.json"
        )
        provider_snapshot = (
            load_deepseek_provider_snapshot(
                provider_snapshot_path, target_manifest_path=args.manifest
            )
            if provider_snapshot_path.is_file()
            else None
        )
        print(
            f"benchmark={manifest.benchmark.name}\nstatus={manifest.status}\n"
            f"repository_commit={manifest.benchmark.repository_commit}\n"
            f"query_dataset={manifest.benchmark.query_dataset.name}@"
            f"{manifest.benchmark.query_dataset.revision}\n"
            f"split={manifest.split.version}\nmodels="
            + ",".join(track.model for track in manifest.model_tracks)
            + (
                f"\njudge={evaluator.judge.name}@{evaluator.judge.revision}"
                if evaluator
                else "\njudge=unbound"
            )
            + (
                "\nprovider_versions="
                + ",".join(
                    f"{model.api_model}:{model.documented_model_version}"
                    for model in provider_snapshot.models
                )
                if provider_snapshot
                else "\nprovider_versions=unbound"
            )
        )
        return 0
    if args.command == "freeze-browsecomp-plus-split":
        artifact = freeze_query_partitions(
            manifest_path=args.manifest,
            query_ids_tsv=args.query_ids_tsv,
            output_path=args.output,
        )
        print(
            f"output={args.output}\nqueries={artifact.query_count}\n"
            f"development={artifact.development_count}\n"
            f"sealed_holdout={artifact.sealed_holdout_count}\n"
            f"query_ids_sha256={artifact.query_ids_sha256}"
        )
        return 0
    if args.command == "prepare-browsecomp-plus-dev-queries":
        artifact = fetch_development_queries(
            manifest_path=args.manifest,
            partitions_path=args.partitions,
            output_path=args.output,
            limit=args.limit,
            offset=args.offset,
        )
        print(
            f"output={args.output}\npartition={artifact.partition}\n"
            f"queries={artifact.query_count}\nqueries_sha256={artifact.queries_sha256}"
        )
        return 0
    if args.command == "run-pi-browsecomp-smoke":
        summary = run_pi_unscored_smoke(
            manifest_path=args.manifest,
            partitions_path=args.partitions,
            queries_path=args.queries,
            output_dir=args.output_dir,
            node_executable=args.node,
            adapter_dir=args.adapter_dir,
            search_url=args.search_url,
            model=args.model,
            control_policy=args.control_policy,
            timeout_seconds=args.timeout_seconds,
            retriever_id=args.retriever_id,
            retriever_manifest_path=args.retriever_manifest,
            resume_failed=args.resume_failed,
        )
        print(
            f"output={args.output_dir}\nstatus={summary.status}\n"
            f"control_policy={summary.control_policy}\n"
            f"retriever_id={summary.retriever_id}\n"
            f"queries={summary.query_count}\nsucceeded={summary.succeeded}\n"
            f"budget_exhausted={summary.budget_exhausted}\nfailed={summary.failed}\n"
            f"schema_complete={summary.schema_complete}\n"
            f"resume_count={summary.resume_count}\n"
            f"attempts={sum(item.attempt_count for item in summary.items)}\n"
            "answer_compiler_invocations="
            f"{summary.answer_compiler_invocations}\n"
            f"search_calls={summary.total_search_calls}\n"
            f"output_tokens={summary.total_output_tokens}\n"
            "output_budget_overshoot_tokens="
            f"{summary.total_output_budget_overshoot_tokens}\n"
            f"tokens={summary.total_tokens}\ncost_usd={summary.total_cost_usd:.8f}\n"
            "gold_accessed=false"
        )
        return 0 if summary.succeeded == summary.query_count else 1
    if args.command == "audit-pi-browsecomp-resume":
        audit = audit_pi_failed_resume(
            manifest_path=args.manifest,
            partitions_path=args.partitions,
            queries_path=args.queries,
            output_dir=args.output_dir,
            search_url=args.search_url,
            model=args.model,
            control_policy=args.control_policy,
            retriever_id=args.retriever_id,
            retriever_manifest_path=args.retriever_manifest,
        )
        print(
            f"status={audit.status}\n"
            f"source_summary_sha256={audit.source_summary_sha256}\n"
            f"queries={audit.query_count}\n"
            f"immutable_queries={audit.immutable_query_count}\n"
            f"retry_eligible={audit.retry_eligible_count}\n"
            f"retry_query_ids_sha256={audit.retry_query_ids_sha256}\n"
            f"resume_count={audit.resume_count}\n"
            f"attempts={audit.total_attempts}\n"
            f"tokens={audit.cumulative_total_tokens}\n"
            f"cost_usd={audit.cumulative_cost_usd:.8f}\n"
            f"provider_calls={audit.provider_calls}\n"
            "gold_accessed=false"
        )
        return 0
    if args.command == "export-pi-browsecomp-runs":
        export = export_pi_runs_for_official_evaluator(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
        )
        print(
            f"output={args.output_dir}\nqueries={export.query_count}\n"
            f"completed={export.completed}\nincomplete={export.incomplete}\n"
            f"source_summary_sha256={export.source_summary_sha256}"
        )
        return 0
    if args.command == "prepare-browsecomp-plus-dev-gold":
        gold = freeze_development_gold_slice(
            manifest_path=args.manifest,
            partitions_path=args.partitions,
            source_summary_path=args.source_summary,
            output_path=args.output,
        )
        print(
            f"output={args.output}\nqueries={gold.query_count}\n"
            f"source_summary_sha256={gold.source_summary_sha256}\n"
            f"prediction_set_sha256={gold.prediction_set_sha256}\n"
            "partition=development\ngold_accessed=true"
        )
        return 0
    if args.command == "score-browsecomp-plus-diagnostic":
        diagnostic = score_gold_diagnostic(
            source_dir=args.source_dir,
            gold_slice_path=args.gold_slice,
            output_path=args.output,
        )
        print(
            f"output={args.output}\nstatus={diagnostic.status}\n"
            f"queries={diagnostic.query_count}\n"
            f"schema_complete={diagnostic.schema_complete}\n"
            f"normalized_exact_match={diagnostic.normalized_exact_match}\n"
            f"normalized_exact_match_percent="
            f"{diagnostic.normalized_exact_match_percent:.2f}\n"
            f"evidence_recall_percent={diagnostic.evidence_recall_percent}\n"
            f"gold_recall_percent={diagnostic.gold_recall_percent}\n"
            f"official_accuracy_status={diagnostic.official_accuracy_status}"
        )
        return 0
    if args.command == "export-browsecomp-plus-official-ground-truth":
        export = export_official_development_ground_truth(
            gold_slice_path=args.gold_slice,
            output_dir=args.output_dir,
        )
        print(
            f"output={args.output_dir}\nqueries={export.query_count}\n"
            f"gold_slice_sha256={export.gold_slice_sha256}\n"
            f"ground_truth_sha256={export.ground_truth_sha256}\n"
            "partition=development"
        )
        return 0
    if args.command == "replay-browsecomp-plus-retrieval":
        retriever_manifest = load_retriever_candidates(
            args.retriever_manifest, target_manifest_path=args.manifest
        )
        candidate = select_candidate(retriever_manifest, args.candidate_id)
        searcher = QwenDenseReplaySearcher(
            candidate=candidate,
            model_dir=args.model_dir,
            index_root=args.index_root,
            batch_size=args.batch_size,
        )
        frozen_queries = collect_frozen_search_queries(args.source_dir)
        candidate_results = searcher.search_many(frozen_queries)
        replay = score_retrieval_replay(
            source_dir=args.source_dir,
            gold_slice_path=args.gold_slice,
            retriever_manifest_path=args.retriever_manifest,
            target_manifest_path=args.manifest,
            candidate_id=args.candidate_id,
            candidate_results=candidate_results,
            runtime=searcher.runtime_snapshot,
            output_path=args.output,
        )
        print(
            f"output={args.output}\nstatus={replay.status}\n"
            f"candidate={replay.candidate_id}\nqueries={replay.query_count}\n"
            f"search_calls={replay.search_calls}\n"
            f"unique_search_queries={replay.unique_search_queries}\n"
            f"baseline_evidence_recall_percent="
            f"{replay.baseline_evidence_recall_percent}\n"
            f"candidate_evidence_recall_percent="
            f"{replay.candidate_evidence_recall_percent}\n"
            f"fused_evidence_recall_percent={replay.fused_evidence_recall_percent}\n"
            f"evidence_recall_delta_candidate_pp="
            f"{replay.evidence_recall_delta_candidate_pp}\n"
            f"evidence_recall_delta_fused_pp="
            f"{replay.evidence_recall_delta_fused_pp}\n"
            "official_accuracy_status=planned_not_run"
        )
        return 0
    if args.command == "aggregate-browsecomp-plus-repeats":
        comparison = aggregate_repeat_experiment(
            manifest_path=args.experiment_manifest,
            target_manifest_path=args.target_manifest,
            output_path=args.output,
            validate_existing=args.validate_existing,
        )
        print(
            f"output={args.output}\nstatus={comparison.status}\n"
            f"trials={comparison.trial_count}\n"
            f"queries_per_trial={comparison.queries_per_trial}\n"
            f"paired_query_observations={comparison.paired_query_observations}\n"
            f"baseline_exact_mean="
            f"{comparison.baseline.strict_exact_percent.mean:.2f}\n"
            f"candidate_exact_mean="
            f"{comparison.candidate.strict_exact_percent.mean:.2f}\n"
            f"baseline_evidence_recall_mean="
            f"{comparison.baseline.evidence_recall_percent.mean:.2f}\n"
            f"candidate_evidence_recall_mean="
            f"{comparison.candidate.evidence_recall_percent.mean:.2f}\n"
            "official_accuracy_status=planned_not_run"
        )
        return 0
    if args.command == "prepare-browsecomp-plus-official-judge-batch":
        batch = prepare_official_judge_batch(
            repeat_experiment_path=args.repeat_experiment,
            repeat_comparison_path=args.repeat_comparison,
            target_manifest_path=args.target_manifest,
            official_evaluator_path=args.official_evaluator,
            output_dir=args.output_dir,
        )
        print(
            f"batch_manifest={args.output_dir / 'batch_manifest.json'}\n"
            f"status={batch.status}\ntrials={batch.trial_count}\n"
            f"queries_per_trial={batch.queries_per_trial}\n"
            f"inputs={batch.input_count}\ninput_set_sha256={batch.input_set_sha256}\n"
            "official_accuracy_status=planned_not_run"
        )
        return 0
    if args.command == "validate-browsecomp-plus-official-judge-batch":
        batch = validate_official_judge_batch(args.batch_manifest)
        print(
            f"batch_manifest={args.batch_manifest}\nstatus={batch.status}\n"
            f"inputs={batch.input_count}\ninput_set_sha256={batch.input_set_sha256}"
        )
        return 0
    if args.command == "aggregate-browsecomp-plus-official-judge":
        comparison = aggregate_official_judge_results(
            batch_manifest_path=args.batch_manifest,
            execution_registration_path=args.execution_registration,
            execution_result_path=args.execution_result,
            output_path=args.output,
            validate_existing=args.validate_existing,
        )
        print(
            f"output={args.output}\nstatus={comparison.status}\n"
            f"evaluations={comparison.evaluations}\n"
            f"baseline_accuracy_mean={comparison.baseline.trial_accuracy_percent.mean:.2f}\n"
            f"candidate_accuracy_mean={comparison.candidate.trial_accuracy_percent.mean:.2f}\n"
            f"candidate_wins={comparison.paired.candidate_wins}\n"
            f"baseline_wins={comparison.paired.baseline_wins}\n"
            f"ties={comparison.paired.ties}\nleaderboard_status=not_submitted"
        )
        return 0
    if args.command == "calibrate-browsecomp-plus-screening-judge":
        calibration = calibrate_screening_judge(
            screening_manifest_path=args.screening_manifest,
            screening_result_path=args.screening_result,
            official_comparison_path=args.official_comparison,
            output_path=args.output,
            validate_existing=args.validate_existing,
        )
        print(
            f"output={args.output}\nstatus={calibration['status']}\n"
            f"evaluations={calibration['evaluations']}\n"
            "label_agreement_percent="
            f"{calibration['label_agreement_percent']:.2f}\n"
            f"cohens_kappa={calibration['cohens_kappa']:.4f}\n"
            "absolute_pooled_accuracy_delta_pp="
            f"{calibration['absolute_pooled_accuracy_delta_pp']:.2f}\n"
            "paired_variant_delta_sign_match="
            f"{str(calibration['paired_variant_delta_sign_match']).lower()}\n"
            "official_status=reference_bf16_development_slice\n"
            f"screening_status={calibration['screening_status']}"
        )
        return 0 if calibration["status"] == "accepted_for_development_screening" else 1
    if args.command == "decide-browsecomp-plus-layer":
        decision = decide_browsecomp_layer_promotion(
            repeat_experiment_path=args.repeat_experiment,
            repeat_comparison_path=args.repeat_comparison,
            target_manifest_path=args.target_manifest,
            promotion_gates_path=args.promotion_gates,
            judge_batch_manifest_path=args.judge_batch_manifest,
            judge_execution_registration_path=args.judge_execution_registration,
            judge_execution_result_path=args.judge_execution_result,
            judge_comparison_path=args.judge_comparison,
            output_path=args.output,
            validate_existing=args.validate_existing,
        )
        print(
            f"output={args.output}\ndecision={decision.decision}\n"
            f"claim_qualifier={decision.claim_qualifier}\n"
            f"evidence_recall_delta_pp={decision.evidence_recall_delta_pp:.2f}\n"
            f"official_accuracy_delta_pp={decision.official_accuracy_delta_pp:.2f}\n"
            f"persistent_failure_queries="
            f"{decision.failure_aggregate.persistent_failure_queries}\n"
            f"next_action={decision.next_action}\nleaderboard_status=not_submitted"
        )
        return 0
    if args.command == "run-experiment":
        settings = HarnessConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        summary = run_experiment_batch(
            manifest_path=args.manifest,
            config=settings,
            output_root=args.output_dir,
        )
        print(f"experiment={summary.experiment_id}\nstatus={summary.status}\noutput_dir={summary.output_dir}")
        for variant, aggregate in summary.aggregates.items():
            print(
                f"{variant}: completed={aggregate.completed},failed={aggregate.failed},"
                f"tokens={aggregate.total_tokens},cost_usd={aggregate.total_estimated_cost_usd:.8f},"
                f"evidence_recall={aggregate.mean_evidence_id_recall},"
                f"evidence_precision={aggregate.mean_evidence_id_precision}"
            )
        return 0
    if args.command == "prepare-review":
        packet, template, key = prepare_blind_review(
            summary_path=args.summary,
            suite_path=args.suite,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(f"review_packet={packet}\nannotation_template={template}\nanswer_key={key}")
        return 0
    if args.command == "score-review":
        result = score_blind_review(
            packet_path=args.packet,
            annotations_path=args.annotations,
            answer_key_path=args.answer_key,
            output_path=args.output,
        )
        print(
            f"experiment={result.experiment_id}\nreviewer_type={result.reviewer_type.value}\n"
            f"result_status={result.result_status}\noutput={args.output}"
        )
        for variant, aggregate in result.aggregates.items():
            print(
                f"{variant}: candidates={aggregate.candidate_count},"
                f"obligation_coverage={aggregate.mean_obligation_coverage},"
                f"citation_support={aggregate.mean_citation_support_rate},"
                f"unsupported_claims={aggregate.mean_unsupported_claim_rate},"
                f"irrelevant_claims={aggregate.mean_irrelevant_claim_rate},"
                f"conflict_handling={aggregate.mean_conflict_handling_rate}"
            )
        return 0
    if args.command == "render-review":
        output = render_review_workspace(
            packet_path=args.packet,
            output_path=args.output,
            locale=args.locale,
            translations_path=args.translations,
        )
        print(f"review_workspace={output}")
        return 0
    if args.command == "translate-review":
        settings = HarnessConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        provider = provider_from_config(settings)
        bundle = translate_review_packet(
            packet_path=args.packet,
            output_path=args.output,
            provider=provider,
            max_chunk_characters=args.max_chunk_characters,
            max_output_tokens=args.max_output_tokens,
        )
        print(
            f"translations={args.output}\nlocale={bundle.locale}\nentries={len(bundle.entries)}\n"
            f"provider={bundle.provider}\nmodel={bundle.model}\ncalls={len(bundle.trace)}\n"
            f"tokens={bundle.total_usage.input_tokens + bundle.total_usage.output_tokens}\n"
            f"estimated_cost_usd={bundle.total_usage.estimated_cost_usd:.8f}"
        )
        return 0
    if args.command == "validate-review":
        try:
            count, annotations_hash, reviewer_type = validate_review_submission_file(
                packet_path=args.packet,
                annotations_path=args.annotations,
            )
        except ValueError as error:
            print(f"validation_error={error}", file=sys.stderr)
            return 2
        print(
            f"validated_candidates={count}\nreviewer_type={reviewer_type}\n"
            f"annotations_sha256={annotations_hash}"
        )
        return 0
    return execute(
        question=args.question,
        corpus=args.corpus,
        output_dir=args.output_dir,
        config=args.config,
        variant=args.variant,
    )


if __name__ == "__main__":
    raise SystemExit(main())
