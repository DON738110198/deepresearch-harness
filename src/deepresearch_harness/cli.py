from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import FailureFocus, validate_suite_assets
from .batch import run_experiment_batch
from .contracts import HarnessConfig
from .experiment import validate_experiment_manifest
from .pipeline import (
    BaselineResearchPipeline,
    LocalCorpusCollector,
    ObligationEvidenceDebtPipeline,
    PrimarySourceResearchPipeline,
    SearchWritePipeline,
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
from .web_research import live_collector_from_config


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
