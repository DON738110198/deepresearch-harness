from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import FailureFocus, validate_suite_assets
from .batch import run_experiment_batch
from .contracts import HarnessConfig
from .experiment import validate_experiment_manifest
from .pipeline import BaselineResearchPipeline, LocalCorpusCollector, SearchWritePipeline
from .providers import FakeProvider, provider_from_config
from .review import prepare_blind_review


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def execute(*, question: str, corpus: Path, output_dir: Path, config: Path | None, variant: str) -> int:
    settings = HarnessConfig.model_validate_json(config.read_text(encoding="utf-8")) if config else None
    provider = provider_from_config(settings)
    max_evidence = settings.run.max_evidence if settings else 6
    pipeline_class = SearchWritePipeline if variant == "b0" else BaselineResearchPipeline
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
    run.add_argument("--variant", choices=("b0", "b1"), default="b1")
    validate = subparsers.add_parser("validate-pilot", help="Validate a pilot suite and its corpus references.")
    validate.add_argument("--suite", type=Path, required=True)
    validate_experiment = subparsers.add_parser("validate-experiment", help="Validate a frozen experiment manifest.")
    validate_experiment.add_argument("--manifest", type=Path, required=True)
    run_experiment = subparsers.add_parser("run-experiment", help="Run all B0/B1 tasks from a frozen manifest.")
    run_experiment.add_argument("--manifest", type=Path, required=True)
    run_experiment.add_argument("--config", type=Path, required=True)
    run_experiment.add_argument("--output-dir", type=Path, default=Path("runs") / "experiments")
    prepare_review = subparsers.add_parser("prepare-review", help="Create blinded B0/B1 review artifacts.")
    prepare_review.add_argument("--summary", type=Path, required=True)
    prepare_review.add_argument("--suite", type=Path, required=True)
    prepare_review.add_argument("--output-dir", type=Path, required=True)
    prepare_review.add_argument("--seed", type=int, default=20260812)
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
    if args.command == "validate-experiment":
        manifest = validate_experiment_manifest(args.manifest)
        print(
            f"experiment={manifest.experiment_id}\nstatus={manifest.status}\n"
            f"budget_mode={manifest.budget_mode.value}\nmodel={manifest.provider.model}\nvariants={len(manifest.variants)}"
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
    return execute(
        question=args.question,
        corpus=args.corpus,
        output_dir=args.output_dir,
        config=args.config,
        variant=args.variant,
    )


if __name__ == "__main__":
    raise SystemExit(main())
