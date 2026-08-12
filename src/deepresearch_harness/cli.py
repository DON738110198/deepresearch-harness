from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import HarnessConfig
from .pipeline import BaselineResearchPipeline, LocalCorpusCollector
from .providers import FakeProvider, provider_from_config


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def execute(*, question: str, corpus: Path, output_dir: Path, config: Path | None) -> int:
    settings = HarnessConfig.model_validate_json(config.read_text(encoding="utf-8")) if config else None
    provider = provider_from_config(settings)
    max_evidence = settings.run.max_evidence if settings else 6
    pipeline = BaselineResearchPipeline(provider=provider, collector=LocalCorpusCollector(corpus), output_dir=output_dir, max_evidence=max_evidence)
    state = pipeline.run(question)
    print(f"run_id={state.run_id}")
    print(f"status={state.status.value}")
    print(f"report={state.report_path}")
    print(f"tokens={state.total_usage.input_tokens + state.total_usage.output_tokens}")
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
    args = parser.parse_args()
    if args.command == "demo":
        corpus = project_root() / "examples" / "offline_corpus.json"
        pipeline = BaselineResearchPipeline(provider=FakeProvider(), collector=LocalCorpusCollector(corpus), output_dir=args.output_dir)
        state = pipeline.run("What evidence supports a phased rollout?")
        print(f"run_id={state.run_id}\nstatus={state.status.value}\nreport={state.report_path}")
        return 0
    return execute(question=args.question, corpus=args.corpus, output_dir=args.output_dir, config=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
