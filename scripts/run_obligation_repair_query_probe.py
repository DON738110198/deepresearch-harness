from __future__ import annotations

import argparse
from pathlib import Path

from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.providers import provider_from_config
from deepresearch_harness.repair_query_probe import load_registration, run_probe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a preregistered, prediction-bound obligation repair-query probe. "
            "Gold is used only after retrieval to score known development cases."
        )
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-timeout-seconds", type=int, default=60)
    args = parser.parse_args()

    registration = load_registration(args.registration)
    settings = HarnessConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    if settings.provider.thinking_mode != registration.fixed_contract.thinking_mode:
        parser.error("provider thinking mode differs from the registered contract")
    if settings.provider.base_url != "https://api.deepseek.com":
        parser.error("provider base URL differs from the registered DeepSeek endpoint")
    provider = provider_from_config(settings)
    result = run_probe(
        registration_path=args.registration,
        provider=provider,
        output_path=args.output,
        timeout_seconds=args.search_timeout_seconds,
    )
    print(f"decision={result.decision}")
    print(f"queries={result.query_count}")
    print(f"provider_attempts={result.provider_attempts}")
    print(f"search_calls={result.search_calls}")
    print(f"parse_failures={result.parse_failures}")
    print(f"request_failures={result.request_failures}")
    print(f"baseline_gold_hit_cases={result.baseline_gold_hit_cases}")
    print(f"candidate_gold_hit_cases={result.candidate_gold_hit_cases}")
    print(f"gold_hit_case_delta={result.gold_hit_case_delta}")
    print(f"cost_usd={result.total_usage.estimated_cost_usd:.8f}")
    print(f"output={args.output}")
    return 0 if result.decision == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
