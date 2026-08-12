from pathlib import Path

from deepresearch_harness.experiment import BudgetMode, validate_experiment_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_token_matched_manifest_pins_suite_and_corpus() -> None:
    manifest = validate_experiment_manifest(ROOT / "experiments" / "pilot_v0" / "token_matched.json")

    assert manifest.budget_mode is BudgetMode.TOKEN_MATCHED
    assert manifest.budget.max_total_tokens == 8000
    assert manifest.provider.model == "deepseek-v4-flash"
    assert manifest.implementation_revision == "2781fc9389d450c465a4ea19d76e5f21c4290406"


def test_cost_matched_manifest_has_frozen_pricing() -> None:
    manifest = validate_experiment_manifest(ROOT / "experiments" / "pilot_v0" / "cost_matched.json")

    assert manifest.budget_mode is BudgetMode.COST_MATCHED
    assert manifest.budget.max_estimated_cost_usd == 0.002
    assert manifest.provider.pricing.input_cache_miss_per_million_usd == 0.14
