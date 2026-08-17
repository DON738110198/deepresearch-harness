from pathlib import Path

import pytest
from pydantic import ValidationError

import deepresearch_harness.livedrbench_fresh_public as fresh_public
from deepresearch_harness.livedrbench_fresh_public import (
    EXCLUDED_KEYS,
    KEYS,
    load_fresh_public_registration,
    selected_keys_sha256,
    validate_fresh_public_registration,
    validate_fresh_public_dataset,
)
from deepresearch_harness.public_benchmark import LiveDRBenchTask


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "benchmarks" / "livedrbench_fresh_public_v0" / "registration.json"


def test_fresh_registration_is_hash_bound_and_non_overlapping() -> None:
    registration = validate_fresh_public_registration(REGISTRATION)

    assert registration.selected_task_keys == KEYS
    assert registration.excluded_prior_task_keys == EXCLUDED_KEYS
    assert not set(registration.selected_task_keys) & set(EXCLUDED_KEYS)
    assert registration.candidate.status == "implemented_not_run"
    assert registration.candidate.adapter == "web_research.TavilySearchProvider"
    assert registration.candidate.api_key_env == "TAVILY_API_KEY"
    assert registration.budget.max_search_calls_per_task == 5
    assert registration.budget.max_output_tokens_per_call == 2048
    assert registration.budget.max_search_results_per_query == 5
    assert registration.budget.candidate_max_search_credits_per_task == 5
    assert registration.budget.candidate_max_search_cost_usd_per_task == 0.04


def test_selected_key_hash_uses_final_lf() -> None:
    assert selected_keys_sha256(KEYS) == (
        "a4de10cb92e529184acbe76c6d1a28101d69f283eb05627be364ff00b41b1470"
    )


def test_registration_rejects_changed_selection() -> None:
    payload = load_fresh_public_registration(REGISTRATION).model_dump(mode="json")
    payload["selected_task_keys"] = [10, 23, 38, 86, 4]

    with pytest.raises(ValidationError, match="selected task keys"):
        type(load_fresh_public_registration(REGISTRATION)).model_validate(payload)


def test_registration_rejects_parent_manifest_hash_change(tmp_path: Path) -> None:
    parent = tmp_path / "manifest.json"
    parent.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="parent preview manifest hash"):
        validate_fresh_public_registration(REGISTRATION, parent_manifest_path=parent)


def test_dataset_validation_is_explicit_and_uses_mocked_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registration = load_fresh_public_registration(REGISTRATION)
    categories = registration.category_by_key

    def fake_fetch(manifest: object) -> tuple[list[LiveDRBenchTask], str]:
        assert getattr(manifest, "task_keys") == list(registration.available_task_keys)
        return (
            [
                LiveDRBenchTask(
                    key=key,
                    category=categories[str(key)],
                    question=f"question {key}",
                    ground_truths=[],
                )
                for key in registration.available_task_keys
            ],
            registration.dataset_response_sha256,
        )

    monkeypatch.setattr(fresh_public, "fetch_livedrbench_tasks", fake_fetch)
    monkeypatch.chdir(tmp_path)
    tasks = validate_fresh_public_dataset(registration)

    assert tuple(task.key for task in tasks) == KEYS


def test_dataset_validation_rejects_response_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    registration = load_fresh_public_registration(REGISTRATION)

    def fake_fetch(manifest: object) -> tuple[list[LiveDRBenchTask], str]:
        return ([], "0" * 64)

    monkeypatch.setattr(fresh_public, "fetch_livedrbench_tasks", fake_fetch)
    with pytest.raises(ValueError, match="dataset response hash"):
        validate_fresh_public_dataset(registration)
