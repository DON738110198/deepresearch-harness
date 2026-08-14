from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.development_judge import (
    DevelopmentJudgeObservation,
    DevelopmentJudgeResult,
)
from deepresearch_harness.repeat_development_judge import (
    aggregate_repeat_development_judge,
)
from deepresearch_harness.screening_judge import load_screening_manifest


ROOT = Path(__file__).resolve().parents[1]
JUDGE_MANIFEST = (
    ROOT
    / "benchmarks"
    / "browsecomp_plus_v0"
    / "persistent_bf16_judge_v0.json"
)


def test_repeat_development_judge_aggregates_hash_bound_paired_grid(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    runs = repository / "runs"
    runs.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    target_path = repository / "target.json"
    target_path.write_text("{}", encoding="utf-8")
    repeat_path = runs / "repeat.json"
    comparison_path = runs / "comparison.json"
    comparison_path.write_text("{}", encoding="utf-8")
    results_root = runs / "judge" / "trials"
    calibration_path = runs / "calibration.json"

    pairs = []
    for trial_index in range(1, 4):
        trial_id = f"trial-{trial_index:02d}"
        pair = {
            "trial_id": trial_id,
            "execution_order": (
                "candidate_first" if trial_index == 2 else "baseline_first"
            ),
        }
        for variant in ("baseline", "candidate"):
            root = runs / f"{trial_id}-{variant}"
            root.mkdir()
            summary = root / "summary.json"
            gold = runs / f"{trial_id}-{variant}.gold.json"
            diagnostic = runs / f"{trial_id}-{variant}.diagnostic.json"
            export = runs / f"{trial_id}-{variant}.export.json"
            for path, value in (
                (summary, f"summary-{trial_id}-{variant}"),
                (gold, f"gold-{trial_id}-{variant}"),
                (diagnostic, "diagnostic"),
                (export, "export"),
            ):
                path.write_text(value, encoding="utf-8")
            pair[variant] = {
                "summary_path": summary.relative_to(repository).as_posix(),
                "gold_slice_path": gold.relative_to(repository).as_posix(),
                "diagnostic_path": diagnostic.relative_to(repository).as_posix(),
                "official_export_manifest_path": export.relative_to(
                    repository
                ).as_posix(),
            }
        pairs.append(pair)

    experiment = {
        "schema_version": "browsecomp-plus-repeat-experiment-v1",
        "registered_at": "2026-08-14T00:00:00+00:00",
        "registration_status": "pre_generation",
        "target_manifest_sha256": "1" * 64,
        "development_queries_sha256": "2" * 64,
        "expected_adapter_version": "pi-browsecomp-v6",
        "model": "deepseek-v4-flash",
        "control_policy": "answer_reserve_nonthinking_v0",
        "baseline_retriever_id": "bm25",
        "candidate_retriever_id": "dense",
        "candidate_retriever_manifest_sha256": "3" * 64,
        "provider_failure_retry_policy": {
            "schema_version": "provider-failure-retry-v0",
            "eligible_statuses": ["failed"],
            "immutable_statuses": ["succeeded", "budget_exhausted"],
            "max_resume_invocations": 3,
            "preserve_attempt_artifacts": True,
            "cumulative_usage_accounting": True,
            "generation_controls": "identical_to_registered_variant",
        },
        "minimum_trials": 3,
        "pairs": pairs,
    }
    repeat_path.write_text(json.dumps(experiment), encoding="utf-8")
    manifest = load_screening_manifest(JUDGE_MANIFEST)
    manifest_hash = normalized_text_file_sha256(JUDGE_MANIFEST)
    calibration_path.write_text(
        json.dumps(
            {
                "status": "accepted_for_development_screening",
                "screening_manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    calibration_hash = sha256(calibration_path.read_bytes()).hexdigest()

    labels = {
        ("trial-01", "baseline"): [False, False],
        ("trial-01", "candidate"): [True, False],
        ("trial-02", "baseline"): [True, False],
        ("trial-02", "candidate"): [False, False],
        ("trial-03", "baseline"): [True, True],
        ("trial-03", "candidate"): [True, True],
    }
    for (trial_id, variant), values in labels.items():
        source_root = runs / f"{trial_id}-{variant}"
        result_root = results_root / f"{trial_id}-{variant}"
        item_root = result_root / "results"
        item_root.mkdir(parents=True)
        rows = []
        for index, correct in enumerate(values, start=1):
            item_path = item_root / f"q{index}_eval.json"
            item_path.write_text(f"{trial_id}-{variant}-q{index}", encoding="utf-8")
            rows.append(
                DevelopmentJudgeObservation(
                    query_id=f"q{index}",
                    prediction_sha256=str(index) * 64,
                    correct=correct,
                    confidence=90,
                    parse_error=False,
                    latency_ms=10,
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                    result_path=f"results/q{index}_eval.json",
                    result_sha256=sha256(item_path.read_bytes()).hexdigest(),
                )
            )
        gold_path = runs / f"{trial_id}-{variant}.gold.json"
        result = DevelopmentJudgeResult(
            created_at="2026-08-14T00:00:00+00:00",
            status="succeeded",
            judge_manifest_sha256=manifest_hash,
            calibration_sha256=calibration_hash,
            source_summary_sha256=sha256(
                (source_root / "summary.json").read_bytes()
            ).hexdigest(),
            gold_slice_sha256=sha256(gold_path.read_bytes()).hexdigest(),
            judge_model=manifest.judge.model,
            judge_revision=manifest.judge.revision,
            served_model_name=manifest.engine.served_model_name,
            inference=manifest.inference,
            query_count=2,
            evaluations=2,
            correct=sum(values),
            accuracy_percent=round(sum(values) / 2 * 100, 6),
            parse_failures=0,
            request_failures=0,
            request_errors=[],
            elapsed_ms=20,
            prompt_tokens=200,
            completion_tokens=20,
            total_tokens=220,
            observations=rows,
            claim_boundary="not official",
        )
        (result_root / "development_judge_result.json").write_text(
            result.model_dump_json(), encoding="utf-8"
        )

    monkeypatch.setattr(
        "deepresearch_harness.repeat_development_judge.aggregate_repeat_experiment",
        lambda **_kwargs: SimpleNamespace(
            experiment_manifest_sha256=normalized_text_file_sha256(repeat_path),
            trial_count=3,
            queries_per_trial=2,
        ),
    )
    output_path = runs / "judge" / "comparison.json"
    result = aggregate_repeat_development_judge(
        repeat_experiment_path=repeat_path,
        repeat_comparison_path=comparison_path,
        target_manifest_path=target_path,
        judge_manifest_path=JUDGE_MANIFEST,
        calibration_result_path=calibration_path,
        results_root=results_root,
        output_path=output_path,
    )

    assert result.evaluations == 12
    assert result.baseline.pooled_accuracy_percent == 50.0
    assert result.candidate.pooled_accuracy_percent == 50.0
    assert result.paired.candidate_wins == 1
    assert result.paired.baseline_wins == 1
    assert result.paired.ties == 4
    assert result.total_tokens == 1320
    assert output_path.is_file()
