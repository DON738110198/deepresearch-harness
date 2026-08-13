import json
from hashlib import sha256
from pathlib import Path

import pytest

from deepresearch_harness.browsecomp_evaluation import (
    DevelopmentGoldRow,
    DevelopmentGoldSlice,
    DiagnosticRow,
    DiagnosticSummary,
    _prediction_set_sha256,
)
from deepresearch_harness.browsecomp_plus import normalized_text_file_sha256
from deepresearch_harness.browsecomp_judge import (
    aggregate_official_judge_results,
    prepare_official_judge_batch,
    validate_official_judge_batch,
)
from deepresearch_harness.browsecomp_repeats import aggregate_repeat_experiment
from deepresearch_harness.pi_browsecomp import (
    PiSmokeItem,
    PiSmokeSummary,
    export_pi_runs_for_official_evaluator,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "benchmarks" / "browsecomp_plus_v0" / "target_manifest.json"
RETRIEVERS = ROOT / "benchmarks" / "browsecomp_plus_v0" / "retriever_candidates.json"
EVALUATOR = ROOT / "benchmarks" / "browsecomp_plus_v0" / "official_evaluator.json"


def test_repeat_aggregation_requires_v6_and_reports_paired_outcomes(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    (tmp_path / "runs").mkdir()
    target_hash = normalized_text_file_sha256(TARGET)
    retriever_hash = normalized_text_file_sha256(RETRIEVERS)
    pairs = []
    recalls = [(0.0, 0.5), (0.25, 0.75), (0.5, 1.0)]
    for index, (baseline_first_recall, candidate_first_recall) in enumerate(
        recalls, start=1
    ):
        trial_id = f"trial-{index:02d}"
        (
            baseline_summary,
            baseline_gold,
            baseline_diagnostic,
            baseline_export,
        ) = _write_variant(
            tmp_path,
            trial_id=trial_id,
            variant="baseline",
            retriever_id="bm25",
            retriever_manifest_sha256=None,
            q1_exact=False,
            q1_recall=baseline_first_recall,
            search_calls=2,
            cost_usd=0.02,
            target_hash=target_hash,
        )
        (
            candidate_summary,
            candidate_gold,
            candidate_diagnostic,
            candidate_export,
        ) = _write_variant(
            tmp_path,
            trial_id=trial_id,
            variant="candidate",
            retriever_id="qwen3-embedding-0.6b",
            retriever_manifest_sha256=retriever_hash,
            q1_exact=True,
            q1_recall=candidate_first_recall,
            search_calls=1,
            cost_usd=0.01,
            target_hash=target_hash,
        )
        pairs.append(
            {
                "trial_id": trial_id,
                "execution_order": (
                    "baseline_first" if index % 2 else "candidate_first"
                ),
                "baseline": {
                    "summary_path": baseline_summary,
                    "gold_slice_path": baseline_gold,
                    "diagnostic_path": baseline_diagnostic,
                    "official_export_manifest_path": baseline_export,
                },
                "candidate": {
                    "summary_path": candidate_summary,
                    "gold_slice_path": candidate_gold,
                    "diagnostic_path": candidate_diagnostic,
                    "official_export_manifest_path": candidate_export,
                },
            }
        )

    manifest = {
        "schema_version": "browsecomp-plus-repeat-experiment-v0",
        "registered_at": "2026-08-13T00:00:00Z",
        "registration_status": "pre_generation",
        "target_manifest_sha256": target_hash,
        "development_queries_sha256": "a" * 64,
        "expected_adapter_version": "pi-browsecomp-v6",
        "model": "deepseek-v4-flash",
        "control_policy": "answer_reserve_nonthinking_v0",
        "baseline_retriever_id": "bm25",
        "candidate_retriever_id": "qwen3-embedding-0.6b",
        "candidate_retriever_manifest_sha256": retriever_hash,
        "minimum_trials": 3,
        "pairs": pairs,
    }
    manifest_path = tmp_path / "runs" / "repeat_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "runs" / "comparison.json"

    comparison = aggregate_repeat_experiment(
        manifest_path=manifest_path,
        target_manifest_path=TARGET,
        output_path=output_path,
    )

    assert comparison.trial_count == 3
    assert comparison.registration_status == "pre_generation"
    assert comparison.queries_per_trial == 2
    assert comparison.paired_query_observations == 6
    assert comparison.baseline.evidence_recall_percent.mean == 37.5
    assert comparison.baseline.evidence_recall_percent.sample_stddev == 12.5
    assert comparison.candidate.evidence_recall_percent.mean == 62.5
    assert comparison.candidate.strict_exact_percent.mean == 100.0
    exact = next(
        metric for metric in comparison.paired_metrics if metric.metric == "strict_exact"
    )
    evidence = next(
        metric
        for metric in comparison.paired_metrics
        if metric.metric == "evidence_recall"
    )
    assert (exact.candidate_wins, exact.baseline_wins, exact.ties) == (3, 0, 3)
    assert (evidence.candidate_wins, evidence.baseline_wins, evidence.ties) == (
        3,
        0,
        3,
    )
    assert comparison.official_accuracy_status == "planned_not_run"
    assert output_path.is_file()
    assert (
        aggregate_repeat_experiment(
            manifest_path=manifest_path,
            target_manifest_path=TARGET,
            output_path=output_path,
            validate_existing=True,
        )
        == comparison
    )

    batch_dir = tmp_path / "runs" / "judge-batch"
    batch = prepare_official_judge_batch(
        repeat_experiment_path=manifest_path,
        repeat_comparison_path=output_path,
        target_manifest_path=TARGET,
        official_evaluator_path=EVALUATOR,
        output_dir=batch_dir,
    )
    assert batch.input_count == 12
    assert batch.trial_count == 3
    assert batch.queries_per_trial == 2
    assert validate_official_judge_batch(batch_dir / "batch_manifest.json") == batch
    staged_path = batch_dir / batch.items[0].staged_input_path
    staged_bytes = staged_path.read_bytes()
    staged_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="staged input hash mismatch"):
        validate_official_judge_batch(batch_dir / "batch_manifest.json")
    staged_path.write_bytes(staged_bytes)

    registration_path, execution_result_path = _write_official_judge_execution(
        batch_dir
    )
    judge_comparison_path = tmp_path / "runs" / "official-comparison.json"
    judge_comparison = aggregate_official_judge_results(
        batch_manifest_path=batch_dir / "batch_manifest.json",
        execution_registration_path=registration_path,
        execution_result_path=execution_result_path,
        output_path=judge_comparison_path,
    )
    assert judge_comparison.status == "official_evaluator_development_slice"
    assert judge_comparison.evaluations == 12
    assert judge_comparison.baseline.trial_accuracy_percent.mean == 50.0
    assert judge_comparison.candidate.trial_accuracy_percent.mean == 100.0
    assert (
        judge_comparison.paired.candidate_wins,
        judge_comparison.paired.baseline_wins,
        judge_comparison.paired.ties,
    ) == (3, 0, 3)
    assert (
        aggregate_official_judge_results(
            batch_manifest_path=batch_dir / "batch_manifest.json",
            execution_registration_path=registration_path,
            execution_result_path=execution_result_path,
            output_path=judge_comparison_path,
            validate_existing=True,
        )
        == judge_comparison
    )

    bad_manifest = {**manifest, "pairs": manifest["pairs"][:2]}
    bad_path = tmp_path / "runs" / "bad_manifest.json"
    bad_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 3 items"):
        aggregate_repeat_experiment(
            manifest_path=bad_path,
            target_manifest_path=TARGET,
            output_path=tmp_path / "runs" / "bad_comparison.json",
        )

    unbound_queries = {**manifest}
    unbound_queries.pop("development_queries_sha256")
    unbound_path = tmp_path / "runs" / "unbound_queries_manifest.json"
    unbound_path.write_text(json.dumps(unbound_queries), encoding="utf-8")
    with pytest.raises(ValueError, match="bind the query artifact hash"):
        aggregate_repeat_experiment(
            manifest_path=unbound_path,
            target_manifest_path=TARGET,
            output_path=tmp_path / "runs" / "unbound_queries_comparison.json",
        )

    non_alternating = json.loads(json.dumps(manifest))
    non_alternating["pairs"][1]["execution_order"] = "baseline_first"
    non_alternating_path = tmp_path / "runs" / "non_alternating_manifest.json"
    non_alternating_path.write_text(json.dumps(non_alternating), encoding="utf-8")
    with pytest.raises(ValueError, match="execution order must alternate"):
        aggregate_repeat_experiment(
            manifest_path=non_alternating_path,
            target_manifest_path=TARGET,
            output_path=tmp_path / "runs" / "non_alternating_comparison.json",
        )

    tampered_path = tmp_path / pairs[0]["baseline"]["diagnostic_path"]
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["rows"][0]["evidence_recall"] = 1.0
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic"):
        aggregate_repeat_experiment(
            manifest_path=manifest_path,
            target_manifest_path=TARGET,
            output_path=output_path,
            validate_existing=True,
        )

    registration_bytes = registration_path.read_bytes()
    execution_result_bytes = execution_result_path.read_bytes()
    changed_registration = json.loads(registration_bytes)
    max_tokens_index = changed_registration["command"].index("--max_output_tokens")
    changed_registration["command"][max_tokens_index + 1] = "2048"
    registration_path.write_text(json.dumps(changed_registration), encoding="utf-8")
    changed_execution = json.loads(execution_result_bytes)
    changed_execution["registration_sha256"] = sha256(
        registration_path.read_bytes()
    ).hexdigest()
    execution_result_path.write_text(json.dumps(changed_execution), encoding="utf-8")
    with pytest.raises(ValueError, match="inference contract"):
        aggregate_official_judge_results(
            batch_manifest_path=batch_dir / "batch_manifest.json",
            execution_registration_path=registration_path,
            execution_result_path=execution_result_path,
            output_path=judge_comparison_path,
            validate_existing=True,
        )
    registration_path.write_bytes(registration_bytes)
    execution_result_path.write_bytes(execution_result_bytes)

    execution = json.loads(execution_result_path.read_text(encoding="utf-8"))
    first_result_path = execution_result_path.parent / execution["output_files"][0]["path"]
    first_result_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        aggregate_official_judge_results(
            batch_manifest_path=batch_dir / "batch_manifest.json",
            execution_registration_path=registration_path,
            execution_result_path=execution_result_path,
            output_path=judge_comparison_path,
            validate_existing=True,
        )


def _write_variant(
    root: Path,
    *,
    trial_id: str,
    variant: str,
    retriever_id: str,
    retriever_manifest_sha256: str | None,
    q1_exact: bool,
    q1_recall: float,
    search_calls: int,
    cost_usd: float,
    target_hash: str,
) -> tuple[str, str, str, str]:
    source_dir = root / "runs" / "repeats" / trial_id / variant
    source_dir.mkdir(parents=True)
    items = []
    diagnostic_rows = []
    gold_rows = []
    for query_id, exact, recall in (
        ("q-1", q1_exact, q1_recall),
        ("q-2", True, 0.5),
    ):
        answer = (
            f"Explanation: fixture [{query_id}].\n"
            f"Exact Answer: {'right' if exact else 'wrong'}\nConfidence: 80%"
        )
        run_dir = source_dir / query_id
        run_dir.mkdir()
        run_payload = _v6_run_payload(
            run_id=f"{trial_id}-{variant}-{query_id}",
            query_id=query_id,
            answer=answer,
            search_calls=search_calls,
            cost_usd=cost_usd,
            retrieved_docids=[
                f"{query_id}-relevant-{index}" for index in range(round(recall * 4))
            ],
        )
        run_path = run_dir / "run.json"
        run_path.write_text(json.dumps(run_payload), encoding="utf-8")
        prediction_hash = sha256(answer.encode()).hexdigest()
        items.append(
            PiSmokeItem(
                query_id=query_id,
                status="succeeded",
                control_policy="answer_reserve_nonthinking_v0",
                answer_schema_complete=True,
                run_path=str(run_path),
                run_sha256=sha256(run_path.read_bytes()).hexdigest(),
                prediction_sha256=prediction_hash,
                search_calls=search_calls,
                output_tokens=100,
                output_budget_overshoot_tokens=0,
                total_tokens=150,
                cost_usd=cost_usd,
                latency_ms=1000,
            )
        )
        diagnostic_rows.append(
            DiagnosticRow(
                query_id=query_id,
                status="succeeded",
                answer_schema_complete=True,
                exact_answer_extracted=True,
                normalized_exact_match=exact,
                evidence_recall=recall,
                gold_recall=recall,
                search_calls=search_calls,
                prediction_sha256=prediction_hash,
                reference_answer_sha256=sha256(b"right").hexdigest(),
            )
        )
        gold_rows.append(
            DevelopmentGoldRow(
                query_id=query_id,
                question=f"fixture question {query_id}",
                answer="right",
                gold_docids=[f"{query_id}-relevant-{index}" for index in range(4)],
                evidence_docids=[
                    f"{query_id}-relevant-{index}" for index in range(4)
                ],
            )
        )
    summary = PiSmokeSummary(
        created_at="2026-08-13T00:00:00Z",
        target_manifest_sha256=target_hash,
        development_queries_sha256="a" * 64,
        model="deepseek-v4-flash",
        control_policy="answer_reserve_nonthinking_v0",
        retriever_id=retriever_id,
        retriever_manifest_sha256=retriever_manifest_sha256,
        query_count=2,
        succeeded=2,
        budget_exhausted=0,
        failed=0,
        schema_complete=2,
        answer_compiler_invocations=0,
        total_search_calls=search_calls * 2,
        total_output_tokens=200,
        total_output_budget_overshoot_tokens=0,
        total_tokens=300,
        total_cost_usd=cost_usd * 2,
        total_latency_ms=2000,
        items=items,
    )
    summary_path = source_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    gold = DevelopmentGoldSlice(
        created_at="2026-08-13T00:00:30Z",
        target_manifest_sha256=target_hash,
        source_summary_sha256=sha256(summary_path.read_bytes()).hexdigest(),
        prediction_set_sha256=_prediction_set_sha256(summary),
        accessed_fields=(
            "query_id",
            "query",
            "answer",
            "gold_docs.docid",
            "evidence_docs.docid",
        ),
        excluded_fields=(
            "gold_docs.text",
            "gold_docs.url",
            "evidence_docs.text",
            "evidence_docs.url",
            "negative_docs",
        ),
        query_count=2,
        rows=gold_rows,
    )
    gold_path = source_dir / "gold.json"
    gold_path.write_text(gold.model_dump_json(indent=2), encoding="utf-8")
    diagnostic = DiagnosticSummary(
        created_at="2026-08-13T00:01:00Z",
        source_summary_sha256=sha256(summary_path.read_bytes()).hexdigest(),
        gold_slice_sha256=sha256(gold_path.read_bytes()).hexdigest(),
        query_count=2,
        schema_complete=2,
        exact_answer_extracted=2,
        normalized_exact_match=sum(row.normalized_exact_match for row in diagnostic_rows),
        normalized_exact_match_percent=(
            sum(row.normalized_exact_match for row in diagnostic_rows) * 50.0
        ),
        evidence_recall_percent=sum(row.evidence_recall for row in diagnostic_rows)
        / 2
        * 100,
        gold_recall_percent=sum(row.gold_recall for row in diagnostic_rows) / 2 * 100,
        rows=diagnostic_rows,
    )
    diagnostic_path = source_dir / "diagnostic.json"
    diagnostic_path.write_text(diagnostic.model_dump_json(indent=2), encoding="utf-8")
    export_dir = source_dir.parent / f"{variant}.official_input"
    export_pi_runs_for_official_evaluator(
        source_dir=source_dir,
        output_dir=export_dir,
    )
    return (
        summary_path.relative_to(root).as_posix(),
        gold_path.relative_to(root).as_posix(),
        diagnostic_path.relative_to(root).as_posix(),
        (export_dir / "export_manifest.json").relative_to(root).as_posix(),
    )


def _v6_run_payload(
    *,
    run_id: str,
    query_id: str,
    answer: str,
    search_calls: int,
    cost_usd: float,
    retrieved_docids: list[str],
) -> dict:
    return {
        "schema_version": "pi-browsecomp-run-v0",
        "adapter_version": "pi-browsecomp-v6",
        "pi_version": "0.84.1",
        "run_id": run_id,
        "query_id": query_id,
        "model": "deepseek-v4-flash",
        "thinking_level": "high",
        "control_policy": "answer_reserve_nonthinking_v0",
        "compilation_thinking_level": "high",
        "system_prompt": "",
        "prompt_sha256": sha256(f"prompt-{query_id}".encode()).hexdigest(),
        "started_at": "2026-08-13T00:00:00Z",
        "latency_ms": 1000,
        "status": "succeeded",
        "stop_reason": "completed",
        "answer_text": answer,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 100,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 80,
            "total_tokens": 150,
            "cost_usd": cost_usd,
        },
        "bootstrap_output_tokens": 0,
        "exploration_output_tokens": 100,
        "compilation_output_tokens": 0,
        "answer_compiler_invoked": False,
        "first_tool_deadline_triggered": False,
        "answer_schema_complete": True,
        "output_budget_overshoot_tokens": 0,
        "model_requests": 1,
        "provider_request_limits": [
            {
                "request_index": 1,
                "phase": "exploration",
                "output_limit_field": "max_tokens",
                "thinking_type": "enabled",
                "temperature": None,
                "remaining_output_tokens": 8000,
                "global_remaining_output_tokens": 10000,
                "phase_remaining_output_tokens": 8000,
                "policy_cap_output_tokens": None,
                "requested_output_tokens": 10000,
                "applied_output_tokens": 8000,
            }
        ],
        "search_calls": [
            {
                "query": f"fixture query {index}",
                "outcome": "ok",
                "latency_ms": 1,
                "results": [
                    {"docid": docid, "score": 1.0, "snippet": "fixture"}
                    for docid in (retrieved_docids if index == 0 else [])
                ],
            }
            for index in range(search_calls)
        ],
        "messages": [],
    }


def _write_official_judge_execution(batch_dir: Path) -> tuple[Path, Path]:
    batch_manifest_path = batch_dir / "batch_manifest.json"
    batch_bytes = batch_manifest_path.read_bytes()
    batch = json.loads(batch_bytes)
    execution_dir = batch_dir.parent / "judge-execution"
    eval_dir = execution_dir / "evals" / "batch" / "inputs"
    eval_dir.mkdir(parents=True)
    registration = {
        "schema_version": "browsecomp-plus-official-judge-execution-v0",
        "registered_at": "2026-08-14T00:00:00Z",
        "status": "registered_pre_inference",
        "batch_manifest_sha256": sha256(batch_bytes).hexdigest(),
        "official_evaluator_manifest_sha256": batch[
            "official_evaluator_manifest_sha256"
        ],
        "launcher_path": "audit/run_browsecomp_plus_official_judge.py",
        "launcher_sha256": "e" * 64,
        "asset_verification_path": "audit/judge_asset_verification.json",
        "asset_verification_sha256": "f" * 64,
        "judge_assets_manifest_sha256": batch["judge_assets_manifest_sha256"],
        "repository_commit": batch["repository_commit"],
        "repository_clean": True,
        "evaluator_script_sha256": batch["evaluator_script_sha256"],
        "uv_lock_sha256": batch["uv_lock_sha256"],
        "judge_model": batch["judge_model"],
        "judge_revision": batch["judge_revision"],
        "model_dir": "/fixture/Qwen3-32B",
        "runtime": {
            "python": "3.12.0",
            "torch": "2.7.0+cu126",
            "transformers": "4.53.2",
            "vllm": "0.9.0.1",
            "cuda_available": True,
            "cuda_device_count": 2,
        },
        "gpu_ids": [0, 1],
        "gpu_checks": [
            {
                "captured_at": f"2026-08-14T00:00:0{index}Z",
                "devices": [
                    {
                        "index": gpu_id,
                        "uuid": f"GPU-{gpu_id}",
                        "memory_used_mib": 0,
                        "memory_total_mib": 49140,
                        "utilization_percent": 0,
                        "compute_pids": [],
                    }
                    for gpu_id in (0, 1)
                ],
            }
            for index in range(3)
        ],
        "inference": batch["inference"],
        "batch_size": batch["input_count"],
        "tensor_parallel_size": 2,
        "command": [
            "python",
            "/fixture/official/scripts_evaluation/evaluate_run.py",
            "--input_dir",
            "/fixture/batch/inputs",
            "--ground_truth",
            "/fixture/batch/ground_truth/development_ground_truth.jsonl",
            "--eval_dir",
            "/fixture/execution/evals",
            "--model",
            "/fixture/Qwen3-32B",
            "--temperature",
            "0.7",
            "--top_p",
            "0.8",
            "--top_k",
            "20",
            "--max_output_tokens",
            "4096",
            "--batch_size",
            str(batch["input_count"]),
            "--tensor_parallel_size",
            "2",
            "--qrel_evidence",
            "/fixture/official/topics-qrels/qrel_evidence.txt",
        ],
    }
    registration_path = execution_dir / "execution_registration.json"
    verification_path = execution_dir / registration["asset_verification_path"]
    verification_path.parent.mkdir(parents=True)
    launcher_path = execution_dir / registration["launcher_path"]
    launcher_path.write_text("fixture launcher\n", encoding="utf-8")
    registration["launcher_sha256"] = sha256(launcher_path.read_bytes()).hexdigest()
    verification_payload = {
        "passed": True,
        "matched": 24,
        "file_count": 24,
        "model": batch["judge_model"],
        "revision": batch["judge_revision"],
        "manifest_sha256": batch["judge_assets_manifest_sha256"],
    }
    verification_path.write_text(json.dumps(verification_payload), encoding="utf-8")
    registration["asset_verification_sha256"] = sha256(
        verification_path.read_bytes()
    ).hexdigest()
    registration_path.write_text(json.dumps(registration), encoding="utf-8")

    output_files = []
    for item in batch["items"]:
        source = json.loads(
            (batch_dir / item["staged_input_path"]).read_text(encoding="utf-8")
        )
        response = source["result"][-1]["output"]
        correct = item["variant"] == "candidate" or item["query_id"] == "q-2"
        payload = {
            "query_id": item["query_id"],
            "question": f"fixture question {item['query_id']}",
            "response": response,
            "correct_answer": "right",
            "is_completed": True,
            "judge_response": (
                "extracted_final_answer: fixture\nreasoning: fixture\n"
                f"correct: {'yes' if correct else 'no'}\nconfidence: 80"
            ),
            "judge_result": {
                "extracted_final_answer": "fixture",
                "reasoning": "fixture",
                "correct": correct,
                "confidence": 80.0,
                "parse_error": False,
            },
            "model_info": {
                "judge_model": registration["model_dir"],
                "max_output_tokens": 4096,
            },
        }
        result_path = eval_dir / f"{item['batch_id']}_eval.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        output_files.append(
            {
                "path": result_path.relative_to(execution_dir).as_posix(),
                "sha256": sha256(result_path.read_bytes()).hexdigest(),
            }
        )
    stdout_path = execution_dir / "evaluator.stdout.log"
    stderr_path = execution_dir / "evaluator.stderr.log"
    stdout_path.write_text("fixture success\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    execution_result = {
        "schema_version": "browsecomp-plus-official-judge-execution-result-v0",
        "started_at": "2026-08-14T00:01:00Z",
        "completed_at": "2026-08-14T00:02:00Z",
        "status": "succeeded",
        "exit_code": 0,
        "registration_sha256": sha256(registration_path.read_bytes()).hexdigest(),
        "batch_manifest_sha256": sha256(batch_bytes).hexdigest(),
        "stdout": {
            "path": stdout_path.relative_to(execution_dir).as_posix(),
            "sha256": sha256(stdout_path.read_bytes()).hexdigest(),
        },
        "stderr": {
            "path": stderr_path.relative_to(execution_dir).as_posix(),
            "sha256": sha256(stderr_path.read_bytes()).hexdigest(),
        },
        "output_files": output_files,
    }
    execution_result_path = execution_dir / "execution_result.json"
    execution_result_path.write_text(json.dumps(execution_result), encoding="utf-8")
    return registration_path, execution_result_path
