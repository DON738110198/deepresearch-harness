from pathlib import Path

import pytest

from scripts import run_browsecomp_plus_official_judge as launcher


def test_runtime_versions_extracts_marked_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launcher,
        "run_text",
        lambda command: (
            "INFO import noise\n"
            "__DEEPRESEARCH_RUNTIME__"
            '{"python":"3.10","torch":"2.7.0+cu126",'
            '"transformers":"4.53.2","vllm":"0.9.0.1",'
            '"cuda_available":true,"cuda_device_count":8}\n'
        ),
    )

    runtime = launcher.runtime_versions(Path("/fixture/python"))

    assert runtime["vllm"] == "0.9.0.1"
    assert runtime["cuda_device_count"] == 8


def test_gpu_check_rejects_memory_or_compute_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_output = (
        "0, GPU-zero, 0, 49140, 0\n"
        "1, GPU-one, 0, 49140, 0\n"
    )

    def free_run(command: list[str], *, cwd=None) -> str:
        gpu_query = (
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu"
        )
        return gpu_output if gpu_query in command else ""

    monkeypatch.setattr(launcher, "run_text", free_run)
    snapshot = launcher.gpu_check((0, 1))
    assert [row["index"] for row in snapshot["devices"]] == [0, 1]

    def occupied_run(command: list[str], *, cwd=None) -> str:
        if "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu" in command:
            return gpu_output
        return "GPU-one, 12345\n"

    monkeypatch.setattr(launcher, "run_text", occupied_run)
    with pytest.raises(ValueError, match="selected GPU is occupied: 1"):
        launcher.gpu_check((0, 1))


def test_build_command_pins_official_inference_contract(tmp_path: Path) -> None:
    repository = tmp_path / "BrowseComp-Plus"
    batch_root = tmp_path / "batch"
    ground_truth = batch_root / "ground_truth.jsonl"
    model_dir = tmp_path / "Qwen3-32B"
    eval_dir = tmp_path / "execution" / "evals"
    batch = {
        "evaluator_script": "scripts_evaluation/evaluate_run.py",
        "input_count": 30,
        "inference": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 4096,
            "enable_thinking": False,
        },
    }

    command = launcher.build_command(
        python=Path("/fixture/venv/python"),
        repository=repository,
        batch=batch,
        batch_root=batch_root,
        ground_truth=ground_truth,
        model_dir=model_dir,
        eval_dir=eval_dir,
    )

    assert command[command.index("--batch_size") + 1] == "30"
    assert command[command.index("--tensor_parallel_size") + 1] == "2"
    assert command[command.index("--temperature") + 1] == "0.7"
    assert command[command.index("--max_output_tokens") + 1] == "4096"


def test_execution_contract_allows_only_audited_nccl_transport_override() -> None:
    from deepresearch_harness.browsecomp_judge import (
        OfficialJudgeExecutionRegistration,
    )

    payload = {
        "registered_at": "2026-08-14T00:00:00+00:00",
        "batch_manifest_sha256": "a" * 64,
        "official_evaluator_manifest_sha256": "b" * 64,
        "launcher_path": "audit/launcher.py",
        "launcher_sha256": "c" * 64,
        "asset_verification_path": "audit/assets.json",
        "asset_verification_sha256": "d" * 64,
        "judge_assets_manifest_sha256": "e" * 64,
        "repository_commit": "f" * 40,
        "repository_clean": True,
        "evaluator_script_sha256": "1" * 64,
        "uv_lock_sha256": "2" * 64,
        "judge_model": "Qwen/Qwen3-32B",
        "judge_revision": "3" * 40,
        "model_dir": "/models/Qwen3-32B",
        "runtime": {
            "python": "3.10",
            "torch": "2.7.0+cu126",
            "transformers": "4.53.2",
            "vllm": "0.9.0.1",
            "cuda_available": True,
            "cuda_device_count": 8,
        },
        "gpu_ids": [4, 5],
        "gpu_checks": [
            {
                "captured_at": f"2026-08-14T00:00:0{index}+00:00",
                "devices": [
                    {
                        "index": gpu_id,
                        "uuid": f"GPU-{gpu_id}",
                        "memory_used_mib": 0,
                        "memory_total_mib": 49140,
                        "utilization_percent": 0,
                        "compute_pids": [],
                    }
                    for gpu_id in (4, 5)
                ],
            }
            for index in range(3)
        ],
        "inference": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 4096,
            "enable_thinking": False,
        },
        "batch_size": 30,
        "tensor_parallel_size": 2,
        "environment_overrides": {"NCCL_P2P_DISABLE": "1"},
        "command": ["python", "evaluate_run.py"],
    }

    registration = OfficialJudgeExecutionRegistration.model_validate(payload)
    assert registration.environment_overrides == {"NCCL_P2P_DISABLE": "1"}

    payload["environment_overrides"] = {"UNTRACKED_RUNTIME_CHANGE": "1"}
    with pytest.raises(ValueError, match="unsupported environment override"):
        OfficialJudgeExecutionRegistration.model_validate(payload)
