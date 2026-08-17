from __future__ import annotations

import json
from pathlib import Path

from deepresearch_harness.provider_balance import check_deepseek_balance


def test_balance_audit_fails_closed_without_persisting_key(tmp_path: Path) -> None:
    key = "test-secret-never-persist"

    def unavailable(_request: object, _timeout: float) -> tuple[int, bytes]:
        return 200, json.dumps(
            {
                "is_available": False,
                "balance_infos": [
                    {
                        "currency": "CNY",
                        "total_balance": "-0.20",
                        "granted_balance": "0.00",
                        "topped_up_balance": "-0.20",
                    }
                ],
            }
        ).encode()

    output = tmp_path / "balance.json"
    audit = check_deepseek_balance(
        api_key=key,
        output_path=output,
        transport=unavailable,
    )

    assert audit.resume_allowed is False
    assert audit.model_inference_calls == 0
    assert key not in output.read_text(encoding="utf-8")


def test_balance_audit_allows_positive_available_balance(tmp_path: Path) -> None:
    def available(_request: object, _timeout: float) -> tuple[int, bytes]:
        return 200, json.dumps(
            {
                "is_available": True,
                "balance_infos": [
                    {
                        "currency": "CNY",
                        "total_balance": "10.00",
                        "granted_balance": "0.00",
                        "topped_up_balance": "10.00",
                    }
                ],
            }
        ).encode()

    audit = check_deepseek_balance(
        api_key="test-key",
        output_path=tmp_path / "balance.json",
        transport=available,
    )

    assert audit.resume_allowed is True
