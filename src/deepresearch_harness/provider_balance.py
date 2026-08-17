from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEEPSEEK_BALANCE_ENDPOINT = "https://api.deepseek.com/user/balance"


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BalanceInfo(StrictContract):
    currency: str = Field(min_length=1)
    total_balance: str = Field(min_length=1)
    granted_balance: str = Field(min_length=1)
    topped_up_balance: str = Field(min_length=1)

    @model_validator(mode="after")
    def amounts_are_decimal(self) -> "BalanceInfo":
        for value in (
            self.total_balance,
            self.granted_balance,
            self.topped_up_balance,
        ):
            try:
                Decimal(value)
            except InvalidOperation as error:
                raise ValueError("DeepSeek balance amount is not decimal") from error
        return self


class DeepSeekBalanceAudit(StrictContract):
    schema_version: Literal["deepseek-balance-audit-v0"] = "deepseek-balance-audit-v0"
    checked_at: str = Field(min_length=1)
    endpoint: Literal["https://api.deepseek.com/user/balance"] = (
        DEEPSEEK_BALANCE_ENDPOINT
    )
    key_env: Literal["DEEPSEEK_API_KEY"] = "DEEPSEEK_API_KEY"
    key_present: Literal[True] = True
    provider_reported_available: bool
    positive_total_balance: bool
    resume_allowed: bool
    balances: list[BalanceInfo]
    model_inference_calls: Literal[0] = 0

    @model_validator(mode="after")
    def resume_gate_matches_balance(self) -> "DeepSeekBalanceAudit":
        expected = self.provider_reported_available and self.positive_total_balance
        if self.resume_allowed != expected:
            raise ValueError("DeepSeek resume gate differs from balance response")
        return self


BalanceTransport = Callable[[Request, float], tuple[int, bytes]]


def check_deepseek_balance(
    *,
    api_key: str,
    output_path: Path,
    timeout_seconds: float = 20,
    transport: BalanceTransport | None = None,
) -> DeepSeekBalanceAudit:
    if not api_key.strip():
        raise ValueError("DEEPSEEK_API_KEY is missing")
    if output_path.exists():
        raise ValueError("DeepSeek balance audit output already exists")
    request = Request(
        DEEPSEEK_BALANCE_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    status, body = (transport or _urlopen_transport)(request, timeout_seconds)
    if status != 200:
        raise RuntimeError(f"DeepSeek balance endpoint returned HTTP {status}")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek balance response must be an object")
    balances = [
        BalanceInfo.model_validate(value)
        for value in payload.get("balance_infos", [])
    ]
    positive = any(Decimal(item.total_balance) > 0 for item in balances)
    available = payload.get("is_available") is True
    audit = DeepSeekBalanceAudit(
        checked_at=datetime.now(timezone.utc).isoformat(),
        provider_reported_available=available,
        positive_total_balance=positive,
        resume_allowed=available and positive,
        balances=balances,
    )
    _atomic_write(output_path, audit.model_dump(mode="json"))
    return audit


def _urlopen_transport(request: Request, timeout_seconds: float) -> tuple[int, bytes]:
    with urlopen(request, timeout=timeout_seconds) as response:
        return int(response.status), response.read()


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
