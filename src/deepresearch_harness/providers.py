from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import HarnessConfig, Pricing, Usage


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    latency_ms: int


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(
        self,
        *,
        stage: str,
        prompt: str,
        json_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> Completion:
        """Return a completion for a named pipeline stage."""


class FakeProvider(LLMProvider):
    """Deterministic provider used by tests and the local smoke demo."""

    name = "fake"
    model = "deterministic-fake-v1"

    def complete(
        self,
        *,
        stage: str,
        prompt: str,
        json_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> Completion:
        started = time.perf_counter()
        if stage == "plan":
            payload = json.loads(prompt)
            plan = {
                    "steps": [
                        {"id": "scope", "objective": "Identify evidence relevant to the question."},
                        {"id": "synthesize", "objective": "Write only evidence-backed conclusions."},
                    ],
            }
            if "obligations" in payload["json_example"]:
                plan["obligations"] = [
                    {
                        "id": "blast-radius",
                        "description": "Determine how a phased rollout limits initial impact.",
                        "search_query": "phased rollout blast radius evidence",
                    },
                    {
                        "id": "rollback",
                        "description": "Determine the rollback readiness requirement.",
                        "search_query": "phased rollout rollback readiness evidence",
                    },
                    {
                        "id": "observability",
                        "description": "Determine the monitoring gate for expansion.",
                        "search_query": "phased rollout observability expansion gate evidence",
                    },
                ]
            else:
                plan["search_queries"] = ["phased rollout pilot rollback observability"]
            text = json.dumps(plan)
        elif stage == "ledger":
            payload = json.loads(prompt)
            evidence = payload["evidence"]
            claims = [
                {
                    "id": f"claim-{item['id']}",
                    "text": item["excerpt"],
                    "evidence_ids": [item["id"]],
                    "support": "direct",
                }
                for item in evidence
            ]
            ledger = {"claims": claims}
            if "obligations" in payload:
                debts = []
                for index, obligation in enumerate(payload["obligations"]):
                    assigned = claims[index :: len(payload["obligations"])]
                    debts.append(
                        {
                            "obligation_id": obligation["id"],
                            "status": "resolved" if assigned else "open",
                            "evidence_ids": [item["evidence_ids"][0] for item in assigned],
                            "claim_ids": [item["id"] for item in assigned],
                            "detail": (
                                "Deterministic fixture evidence assigned."
                                if assigned
                                else "No deterministic fixture evidence was available."
                            ),
                        }
                    )
                ledger["evidence_debts"] = debts
            text = json.dumps(ledger)
        elif stage == "benchmark_write":
            text = json.dumps({"answer": []})
        elif stage == "write":
            payload = json.loads(prompt)
            lines = [f"# Research report\n\n## Question\n{payload['question']}\n\n## Evidence-backed findings"]
            for claim in payload["claims"]:
                marker = payload["citations"][claim["id"]]
                lines.append(f"- {claim['text']} {marker}")
            lines.append("\n## References")
            for evidence in payload.get("evidence", []):
                lines.append(f"- [{evidence['id']}] {evidence['title']}: {evidence['url']}")
            text = "\n".join(lines) + "\n"
        elif stage == "direct_write":
            payload = json.loads(prompt)
            evidence = payload["evidence"]
            claims = [
                {
                    "id": f"claim-{item['id']}",
                    "text": item["excerpt"],
                    "evidence_ids": [item["id"]],
                    "support": "direct",
                }
                for item in evidence
            ]
            lines = [f"# Research report\n\n## Question\n{payload['question']}\n\n## Evidence-backed findings"]
            for claim in claims:
                lines.append(f"- {claim['text']} [[{claim['id']}]]")
            lines.append("\n## References")
            for item in evidence:
                lines.append(f"- [{item['id']}] {item['title']}: {item['url']}")
            text = json.dumps({"claims": claims, "report": "\n".join(lines) + "\n"})
        elif stage == "review_translate":
            payload = json.loads(prompt)
            text = json.dumps(
                {
                    "translations": [
                        {
                            "id": item["id"],
                            "text": f"中文辅助翻译：{item['source']}",
                        }
                        for item in payload["items"]
                    ]
                },
                ensure_ascii=False,
            )
        else:
            raise ValueError(f"unsupported fake stage: {stage}")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return Completion(text=text, usage=Usage(input_tokens=len(prompt.split()), output_tokens=len(text.split())), latency_ms=latency_ms)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: int = 60,
        thinking_mode: str | None = None,
        pricing: Pricing | None = None,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key in environment variable {api_key_env}")
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._thinking_mode = thinking_mode
        self._pricing = pricing or Pricing()

    def complete(
        self,
        *,
        stage: str,
        prompt: str,
        json_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return the requested content only. Do not claim unsupported facts."},
                {"role": "user", "content": prompt},
            ],
        }
        if self._thinking_mode is None:
            payload["temperature"] = 0
        else:
            payload["thinking"] = {"type": self._thinking_mode}
            if self._thinking_mode == "disabled":
                payload["temperature"] = 0
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        request = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"OpenAI-compatible request failed during {stage}: {error}") from error
        usage = body.get("usage", {})
        text = body["choices"][0]["message"]["content"]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cache_hit_tokens = usage.get("prompt_cache_hit_tokens", 0)
        cache_miss_tokens = usage.get("prompt_cache_miss_tokens", 0)
        unclassified_input = max(input_tokens - cache_hit_tokens - cache_miss_tokens, 0)
        default_input_price = self._pricing.input_per_million_usd
        hit_price = self._pricing.input_cache_hit_per_million_usd or default_input_price
        miss_price = self._pricing.input_cache_miss_per_million_usd or default_input_price
        estimated_cost_usd = (
            cache_hit_tokens * hit_price
            + cache_miss_tokens * miss_price
            + unclassified_input * default_input_price
            + output_tokens * self._pricing.output_per_million_usd
        ) / 1_000_000
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cache_hit_tokens=cache_hit_tokens,
                input_cache_miss_tokens=cache_miss_tokens,
                estimated_cost_usd=estimated_cost_usd,
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def provider_from_config(config: HarnessConfig | None) -> LLMProvider:
    if config is None or config.provider.kind == "fake":
        return FakeProvider()
    provider = config.provider
    if not all([provider.model, provider.base_url, provider.api_key_env]):
        raise ValueError("openai_compatible provider requires model, base_url, and api_key_env")
    return OpenAICompatibleProvider(
        model=provider.model,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        timeout_seconds=provider.timeout_seconds,
        thinking_mode=provider.thinking_mode,
        pricing=provider.pricing,
    )
