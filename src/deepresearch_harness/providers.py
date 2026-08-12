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
    def complete(self, *, stage: str, prompt: str) -> Completion:
        """Return a completion for a named pipeline stage."""


class FakeProvider(LLMProvider):
    """Deterministic provider used by tests and the local smoke demo."""

    name = "fake"
    model = "deterministic-fake-v1"

    def complete(self, *, stage: str, prompt: str) -> Completion:
        started = time.perf_counter()
        if stage == "plan":
            text = json.dumps(
                {
                    "steps": [
                        {"id": "scope", "objective": "Identify evidence relevant to the question."},
                        {"id": "synthesize", "objective": "Write only evidence-backed conclusions."},
                    ],
                    "search_queries": ["phased rollout pilot rollback observability"],
                }
            )
        elif stage == "ledger":
            evidence = json.loads(prompt)["evidence"]
            text = json.dumps(
                {
                    "claims": [
                        {
                            "id": f"claim-{item['id']}",
                            "text": item["excerpt"],
                            "evidence_ids": [item["id"]],
                            "support": "direct",
                        }
                        for item in evidence
                    ]
                }
            )
        elif stage == "write":
            payload = json.loads(prompt)
            lines = [f"# Research report\n\n## Question\n{payload['question']}\n\n## Evidence-backed findings"]
            for claim in payload["claims"]:
                marker = payload["citations"][claim["id"]]
                lines.append(f"- {claim['text']} {marker}")
            lines.append("\n## References")
            for evidence in payload["evidence"]:
                lines.append(f"- [{evidence['id']}] {evidence['title']}: {evidence['url']}")
            text = "\n".join(lines) + "\n"
        else:
            raise ValueError(f"unsupported fake stage: {stage}")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return Completion(text=text, usage=Usage(input_tokens=len(prompt.split()), output_tokens=len(text.split())), latency_ms=latency_ms)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, *, model: str, base_url: str, api_key_env: str, timeout_seconds: int = 60, pricing: Pricing | None = None) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key in environment variable {api_key_env}")
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._pricing = pricing or Pricing()

    def complete(self, *, stage: str, prompt: str) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return the requested content only. Do not claim unsupported facts."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
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
        estimated_cost_usd = (
            input_tokens * self._pricing.input_per_million_usd + output_tokens * self._pricing.output_per_million_usd
        ) / 1_000_000
        return Completion(
            text=text,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=estimated_cost_usd),
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
        pricing=provider.pricing,
    )
