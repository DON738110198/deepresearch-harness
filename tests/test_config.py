import json

import pytest

from deepresearch_harness.contracts import HarnessConfig, Pricing
from deepresearch_harness.providers import OpenAICompatibleProvider, provider_from_config


def test_openai_provider_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_TEST_KEY", raising=False)
    config = HarnessConfig.model_validate(
        {
            "provider": {
                "kind": "openai_compatible",
                "model": "test-model",
                "base_url": "https://example.test/v1",
                "api_key_env": "HARNESS_TEST_KEY",
            }
        }
    )
    with pytest.raises(RuntimeError, match="HARNESS_TEST_KEY"):
        provider_from_config(config)


def test_cost_budget_requires_pricing() -> None:
    with pytest.raises(ValueError, match="cost budget requires"):
        HarnessConfig.model_validate(
            {
                "provider": {
                    "kind": "openai_compatible",
                    "model": "test-model",
                    "base_url": "https://example.test/v1",
                    "api_key_env": "HARNESS_TEST_KEY",
                },
                "run": {"budget": {"max_estimated_cost_usd": 0.01}},
            }
        )


def test_provider_records_cache_aware_cost_and_structured_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_TEST_KEY", "test-only-key")
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_cache_hit_tokens":40,"prompt_cache_miss_tokens":60}}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("deepresearch_harness.providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url="https://example.test/v1",
        api_key_env="HARNESS_TEST_KEY",
        thinking_mode="disabled",
        pricing=Pricing(
            input_cache_hit_per_million_usd=0.0028,
            input_cache_miss_per_million_usd=0.14,
            output_per_million_usd=0.28,
        ),
    )

    completion = provider.complete(stage="plan", prompt="return json", json_output=True, max_output_tokens=128)

    payload = captured["payload"]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 128
    assert completion.usage.input_cache_hit_tokens == 40
    assert completion.usage.input_cache_miss_tokens == 60
    assert completion.usage.estimated_cost_usd == pytest.approx(0.000022512)
