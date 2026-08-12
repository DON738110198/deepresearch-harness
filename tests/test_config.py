import pytest

from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.providers import provider_from_config


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
