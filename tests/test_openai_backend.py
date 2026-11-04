import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import lmstudio_json, lmstudio_pipeline, lmstudio_settings
from minimax_h3_novel_pipeline.nodes import lmstudio_config
from tests.test_structured_json import SCHEMA, Stream, client_for


@pytest.fixture(autouse=True)
def reset_settings():
    lmstudio_settings.set_provider_settings(provider_settings())
    yield
    lmstudio_settings.set_provider_settings(provider_settings(openai_enabled=False))
    lmstudio_settings.set_openai_settings("", "gpt-4.1")
    lmstudio_settings.set_connection_settings(lmstudio_settings.DEFAULT_API_URL, "")


def test_openai_configuration_keeps_key_separate_and_uses_fixed_endpoint(monkeypatch):
    lmstudio_settings.set_connection_settings(lmstudio_settings.DEFAULT_API_URL, "local-key")
    lmstudio_settings.set_provider_settings(provider_settings())
    monkeypatch.setattr(lmstudio_config, "reserve_run", lambda: "run")
    config, status = lmstudio_config.LMStudioConfigurationNode().run("http://untrusted.example/v1", backend="OpenAI")
    assert config["api_url"] == lmstudio_settings.OPENAI_API_URL
    assert config["backend"] == "OpenAI"
    assert "openai-key" not in str(config) + status
    assert lmstudio_settings.get_api_key(lmstudio_settings.DEFAULT_API_URL) == "local-key"
    with pytest.raises(ValueError):
        lmstudio_settings.get_openai_api_key("https://attacker.example/v1")


def test_openai_client_and_retries_keep_structured_schema():
    lmstudio_settings.set_provider_settings(provider_settings())
    first, second = Stream(["{"]), Stream([json.dumps({"value": "ok"})])
    client, create = client_for(first, second)
    make_client = Mock(return_value=client)
    module = SimpleNamespace(make_client=make_client)
    config = {"backend": "OpenAI", "openai_model": "gpt-4.1"}
    result, model = lmstudio_pipeline.make_client_and_model(module, lmstudio_settings.OPENAI_API_URL, config)
    assert model == "gpt-4.1"
    transport = make_client.call_args.kwargs["http_client"]
    try:
        assert not transport.follow_redirects and not transport.trust_env
        assert lmstudio_json.chat_json(result, model, "system", "user", SCHEMA, 0.1, 1024) == {"value": "ok"}
        assert create.call_count == 2
        assert all(call.kwargs["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
                   for call in create.call_args_list)
        with pytest.raises(ValueError):
            lmstudio_pipeline.make_client_and_model(module, "https://attacker.example/v1", config)
    finally:
        transport.close()


def provider_settings(openai_enabled=True):
    return {
        "lmstudio": {"enabled": True, "api_url": lmstudio_settings.DEFAULT_API_URL, "api_key": "local-key"},
        "openai": {"enabled": openai_enabled, "api_url": lmstudio_settings.OPENAI_API_URL,
                   "api_key": "openai-key", "model": "gpt-4.1"},
    }