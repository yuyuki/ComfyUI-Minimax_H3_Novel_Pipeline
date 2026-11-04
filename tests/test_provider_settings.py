"""Offline settings validation and connection-test behavior."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone

import httpx
import pytest

from minimax_h3_novel_pipeline.core import lmstudio_settings as settings
from tests.test_openai_backend import provider_settings


@pytest.fixture(autouse=True)
def restore():
    settings.set_provider_settings(provider_settings())
    yield
    settings.set_provider_settings(provider_settings(openai_enabled=False))
    settings.set_connection_settings(settings.DEFAULT_API_URL, "")


def test_disabled_providers_cannot_supply_credentials():
    data = provider_settings(openai_enabled=False)
    data["lmstudio"]["enabled"] = False
    settings.set_provider_settings(data)
    with pytest.raises(ValueError, match="disabled"):
        settings.get_api_key(settings.DEFAULT_API_URL)
    with pytest.raises(ValueError, match="disabled"):
        settings.get_openai_api_key(settings.OPENAI_API_URL)


def test_node_inputs_only_expose_provider_and_qwen_controls():
    from minimax_h3_novel_pipeline.nodes.lmstudio_config import LMStudioConfigurationNode

    inputs = LMStudioConfigurationNode.INPUT_TYPES()["required"]
    assert inputs["backend"][0] == ["LM Studio", "OpenAI"]
    assert "api_url" not in inputs and "model_family" not in inputs


def test_options_only_include_enabled_complete_saved_providers():
    data = provider_settings()
    data["lmstudio"]["model"] = "qwen3.5-test"
    settings.set_provider_settings(data)
    assert settings.get_provider_options() == [
        {"backend": "LM Studio", "model": "qwen3.5-test", "model_family": "Qwen"},
        {"backend": "OpenAI", "model": "gpt-4.1", "model_family": "Mistral"},
    ]
    data["lmstudio"]["enabled"] = False
    settings.set_provider_settings(data)
    assert [p["backend"] for p in settings.get_provider_options()] == ["OpenAI"]
    data["openai"]["model"] = ""
    settings.set_provider_settings(data)
    assert settings.get_provider_options() == []
    data["openai"].update(model="gpt-4.1", api_key="")
    settings.set_provider_settings(data)
    assert settings.get_provider_options() == []


@pytest.mark.parametrize("model,family", [
    ("qwen3.5-test", "Qwen"), ("mistral-small", "Mistral"),
    ("google/gemma-3", "Gemma"), ("gpt-oss-20b", "GPT-OSS"), ("custom-chat", "Mistral"),
])
def test_node_resolves_saved_endpoint_model_and_profile(monkeypatch, model, family):
    from minimax_h3_novel_pipeline.nodes import lmstudio_config

    data = provider_settings()
    data["lmstudio"].update(api_url="https://trusted.example/v1", model=model)
    settings.set_provider_settings(data)
    monkeypatch.setattr(lmstudio_config, "reserve_run", lambda: "test-run")
    config, status = lmstudio_config.LMStudioConfigurationNode().run()
    assert config["api_url"] == "https://trusted.example/v1"
    assert config["lmstudio_model"] == model
    assert config["model_family"] == family
    assert "local-key" not in str(config) + status
    assert "openai-key" not in str(settings.get_provider_options())


@pytest.mark.parametrize("field,value", [("enabled", False), ("api_key", ""), ("model", "")])
def test_node_rejects_unavailable_provider_before_reserving_run(monkeypatch, field, value):
    from minimax_h3_novel_pipeline.nodes import lmstudio_config

    data = provider_settings()
    data["openai"][field] = value
    settings.set_provider_settings(data)
    monkeypatch.setattr(lmstudio_config, "reserve_run", lambda: pytest.fail("Run reserved for unavailable provider"))
    with pytest.raises(ValueError, match="disabled|saved API key and model"):
        lmstudio_config.LMStudioConfigurationNode().run(backend="OpenAI")


def test_invalid_snapshot_preserves_both_connections():
    data = provider_settings()
    data["lmstudio"]["api_url"] = "https://new.example/v1"
    data["lmstudio"]["api_key"] = "replacement"
    data["openai"]["api_url"] = "https://untrusted.example/v1"
    with pytest.raises(ValueError):
        settings.set_provider_settings(data)
    assert settings.get_api_key(settings.DEFAULT_API_URL) == "local-key"
    assert settings.get_openai_api_key(settings.OPENAI_API_URL) == "openai-key"


@pytest.mark.parametrize("status,payload,error", [
    (200, {"data": [{"id": "gpt-4.1"}]}, None),
    (200, {"data": []}, "not available"),
    (200, {}, "model list"),
    (401, {"error": "secret-echo"}, "Authentication failed"),
    (403, {}, "Authentication failed"),
    (429, {}, "rate limit"),
    (307, {}, "HTTP 307"),
])
def test_connection_is_safe_and_does_not_save(monkeypatch, status, payload, error):
    requests = []
    original = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json=payload, headers={"Location": "https://untrusted.example"})

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        assert kwargs["timeout"] == 15.0
        return original(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    draft = deepcopy(provider_settings()["openai"])
    draft["api_key"] = "draft-key"
    if error:
        with pytest.raises(ValueError, match=error) as caught:
            asyncio.run(settings.test_connection("openai", draft))
        assert "secret-echo" not in str(caught.value)
    else:
        assert asyncio.run(settings.test_connection("openai", draft)) == "Connected. Model available."
    assert len(requests) == 1
    assert str(requests[0].url) == settings.OPENAI_API_URL + "/models"
    assert requests[0].headers["Authorization"] == "Bearer draft-key"
    assert settings.get_openai_api_key(settings.OPENAI_API_URL) == "openai-key"


def test_untrusted_openai_test_never_creates_client(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Client created before validating destination")
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    draft = provider_settings()["openai"]
    draft["api_url"] = "https://untrusted.example/v1"
    with pytest.raises(ValueError, match="restricted"):
        asyncio.run(settings.test_connection("openai", draft))


@pytest.mark.parametrize("payload,expected", [
    ({"data": [{"id": "z-model"}, {"id": "audio-model"}, {"id": "z-model"}]}, ["audio-model", "z-model"]),
    ({"data": []}, []),
])
@pytest.mark.parametrize("provider", ["lmstudio", "openai"])
def test_fetch_models_accepts_no_selection_and_does_not_save(monkeypatch, payload, expected, provider):
    original = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        return original(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    draft = provider_settings()[provider]
    draft.update(api_key="draft-key", model="")
    assert asyncio.run(settings.fetch_models(provider, draft)) == expected
    assert len(requests) == 1
    assert str(requests[0].url) == draft["api_url"] + "/models"
    assert requests[0].headers["Authorization"] == "Bearer draft-key"
    assert settings.get_openai_api_key(settings.OPENAI_API_URL) == "openai-key"
    assert settings.get_openai_model() == "gpt-4.1"
    assert settings.get_lmstudio_model() == ""


def test_lmstudio_selection_is_saved_atomically_and_tested(monkeypatch):
    data = provider_settings()
    data["lmstudio"]["model"] = " local-model "
    settings.set_provider_settings(data)
    assert settings.get_lmstudio_model() == "local-model"
    data["lmstudio"]["model"] = ["invalid"]
    with pytest.raises(ValueError, match="Model must be a string"):
        settings.set_provider_settings(data)
    assert settings.get_lmstudio_model() == "local-model"
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "local-model"}]})),
        **kwargs))
    data["lmstudio"]["model"] = "local-model"
    assert asyncio.run(settings.test_connection("lmstudio", data["lmstudio"])) == "Connected. Model available."
    data["lmstudio"]["model"] = "missing-model"
    with pytest.raises(ValueError, match="not available"):
        asyncio.run(settings.test_connection("lmstudio", data["lmstudio"]))


@pytest.mark.parametrize("entry", [{}, {"id": None}, {"id": 12}, {"id": ""}])
def test_fetch_models_rejects_malformed_ids(monkeypatch, entry):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [entry]})), **kwargs))
    with pytest.raises(ValueError, match="invalid model list"):
        asyncio.run(settings.fetch_models("openai", provider_settings()["openai"]))


def test_fetch_models_filters_past_shutdown_dates(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 4, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(settings, "datetime", FixedDatetime)
    entries = [
        {"id": "gpt-3.5-turbo-0125", "shutdown_date": "2026-10-23"},
        {"id": "o1-pro", "shutdown_date": "2026-10-23"},
        {"id": "yesterday", "shutdown_date": "2026-11-03"},
        {"id": "today", "shutdown_date": "2026-11-04"},
        {"id": "future", "shutdown_date": "2026-11-05"},
        {"id": "missing"},
        {"id": "null", "shutdown_date": None},
        {"id": "invalid", "shutdown_date": "2026-02-30"},
        {"id": "empty", "shutdown_date": ""},
        {"id": "number", "shutdown_date": 123},
    ]
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": entries})),
        **kwargs))
    draft = provider_settings()["openai"]
    assert asyncio.run(settings.fetch_models("openai", draft)) == [
        "empty", "future", "invalid", "missing", "null", "number", "today",
    ]
    entries[:] = entries[:2]
    assert asyncio.run(settings.fetch_models("openai", draft)) == []


def test_empty_model_can_be_saved_and_connection_tested(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "available"}]})),
        **kwargs))
    data = provider_settings()
    data["openai"]["model"] = ""
    settings.set_provider_settings(data)
    assert settings.get_openai_model() == ""
    assert asyncio.run(settings.test_connection("openai", data["openai"])) == "Connected. 1 model(s) available."
