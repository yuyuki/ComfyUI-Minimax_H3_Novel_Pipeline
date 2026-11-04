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
def test_fetch_models_accepts_no_selection_and_does_not_save(monkeypatch, payload, expected):
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
    draft = provider_settings()["openai"]
    draft.update(api_key="draft-key", model="")
    assert asyncio.run(settings.fetch_models("openai", draft)) == expected
    assert len(requests) == 1
    assert str(requests[0].url) == settings.OPENAI_API_URL + "/models"
    assert requests[0].headers["Authorization"] == "Bearer draft-key"
    assert settings.get_openai_api_key(settings.OPENAI_API_URL) == "openai-key"
    assert settings.get_openai_model() == "gpt-4.1"


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
