"""Offline settings validation and connection-test behavior."""
import asyncio
from copy import deepcopy

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
