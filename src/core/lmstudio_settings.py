"""In-memory bridge for the LM Studio key and trusted endpoint from settings."""
from __future__ import annotations

from threading import RLock
from urllib.parse import urlsplit


_lock = RLock()
_api_key = ""
DEFAULT_API_URL = "http://127.0.0.1:1234/v1"
_trusted_api_url = DEFAULT_API_URL
OPENAI_API_URL = "https://api.openai.com/v1"
_openai_api_key = ""
_openai_model = "gpt-4.1"
_enabled = {"LM Studio": True, "OpenAI": False}


def validate_connection(provider: str, data: dict) -> dict:
    """Validate a settings card before storing or testing its credentials."""
    if provider not in {"lmstudio", "openai"} or not isinstance(data, dict):
        raise ValueError("Invalid provider settings")
    url = _normalize_api_url(data.get("api_url"))
    key = data.get("api_key", "")
    enabled = data.get("enabled", False)
    if not isinstance(key, str) or not isinstance(enabled, bool):
        raise ValueError("Invalid API key or enabled flag")
    result = {"api_url": url, "api_key": key.strip(), "enabled": enabled}
    if provider == "openai":
        if url != OPENAI_API_URL:
            raise ValueError("OpenAI credentials are restricted to the public OpenAI API URL")
        model = data.get("model", "")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI model must not be empty")
        result["model"] = model.strip()
    return result


def set_provider_settings(data: dict) -> None:
    """Install a complete, validated snapshot atomically."""
    if not isinstance(data, dict):
        raise ValueError("Invalid provider settings")
    local = validate_connection("lmstudio", data.get("lmstudio"))
    cloud = validate_connection("openai", data.get("openai"))
    with _lock:
        set_connection_settings(local["api_url"], local["api_key"])
        set_openai_settings(cloud["api_key"], cloud["model"])
        _enabled.update({"LM Studio": local["enabled"], "OpenAI": cloud["enabled"]})


def require_enabled(backend: str) -> None:
    with _lock:
        if not _enabled.get(backend, False):
            raise ValueError(f"{backend} is disabled. Enable it in ComfyUI Settings → MiniMax H3 Novel → Providers.")


async def test_connection(provider: str, data: dict) -> str:
    """Test a draft without changing saved settings or issuing generation calls."""
    import httpx

    connection = validate_connection(provider, data)
    key = connection["api_key"]
    if provider == "openai" and not key:
        raise ValueError("Enter an OpenAI API key before testing")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=15.0) as client:
            response = await client.get(connection["api_url"] + "/models", headers=headers)
        if response.status_code in {401, 403}:
            raise ValueError("Authentication failed. Check the API key and permissions.")
        if response.status_code == 429:
            raise ValueError("Provider rate limit reached. Retry later.")
        if not response.is_success:
            raise ValueError(f"Connection test failed (HTTP {response.status_code}).")
        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise ValueError("The endpoint did not return a model list.")
        if provider == "openai" and not any(isinstance(item, dict) and item.get("id") == connection["model"] for item in models):
            raise ValueError("Connected, but the configured model is not available to this API key.")
        return "Connected. Model available." if provider == "openai" else f"Connected. {len(models)} model(s) available."
    except httpx.TimeoutException:
        raise ValueError("Connection timed out after 15 seconds.") from None
    except httpx.HTTPError:
        raise ValueError("Could not reach the provider. Check the URL and server.") from None


def set_openai_settings(api_key: str, model: str) -> None:
    global _openai_api_key, _openai_model
    if not isinstance(api_key, str) or not isinstance(model, str) or not model.strip():
        raise ValueError("OpenAI API key and model must be strings, and model must not be empty")
    with _lock:
        _openai_api_key, _openai_model = api_key.strip(), model.strip()


def get_openai_model() -> str:
    with _lock:
        return _openai_model


def get_openai_api_key(api_url: str) -> str:
    if api_url != OPENAI_API_URL:
        raise ValueError("OpenAI credentials are restricted to the public OpenAI API URL")
    with _lock:
        require_enabled("OpenAI")
        return _openai_api_key


def _normalize_api_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("Invalid LM Studio API URL.")
    url = url.strip().rstrip("/")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and parsed.port != 0 and parsed.username is None
                 and parsed.password is None and not parsed.query
                 and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid or any(c.isspace() or ord(c) < 32 for c in url) or "\\" in url or "?" in url or "#" in url:
        raise ValueError("Invalid LM Studio API URL; use an HTTP(S) endpoint without credentials, query, or fragment.")
    return url


def validate_api_url(value: str) -> str:
    """Bind credentials to the operator's endpoint, never a workflow's choice."""
    with _lock:
        trusted = _trusted_api_url
    if _normalize_api_url(value) != trusted:
        raise ValueError(
            "api_url must match Trusted API URL in ComfyUI Settings → "
            "MiniMax H3 Novel → LM Studio (default: http://127.0.0.1:1234/v1)."
        )
    return trusted


def set_connection_settings(api_url: str, api_key: str) -> None:
    """Receive the endpoint and its key together from the protected settings route."""
    global _trusted_api_url, _api_key
    trusted = _normalize_api_url(api_url)
    if not isinstance(api_key, str):
        raise ValueError("api_key must be a string")
    with _lock:
        _trusted_api_url = trusted
        _api_key = api_key.strip()


def set_api_key(value: str) -> None:
    """Store the key in memory only; never log or write it to a workflow/file."""
    global _api_key
    with _lock:
        _api_key = str(value or "").strip()


def get_api_key(api_url: str | None = None) -> str:
    with _lock:
        require_enabled("LM Studio")
        if api_url is not None:
            validate_api_url(api_url)
        return _api_key
