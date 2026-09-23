"""In-memory bridge for the LM Studio key and trusted endpoint from settings."""
from __future__ import annotations

from threading import RLock
from urllib.parse import urlsplit


_lock = RLock()
_api_key = ""
DEFAULT_API_URL = "http://127.0.0.1:1234/v1"
_trusted_api_url = DEFAULT_API_URL


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
        if api_url is not None:
            validate_api_url(api_url)
        return _api_key
