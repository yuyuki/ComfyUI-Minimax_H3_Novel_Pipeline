"""In-memory bridge for the LM Studio API key sent by ComfyUI settings."""
from __future__ import annotations

from threading import RLock
import os
from urllib.parse import urlsplit


_lock = RLock()
_api_key = ""


def validate_api_url(value: str) -> str:
    """Bind credentials to the operator's endpoint, never a workflow's choice."""
    trusted = os.environ.get("MINIMAX_H3_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")

    def normalize(url: str) -> str:
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

    trusted = normalize(trusted)
    if normalize(value) != trusted:
        raise ValueError(
            "api_url must match the server's trusted LM Studio endpoint. "
            "Set MINIMAX_H3_LMSTUDIO_BASE_URL in the environment running ComfyUI "
            "to authorize a different endpoint (default: http://127.0.0.1:1234/v1)."
        )
    return trusted


def set_api_key(value: str) -> None:
    """Store the key in memory only; never log or write it to a workflow/file."""
    global _api_key
    with _lock:
        _api_key = str(value or "").strip()


def get_api_key() -> str:
    with _lock:
        return _api_key
