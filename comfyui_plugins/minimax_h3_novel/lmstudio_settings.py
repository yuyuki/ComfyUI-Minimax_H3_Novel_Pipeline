"""In-memory bridge for the LM Studio API key sent by ComfyUI settings."""
from __future__ import annotations

from threading import RLock


_lock = RLock()
_api_key = ""


def set_api_key(value: str) -> None:
    """Store the key in memory only; never log or write it to a workflow/file."""
    global _api_key
    with _lock:
        _api_key = str(value or "").strip()


def get_api_key() -> str:
    with _lock:
        return _api_key
