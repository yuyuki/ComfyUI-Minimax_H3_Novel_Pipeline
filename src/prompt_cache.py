"""Fingerprints of prompt text and non-secret settings used by disk caches."""
import hashlib
import json

from . import lmstudio_json


def fingerprint(model, args, *content):
    settings = {key: value for key, value in vars(args).items()
                if key not in {"out_dir", "force"} and isinstance(value, (str, int, float, bool, type(None)))}
    backend = {key: getattr(lmstudio_json, key) for key in (
        "THINKING_ENABLED", "CHAT_BACKEND", "QWEN35_LENGTH_RETRIES", "QWEN35_TOP_K",
        "QWEN35_MIN_P", "QWEN35_REPEAT_PENALTY",
    )}
    return hashlib.sha256(json.dumps([model, settings, backend, content],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()
