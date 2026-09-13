"""Fingerprints of prompt text and non-secret settings used by disk caches."""
import hashlib
import json

from . import lmstudio_json


def fingerprint(model, args, *content, client=None):
    settings = {key: value for key, value in vars(args).items()
                if key not in {"out_dir", "force"} and isinstance(value, (str, int, float, bool, type(None)))}
    backend = {key: getattr(lmstudio_json, key) for key in (
        "THINKING_ENABLED", "CHAT_BACKEND", "QWEN35_LENGTH_RETRIES", "QWEN35_TOP_K",
        "QWEN35_MIN_P", "QWEN35_REPEAT_PENALTY",
    )}
    if getattr(client, "__dict__", {}).get("_minimax_h3_profile") is not None:
        profile, model_settings = lmstudio_json.model_settings(client, model)
        backend = {"profile": profile.NAME, "settings": model_settings, "chat_backend": lmstudio_json.CHAT_BACKEND}
    return hashlib.sha256(json.dumps([model, settings, backend, content],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()
