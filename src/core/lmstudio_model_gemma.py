"""Gemma: standard chat messages with LM Studio structured JSON output."""

NAME = "Gemma"


def settings_from_config(config):
    return {"thinking": bool(config.get("thinking", False))}


def matches(model):
    return "gemma" in model.casefold()


def request_settings(model, system, user, settings):
    thinking = settings["thinking"]
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    if not thinking and "gemma4" in model.casefold().replace("-", "").replace("_", "").replace(" ", ""):
        extra["reasoning_effort"] = "none"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], extra, 1.0


def retries(model, settings):
    return 1


def allows_chatml(model, settings):
    return False
