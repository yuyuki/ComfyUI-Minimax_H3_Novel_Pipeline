"""Gemma: standard chat messages with LM Studio structured JSON output."""

NAME = "Gemma"


def settings_from_config(config):
    return {"thinking": bool(config.get("thinking", False))}


def matches(model):
    return "gemma" in model.casefold()


def request_settings(model, system, user, settings):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], {"chat_template_kwargs": {"enable_thinking": settings["thinking"]}}, 1.0


def retries(model, settings):
    return 1


def allows_chatml(model, settings):
    return False
