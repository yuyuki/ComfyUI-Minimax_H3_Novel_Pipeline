"""Gemma: standard chat messages with LM Studio structured JSON output."""

NAME = "Gemma"


def settings_from_config(config):
    return {"thinking": False}


def matches(model):
    return "gemma" in model.casefold()


def request_settings(model, system, user, settings):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], {}, 1.0


def retries(model, settings):
    return 1


def allows_chatml(model, settings):
    return False
