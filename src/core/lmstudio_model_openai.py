"""OpenAI hosted chat: no LM Studio template controls."""

NAME = "OpenAI"


def settings_from_config(config):
    return {"thinking": False}


def request_settings(model, system, user, settings):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], {}, 0.9


def retries(model, settings):
    return 1


def allows_chatml(model, settings):
    return False
