"""Mistral Small Instruct: standard chat template and structured JSON."""

NAME = "Mistral"


def settings_from_config(config):
    return {"thinking": False}


def matches(model):
    return "mistral" in model.casefold()


def request_settings(model, system, user, settings):
    # Let LM Studio apply the GGUF's Mistral template. No Qwen control tokens.
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], {}, 0.9


def retries(model, settings):
    return 1


def allows_chatml(model, settings):
    return False
