"""GPT-OSS: LM Studio owns Harmony rendering and reasoning configuration."""

NAME = "GPT-OSS"


def settings_from_config(config):
    # GPT-OSS reasons natively; the Qwen thinking switch does not disable it.
    return {"thinking": True}


def matches(model):
    return "gpt-oss" in model.casefold().replace("_", "-")


def request_settings(model, system, user, settings):
    # Send ordinary chat messages, never raw ChatML or handcrafted Harmony tokens.
    # Keep reasoning effort under the operator's LM Studio model settings.
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], {}, 1.0


def retries(model, settings):
    return 1


def allows_chatml(model, settings):
    return False
