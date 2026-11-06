"""Mistral: standard chat template, optional reasoning and structured JSON."""

NAME = "Mistral"


def settings_from_config(config):
    return {"thinking": bool(config.get("thinking", False))}


def matches(model):
    return "mistral" in model.casefold()


def request_settings(model, system, user, settings):
    # Let LM Studio apply the GGUF's Mistral template. No Qwen control tokens.
    effort = "high" if settings["thinking"] else "none"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], {"reasoning_effort": effort, "chat_template_kwargs": {"reasoning_effort": effort}}, 0.9


def retries(model, settings):
    return 1


def allows_chatml(model, settings):
    return False
