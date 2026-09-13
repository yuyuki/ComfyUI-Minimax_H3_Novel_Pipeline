"""Qwen identification, sampling and Qwen3.5 template compatibility."""
from openai import APIError

NAME = "Qwen"
DEFAULTS = dict(thinking=False, length_retries=2, top_k=20, min_p=0.0, repeat_penalty=1.05)


def settings_from_config(config):
    return {
        "thinking": bool(config.get("thinking", DEFAULTS["thinking"])),
        "length_retries": max(0, int(config.get("qwen35_length_retries", DEFAULTS["length_retries"]))),
        "top_k": max(1, int(config.get("qwen35_top_k", DEFAULTS["top_k"]))),
        "min_p": min(1.0, max(0.0, float(config.get("qwen35_min_p", DEFAULTS["min_p"])))),
        "repeat_penalty": min(2.0, max(0.8, float(config.get("qwen35_repeat_penalty", DEFAULTS["repeat_penalty"])))),
    }


def matches(model):
    return "qwen" in model.casefold()


def is_qwen35(model):
    normalized = model.casefold().replace("_", "").replace("-", "").replace(".", "")
    return "qwen35" in normalized


def supports_thinking_prefill(model):
    normalized = model.casefold().replace("_", "").replace("-", "").replace(".", "")
    return is_qwen35(model) or "qwen38" in normalized


def request_settings(model, system, user, settings):
    thinking = settings["thinking"]
    special = is_qwen35(model)
    messages = [
        {"role": "system", "content": ("/think" if thinking else "/no_think") + "\n\n" + system},
        {"role": "user", "content": user},
    ]
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    if special:
        extra.update({key: settings[key] for key in ("top_k", "min_p", "repeat_penalty")})
    if supports_thinking_prefill(model) and not thinking:
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
    return messages, extra, 0.8 if special else 0.9


def retries(model, settings):
    return settings["length_retries"] if is_qwen35(model) else 1


def allows_chatml(model, settings):
    return supports_thinking_prefill(model) and not settings["thinking"]


def is_thinking_grammar_error(error):
    message = str(error).casefold()
    return (
        isinstance(error, APIError)
        and getattr(error, "status_code", None) in (None, 400)
        and "failed to initialize samplers" in message
        and "unexpected empty grammar stack" in message
        and "<think>" in message
    )


def completion_options(messages):
    prompt = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages[:2]
    ) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return dict(prompt=prompt, stop=["<|im_end|>", "<|endoftext|>"])
