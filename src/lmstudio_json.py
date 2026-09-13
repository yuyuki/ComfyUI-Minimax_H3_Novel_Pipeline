"""Structured LM Studio JSON requests shared by all three ComfyUI stages."""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from openai import OpenAI

from . import lmstudio_model_qwen, lmstudio_models

THINKING_ENABLED = lmstudio_model_qwen.DEFAULTS["thinking"]
CHAT_BACKEND = "structured-json"
QWEN35_LENGTH_RETRIES = lmstudio_model_qwen.DEFAULTS["length_retries"]
QWEN35_TOP_K = lmstudio_model_qwen.DEFAULTS["top_k"]
QWEN35_MIN_P = lmstudio_model_qwen.DEFAULTS["min_p"]
QWEN35_REPEAT_PENALTY = lmstudio_model_qwen.DEFAULTS["repeat_penalty"]

def _is_comfy_interrupt(error: BaseException) -> bool:
    """Do not retry a ComfyUI Stop request as though it were an LLM error."""
    return error.__class__.__name__ == "InterruptProcessingException"

def select_model(client: OpenAI, requested: str | None) -> str:
    if requested:
        return requested
    models = list(client.models.list().data)
    if not models:
        raise RuntimeError("LM Studio exposes no models. Load one first.")
    return next((m.id for m in models if "qwen" in m.id.casefold()), models[0].id)

def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise

def _complete_json_prefix(text: str) -> str | None:
    """Return the first complete top-level JSON object, or None if incomplete.

    This is intentionally a small streaming parser. It tracks string/escape state
    so braces inside JSON strings do not affect nesting depth.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def _qwen35_compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a retry schema small enough to finish under a short token cap."""
    compact = copy.deepcopy(schema)
    root_props = compact["schema"]["properties"]
    for name in ("characters", "locations", "objects"):
        if name in root_props:
            root_props[name]["maxItems"] = min(3, int(root_props[name].get("maxItems", 3)))

    def limit(node: dict[str, Any], field_name: str = "") -> None:
        if node.get("type") == "string" and "maxLength" in node:
            limit_by_field = {
                "chunk_summary": 240,
                "canonical_name": 80,
                "stable_visual_description": 180,
                "chapter_appearance": 160,
                "chapter_state": 160,
                "voice_description": 100,
                "evidence": 80,
            }
            node["maxLength"] = min(int(node["maxLength"]), limit_by_field.get(field_name, 80))
        if node.get("type") == "array":
            item_limit = {"aliases": 2, "distinguishing_features": 3, "reference_view_hints": 2, "evidence": 1}
            if field_name in item_limit:
                node["maxItems"] = min(int(node.get("maxItems", item_limit[field_name])), item_limit[field_name])
            items = node.get("items")
            if isinstance(items, dict):
                limit(items, field_name)
        for name, value in node.get("properties", {}).items():
            if isinstance(value, dict):
                limit(value, name)

    limit(compact["schema"])
    return compact


def model_settings(client, model):
    profile = getattr(client, "__dict__", {}).get("_minimax_h3_profile") or lmstudio_models.profile_for_model(model)
    settings = getattr(client, "__dict__", {}).get("_minimax_h3_settings") or dict(
        thinking=THINKING_ENABLED, length_retries=QWEN35_LENGTH_RETRIES,
        top_k=QWEN35_TOP_K, min_p=QWEN35_MIN_P, repeat_penalty=QWEN35_REPEAT_PENALTY,
    )
    return profile, settings


def chat_json(client: OpenAI, model: str, system: str, user: str,
              schema: dict[str, Any], temperature: float, max_tokens: int) -> dict[str, Any]:
    from .lmstudio_pipeline import comfy_interrupt_check

    profile, settings = model_settings(client, model)
    allow_chatml = profile.allows_chatml(model, settings)
    # Structured-output backends can occasionally end a response mid-string
    # even for non-Qwen3.5 models. Always allow one compact retry instead of
    # failing the whole ComfyUI run on that transient malformed response.
    retries = profile.retries(model, settings)
    # Keep compatibility discoveries on the client so subsequent stage requests
    # do not repeat a known sampler failure. A new client probes normally again.
    backend_key = (str(getattr(client, "base_url", "")), model)
    successful_chatml = vars(client).get("_minimax_h3_chatml_models", set())
    raw_chatml = allow_chatml and backend_key in successful_chatml
    attempt = 0
    while attempt <= retries:
        comfy_interrupt_check()
        request_schema = _qwen35_compact_schema(schema) if attempt else schema
        note = "\nReturn compact JSON with short descriptions and finish within the output limit." if attempt else ""
        messages, extra, top_p = profile.request_settings(model, system, user + note, settings)
        started = time.perf_counter()
        stream = None
        raw = ""
        content_chars = reasoning_chars = 0
        finish_reason = "not_received"
        local_stop = "stream_end"
        try:
            options = dict(
                model=model,
                temperature=min(temperature, 0.12) if attempt else temperature,
                top_p=top_p, max_tokens=max_tokens, stream=True,
            )
            response_format = {"type": "json_schema", "json_schema": request_schema}
            if raw_chatml:
                stream = client.completions.create(
                    **options, **profile.completion_options(messages),
                    extra_body={**extra, "response_format": response_format},
                )
            else:
                stream = client.chat.completions.create(
                    **options, messages=messages, response_format=response_format, extra_body=extra,
                )
            for event in stream:
                comfy_interrupt_check()
                if not event.choices:
                    continue
                choice = event.choices[0]
                reason = getattr(choice, "finish_reason", None)
                if reason is not None:
                    # Only log known metadata, never arbitrary server text.
                    finish_reason = reason if reason in {"stop", "length", "content_filter", "tool_calls", "function_call"} else "other"
                delta = None if raw_chatml else choice.delta
                content = (choice.text if raw_chatml else delta.content) or ""
                content_chars += len(content)
                for field in ("reasoning_content", "reasoning"):
                    reasoning = getattr(delta, field, None)
                    if isinstance(reasoning, str):
                        reasoning_chars += len(reasoning)
                raw += content
                complete = _complete_json_prefix(raw)
                if complete is not None:
                    raw = complete
                    local_stop = "json_complete"
                    break
        except BaseException as error:
            local_stop = "interrupted_or_error"
            if (allow_chatml and not raw_chatml
                    and not _is_comfy_interrupt(error) and lmstudio_model_qwen.is_thinking_grammar_error(error)):
                raw_chatml = True
                local_stop = "thinking_grammar_fallback"
                print("    LLM: thinking-token sampler failure; retrying structured JSON with raw ChatML.", flush=True)
                continue
            raise
        finally:
            if stream is not None:
                stream.close()
            diagnostics = (
                f"attempt={attempt + 1}, thinking={settings['thinking'] if profile.NAME == 'Qwen' else False}, max_tokens={max_tokens}, "
                f"content_chars={content_chars}, reasoning_chars={reasoning_chars}, "
                f"finish_reason={finish_reason}, local_stop={local_stop}"
            )
            print(f"    LLM stream: {diagnostics}", flush=True)
        try:
            result = parse_json(raw)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object.")
            if raw_chatml:
                successful_chatml.add(backend_key)
                client._minimax_h3_chatml_models = successful_chatml
            print(f"    LLM: structured JSON, {time.perf_counter() - started:.1f}s, attempt={attempt + 1}")
            return result
        except (ValueError, TypeError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Invalid structured JSON after {attempt + 1} attempt(s). {diagnostics}"
                ) from error
            attempt += 1
    raise AssertionError("Unreachable")
