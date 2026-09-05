#!/usr/bin/env python3
"""
STEP 1/3 — Extract chapter-local reference catalogs for a novel.

Reads one or more .txt/.md/.pdf chapter files and asks a local LM Studio LLM
for source-grounded characters, locations and important objects. Each chapter
gets its own JSON file with local IDs plus recommended reference-image views.

Example:
    python 01_extract_chapter_references.py chapters \
        --out-dir chapter_references \
        --model "your-loaded-qwen-model-id"

Requirements:
    pip install openai pypdf
"""
from __future__ import annotations

import argparse
import hashlib
import glob
import json
import re
import sys
import time
import threading
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

SCHEMA_VERSION = "minimax-h3-novel-refs.chapter.v2"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
SCRIPT_VERSION = "3.0.0"

# Qwen thinking control. Non-thinking is the default for this pipeline.
THINKING_ENABLED = False
CHAT_BACKEND = "auto"
QWEN35_MAX_OUTPUT_TOKENS = 2200
QWEN35_LENGTH_RETRIES = 2

CHARACTER_VIEWS = [
    "face_front",
    "full_body_front",
    "three_quarter",
    "back_view",
    "profile",
    "expression_closeup",
    "costume_detail",
]
LOCATION_VIEWS = [
    "wide_establishing",
    "secondary_angle",
    "reverse_angle",
    "key_detail",
    "interior_zone",
    "exterior_approach",
]
OBJECT_VIEWS = [
    "hero_three_quarter",
    "side_profile",
    "detail_closeup",
    "scale_context",
]

IMPORTANCE = ["major", "recurring", "minor", "background"]
PRIORITY = ["required", "recommended", "optional"]


def entity_schema(kind: str) -> dict[str, Any]:
    if kind == "character":
        props = {
            "canonical_name": {"type": "string", "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "stable_visual_description": {"type": "string", "maxLength": 500},
            "chapter_appearance": {"type": "string", "maxLength": 500},
            "distinguishing_features": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 120}},
            "voice_description": {"type": "string", "maxLength": 300},
            "speaks": {"type": "boolean"},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "enum": CHARACTER_VIEWS},
            },
            "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
        }
    elif kind == "location":
        props = {
            "canonical_name": {"type": "string", "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "stable_visual_description": {"type": "string", "maxLength": 500},
            "chapter_state": {"type": "string", "maxLength": 500},
            "distinguishing_features": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 120}},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 4,
                "items": {"type": "string", "enum": LOCATION_VIEWS},
            },
            "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
        }
    else:
        props = {
            "canonical_name": {"type": "string", "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "stable_visual_description": {"type": "string", "maxLength": 500},
            "chapter_state": {"type": "string", "maxLength": 500},
            "distinguishing_features": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 120}},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "enum": OBJECT_VIEWS},
            },
            "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
        }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


CHUNK_SCHEMA = {
    "name": "chapter_reference_candidates_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "chunk_summary": {"type": "string", "maxLength": 450},
            "characters": {"type": "array", "items": entity_schema("character")},
            "locations": {"type": "array", "items": entity_schema("location")},
            "objects": {"type": "array", "items": entity_schema("object")},
        },
        "required": ["chunk_summary", "characters", "locations", "objects"],
        "additionalProperties": False,
    },
}


def merge_entity_schema(kind: str) -> dict[str, Any]:
    s = entity_schema(kind)
    props = {
        "source_candidate_ids": {"type": "array", "maxItems": 24, "items": {"type": "string", "maxLength": 40}},
        **s["properties"],
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


MERGE_SCHEMA = {
    "name": "chapter_reference_catalog_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "chapter_summary": {"type": "string", "maxLength": 600},
            "characters": {"type": "array", "items": merge_entity_schema("character")},
            "locations": {"type": "array", "items": merge_entity_schema("location")},
            "objects": {"type": "array", "items": merge_entity_schema("object")},
        },
        "required": ["chapter_summary", "characters", "locations", "objects"],
        "additionalProperties": False,
    },
}


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value)]


def slug(text: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", text.strip(), flags=re.UNICODE).strip("._")
    return value or "chapter"


def discover_inputs(items: list[Path]) -> list[Path]:
    """Expand * and ? internally, then return supported files alphabetically."""
    expanded: list[Path] = []
    for item in items:
        raw = str(item)
        if "*" in raw or "?" in raw:
            matches = [Path(p) for p in glob.glob(raw)]
            if not matches:
                print(f"WARNING: input pattern matched nothing: {raw}", file=sys.stderr)
            expanded.extend(matches)
        else:
            expanded.append(item)

    found: list[Path] = []
    for item in expanded:
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(item)
        elif item.is_dir():
            found.extend(
                p for p in item.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            print(f"WARNING: ignoring unsupported/missing input: {item}", file=sys.stderr)

    unique = {str(p.resolve()).casefold(): p for p in found}
    return sorted(unique.values(), key=lambda p: (p.name.casefold(), str(p).casefold()))

def read_chapter(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    elif path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF input requires: pip install pypdf") from exc
        text = "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if len(text) < 100:
        raise ValueError("Chapter is empty or too short after extraction.")
    return text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_chunks(text: str, max_chars: int, overlap_paragraphs: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        add = len(para) + (2 if current else 0)
        if current and current_len + add > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(x) for x in current) + max(0, len(current) - 1) * 2
        current.append(para)
        current_len += add
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=300.0, max_retries=2)


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


def _is_qwen35_model(model: str) -> bool:
    normalized = model.casefold().replace("_", "").replace("-", "").replace(".", "")
    return "qwen35" in normalized


def _use_qwen35_chatml(model: str) -> bool:
    if CHAT_BACKEND == "qwen35-chatml":
        return True
    if CHAT_BACKEND == "openai-chat":
        return False
    return _is_qwen35_model(model)





def _format_eta(seconds: float | None) -> str:
    """Compact approximate remaining-time formatter for the live progress line."""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0 or seconds != seconds:
        return "estimating..."
    seconds = min(float(seconds), 99 * 24 * 3600 + 23 * 3600 + 59 * 60 + 59)
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"~{days}d {hours:02d}:{minutes:02d}"
    if hours:
        return f"~{hours:d}:{minutes:02d}:{secs:02d}"
    return f"~{minutes:02d}:{secs:02d}"


class _RunProgress:
    """Adaptive two-level progress/ETA tracker for one script run.

    The script supplies deterministic progress spans for known work (chunks,
    scenes, batches, etc.).  The currently running LLM call contributes a
    fractional amount learned from recently completed calls.  This keeps the
    percentage monotonic and avoids pretending max_tokens is the expected output.
    """

    def __init__(self, total_items: int) -> None:
        self.total_items = max(1, int(total_items))
        self.run_started = time.perf_counter()
        self.input_started = self.run_started
        self.current_index = 1
        self.current_name = ""
        self.current_progress = 0.0
        self.op_base = 0.0
        self.op_span = 0.0
        self.op_peak = 0.0
        self.call_samples: list[tuple[float, int]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def start_item(self, index: int, name: str = "") -> None:
        with self._lock:
            self.current_index = min(self.total_items, max(1, int(index)))
            self.current_name = name
            self.input_started = time.perf_counter()
            self.current_progress = 0.0
            self.op_base = 0.0
            self.op_span = 0.0
            self.op_peak = 0.0

    def start_operation(self, base: float, span: float) -> None:
        with self._lock:
            self.op_base = self._clamp(base)
            self.op_span = max(0.0, min(1.0 - self.op_base, float(span)))
            self.current_progress = max(self.current_progress, self.op_base)
            self.op_peak = 0.0

    def advance(self, progress: float) -> None:
        with self._lock:
            self.current_progress = max(self.current_progress, self._clamp(progress))
            self.op_base = self.current_progress
            self.op_span = 0.0
            self.op_peak = 0.0

    def finish_operation(self) -> None:
        with self._lock:
            self.current_progress = max(self.current_progress, self._clamp(self.op_base + self.op_span))
            self.op_base = self.current_progress
            self.op_span = 0.0
            self.op_peak = 0.0

    def finish_item(self) -> None:
        self.advance(1.0)

    def record_call(self, elapsed: float, token_events: int) -> None:
        if elapsed <= 0:
            return
        with self._lock:
            self.call_samples.append((float(elapsed), max(0, int(token_events))))
            if len(self.call_samples) > 24:
                del self.call_samples[:-24]

    @staticmethod
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0

    def _active_fraction(self, token_events: int, elapsed: float) -> float:
        durations = [d for d, _ in self.call_samples if d > 0]
        token_counts = [float(t) for _, t in self.call_samples if t > 0]
        expected_duration = self._median(durations)
        expected_tokens = self._median(token_counts)
        if expected_duration is None and expected_tokens is None:
            return 0.0

        time_fraction = (elapsed / expected_duration) if expected_duration else 0.0
        token_fraction = (token_events / expected_tokens) if expected_tokens else 0.0
        if token_events <= 0:
            # During TTFT only advance cautiously from historical wall-clock data.
            estimate = 0.25 * time_fraction
        elif expected_duration and expected_tokens:
            estimate = 0.45 * time_fraction + 0.55 * token_fraction
        elif expected_tokens:
            estimate = token_fraction
        else:
            estimate = time_fraction
        return min(0.97, max(0.0, estimate))

    def snapshot(self, token_events: int = 0, active_elapsed: float = 0.0) -> tuple[float, float | None, float, float | None]:
        now = time.perf_counter()
        with self._lock:
            active = self._active_fraction(token_events, active_elapsed)
            self.op_peak = max(self.op_peak, active)
            current = max(
                self.current_progress,
                self._clamp(self.op_base + self.op_span * self.op_peak),
            )
            total = self._clamp(((self.current_index - 1) + current) / self.total_items)
            input_elapsed = max(0.0, now - self.input_started)
            run_elapsed = max(0.0, now - self.run_started)

        # Ratio-based ETA becomes useful as soon as deterministic progress exists.
        input_eta = input_elapsed * (1.0 - current) / current if current >= 0.01 else None
        total_eta = run_elapsed * (1.0 - total) / total if total >= 0.005 else None
        return current, input_eta, total, total_eta


_RUN_PROGRESS: _RunProgress | None = None


def _progress_start_item(index: int, name: str = "") -> None:
    if _RUN_PROGRESS is not None:
        _RUN_PROGRESS.start_item(index, name)


def _progress_start_operation(base: float, span: float) -> None:
    if _RUN_PROGRESS is not None:
        _RUN_PROGRESS.start_operation(base, span)


def _progress_advance(progress: float) -> None:
    if _RUN_PROGRESS is not None:
        _RUN_PROGRESS.advance(progress)


def _progress_finish_operation() -> None:
    if _RUN_PROGRESS is not None:
        _RUN_PROGRESS.finish_operation()


def _progress_finish_item() -> None:
    if _RUN_PROGRESS is not None:
        _RUN_PROGRESS.finish_item()

def _enable_windows_ansi() -> bool:
    """Enable ANSI/VT escape sequences on Windows consoles when possible."""
    if sys.platform != "win32" or not sys.stdout.isatty():
        return sys.stdout.isatty()
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if handle in (0, -1) or not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False


class _LiveTokenRate:
    """Single-line live throughput plus current-input and whole-run ETA display."""

    _GREEN = "\x1b[92m"
    _CYAN = "\x1b[96m"
    _YELLOW = "\x1b[93m"
    _DIM = "\x1b[2m"
    _RESET = "\x1b[0m"
    _CLEAR_LINE = "\x1b[2K"

    def __init__(self, label: str = "LLM", refresh_interval: float = 0.25) -> None:
        self.label = label
        self.started = time.perf_counter()
        self.first_token_at: float | None = None
        self.token_events = 0
        self.refresh_interval = refresh_interval
        self.is_tty = sys.stdout.isatty()
        self.use_ansi = _enable_windows_ansi() if self.is_tty else False
        self._token_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.is_tty:
            self._thread = threading.Thread(target=self._refresh_loop, name="llm-live-meter", daemon=True)
            self._thread.start()

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self.refresh_interval):
            self._render()

    def update(self, piece: str) -> None:
        if not piece:
            return
        now = time.perf_counter()
        with self._token_lock:
            if self.first_token_at is None:
                self.first_token_at = now
            self.token_events += 1

    def _render(self, now: float | None = None) -> None:
        if not self.is_tty:
            return
        now = time.perf_counter() if now is None else now
        with self._token_lock:
            tokens = self.token_events
            first_token_at = self.first_token_at
        total_elapsed = max(now - self.started, 1e-9)
        generation_elapsed = max(now - first_token_at, 0.10) if first_token_at is not None else 0.0
        rate = (tokens / generation_elapsed) if generation_elapsed > 0 else 0.0

        progress_text = ""
        if _RUN_PROGRESS is not None:
            current, current_eta, total, total_eta = _RUN_PROGRESS.snapshot(tokens, total_elapsed)
            current_s = f"{current * 100:5.1f}%"
            total_s = f"{total * 100:5.1f}%"
            if self.use_ansi:
                progress_text = (
                    f" | Current: {self._CYAN}{current_s}{self._RESET} ETA {_format_eta(current_eta)}"
                    f" | Total: {self._YELLOW}{total_s}{self._RESET} ETA {_format_eta(total_eta)}"
                )
            else:
                progress_text = (
                    f" | Current: {current_s} ETA {_format_eta(current_eta)}"
                    f" | Total: {total_s} ETA {_format_eta(total_eta)}"
                )

        if self.use_ansi:
            msg = (
                f"    {self.label}: {self._GREEN}{rate:6.1f} tok/s{self._RESET} | "
                f"{self._DIM}{tokens:5d} tok{self._RESET}{progress_text}"
            )
            sys.stdout.write("\r" + self._CLEAR_LINE + msg)
        else:
            msg = f"    {self.label}: {rate:6.1f} tok/s | {tokens:5d} tok{progress_text}"
            sys.stdout.write("\r" + msg)
        sys.stdout.flush()

    def finish(self) -> tuple[int, float, float]:
        now = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.5, self.refresh_interval * 2))
        if self.is_tty:
            self._render(now)
            if self.use_ansi:
                sys.stdout.write(self._RESET)
            sys.stdout.write("\n")
            sys.stdout.flush()
        with self._token_lock:
            tokens = self.token_events
            first_token_at = self.first_token_at
        elapsed = max(now - self.started, 1e-9)
        generation_elapsed = max(now - first_token_at, 0.10) if first_token_at is not None else elapsed
        rate = tokens / generation_elapsed
        if _RUN_PROGRESS is not None:
            _RUN_PROGRESS.record_call(elapsed, tokens)
        return tokens, elapsed, rate


def _stream_chat_completion(client: OpenAI, **kwargs: Any) -> tuple[str, float, int, str, bool]:
    """Stream a chat completion while displaying live tokens/second."""
    meter = _LiveTokenRate("LLM")
    content_chunks: list[str] = []
    finish_reason = "unknown"
    reasoning_seen = False
    stream = client.chat.completions.create(stream=True, **kwargs)
    try:
        for event in stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            delta = choice.delta
            content = getattr(delta, "content", None) or ""
            reasoning = getattr(delta, "reasoning_content", None) or ""
            if content:
                content_chunks.append(content)
            if reasoning:
                reasoning_seen = True
            meter.update(content or reasoning)
            if getattr(choice, "finish_reason", None):
                finish_reason = str(choice.finish_reason)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    token_events, elapsed, _ = meter.finish()
    return "".join(content_chunks), elapsed, token_events, finish_reason, reasoning_seen

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


def _qwen35_stream_json_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, float, int, str]:
    """Stream manual ChatML JSON and show live generation throughput."""
    chunks: list[str] = []
    complete: str | None = None
    finish_reason = "json_complete"
    meter = _LiveTokenRate("LLM")
    stream = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<END_JSON>"],
        stream=True,
    )
    try:
        for event in stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            piece = choice.text or ""
            if piece:
                chunks.append(piece)
                meter.update(piece)
                current = "".join(chunks)
                complete = _complete_json_prefix(current)
                if complete is not None:
                    break
            if getattr(choice, "finish_reason", None):
                finish_reason = str(choice.finish_reason)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    token_events, elapsed, _ = meter.finish()
    raw = complete if complete is not None else "".join(chunks)
    return raw, elapsed, token_events, finish_reason

def _qwen35_chatml_prompt(system: str, user: str, schema: dict[str, Any]) -> str:
    # The Qwen3.5 GGUF template supplied by the model starts assistant generation
    # with <think>.  In non-thinking mode it inserts an EMPTY closed think block.
    # Building that prefix ourselves via /v1/completions avoids LM Studio builds
    # where enable_thinking/chat_template_kwargs are ignored by /v1/chat/completions.
    schema_text = json.dumps(schema["schema"], ensure_ascii=False)
    system_json = (
        system
        + "\n\nReturn ONLY valid JSON. Do not use Markdown fences or commentary. "
          "Your output must satisfy the JSON schema supplied by the user."
    )
    user_json = user + "\n\nRequired JSON schema:\n" + schema_text
    assistant_prefix = "<think>\n" if THINKING_ENABLED else "<think>\n\n</think>\n\n"
    return (
        "<|im_start|>system\n" + system_json + "<|im_end|>\n"
        "<|im_start|>user\n" + user_json + "<|im_end|>\n"
        "<|im_start|>assistant\n" + assistant_prefix
    )


def chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    # Qwen3.5-specific robust path. It bypasses LM Studio's chat-template
    # thinking toggle and json_schema/reasoning interaction by constructing the
    # model's ChatML prefix explicitly.
    if _use_qwen35_chatml(model):
        effective_max_tokens = min(max_tokens, QWEN35_MAX_OUTPUT_TOKENS)
        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(QWEN35_LENGTH_RETRIES + 1):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\n\nCRITICAL RETRY: The previous answer was truncated or invalid. "
                    "Return a MUCH SMALLER JSON object. Obey every maxItems/maxLength limit. "
                    "Use at most 3 evidence anchors per entity, each <=120 characters. "
                    "Do not quote dialogue. Do not narrate events. Finish and close the JSON well before the token limit."
                )
            prompt = _qwen35_chatml_prompt(system, user + retry_note, schema)
            try:
                raw, elapsed, completion_tokens, finish_reason = _qwen35_stream_json_completion(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=(min(temperature, 0.12) if attempt else temperature),
                    top_p=(0.8 if attempt else 0.9),
                    max_tokens=effective_max_tokens,
                )
                last_raw = raw
                token_note = f", {completion_tokens} streamed tokens, {completion_tokens / max(elapsed, 1e-9):.1f} tok/s"
                print(
                    f"    LLM: qwen35-chatml-stream, thinking={'on' if THINKING_ENABLED else 'off'}, "
                    f"{elapsed:.1f}s{token_note}, stop={finish_reason}, cap={effective_max_tokens}, attempt={attempt + 1}"
                )
                if finish_reason == "length":
                    last_error = RuntimeError(
                        f"generation hit output cap ({effective_max_tokens} tokens) before JSON completed"
                    )
                    if attempt < QWEN35_LENGTH_RETRIES:
                        print("    retrying with aggressive JSON compaction...")
                        continue
                try:
                    return parse_json(raw)
                except Exception as parse_error:
                    last_error = parse_error
                    if attempt < QWEN35_LENGTH_RETRIES:
                        print(f"    JSON incomplete/invalid ({parse_error}); retrying compactly...")
                        continue
            except Exception as error:
                last_error = error
                if attempt < QWEN35_LENGTH_RETRIES:
                    print(f"    Qwen3.5 call failed ({error}); retrying...")
                    continue
        elapsed_total = time.perf_counter() - start_time
        snippet = last_raw[-1200:] if last_raw else "<no output>"
        raise RuntimeError(
            f"Qwen3.5 manual ChatML completion failed after {elapsed_total:.1f}s and "
            f"{QWEN35_LENGTH_RETRIES + 1} attempt(s): {last_error}\n"
            f"Tail of last raw output:\n{snippet}"
        )

    # Generic OpenAI-compatible chat path for non-Qwen3.5 models.
    # Stream both the structured request and fallback so tok/s is visible live.
    thinking_directive = "/think" if THINKING_ENABLED else "/no_think"
    controlled_system = f"{thinking_directive}\\n\\n{system}"
    messages = [{"role": "system", "content": controlled_system}, {"role": "user", "content": user}]
    extra_body = {
        "enableThinking": bool(THINKING_ENABLED),
        "chat_template_kwargs": {"enable_thinking": bool(THINKING_ENABLED)},
    }
    try:
        raw, elapsed, completion_tokens, finish_reason, reasoning_seen = _stream_chat_completion(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": schema},
            extra_body=extra_body,
        )
        if not raw.strip() and reasoning_seen:
            raise RuntimeError(
                "LM Studio returned reasoning_content but empty content; "
                "try --chat-backend qwen35-chatml for a Qwen3.5 model."
            )
        print(
            f"    LLM: openai-chat structured, {elapsed:.1f}s, "
            f"{completion_tokens} streamed tokens, {completion_tokens / max(elapsed, 1e-9):.1f} tok/s, stop={finish_reason}"
        )
        return parse_json(raw)
    except Exception as first_error:
        raw, elapsed, completion_tokens, finish_reason, reasoning_seen = _stream_chat_completion(
            client,
            model=model,
            messages=[
                {"role": "system", "content": controlled_system + "\\nReturn ONLY valid JSON with no Markdown."},
                {
                    "role": "user",
                    "content": user + "\\n\\nRequired JSON schema:\\n" + json.dumps(schema["schema"], ensure_ascii=False),
                },
            ],
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        print(
            f"    LLM: openai-chat JSON fallback, {elapsed:.1f}s, "
            f"{completion_tokens} streamed tokens, {completion_tokens / max(elapsed, 1e-9):.1f} tok/s, stop={finish_reason}"
        )
        try:
            return parse_json(raw)
        except Exception as second_error:
            reasoning_note = "\\nReasoning stream was present." if reasoning_seen else ""
            raise RuntimeError(
                f"Structured output failed: {first_error}\\n"
                f"Fallback JSON failed: {second_error}{reasoning_note}\\n"
                f"Raw output:\\n{raw[:3000]}"
            ) from second_error


def compact_strings(values: Iterable[str], max_items: int, max_len: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = re.sub(r"\s+", " ", str(raw)).strip()[:max_len]
        key = value.casefold()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def clean_entity(entity: dict[str, Any], kind: str) -> dict[str, Any]:
    e = dict(entity)
    e["canonical_name"] = re.sub(r"\s+", " ", e.get("canonical_name", "")).strip()
    e["aliases"] = compact_strings(e.get("aliases", []), 6, 80)
    e["distinguishing_features"] = compact_strings(e.get("distinguishing_features", []), 6, 120)
    e["evidence"] = compact_strings(e.get("evidence", []), 3, 120)
    e["stable_visual_description"] = re.sub(r"\s+", " ", e.get("stable_visual_description", "")).strip()
    e["reference_view_hints"] = list(dict.fromkeys(e.get("reference_view_hints", [])))[:5]
    if kind == "characters":
        e["chapter_appearance"] = re.sub(r"\s+", " ", e.get("chapter_appearance", "")).strip()
        e["voice_description"] = re.sub(r"\s+", " ", e.get("voice_description", "")).strip()
        e["speaks"] = bool(e.get("speaks", False))
    else:
        e["chapter_state"] = re.sub(r"\s+", " ", e.get("chapter_state", "")).strip()
    return e


EXTRACT_SYSTEM = f"""
You are a continuity/reference analyst for a novel-to-video adaptation.

Extract ONLY facts supported by the supplied prose. Never invent age, ethnicity,
hair/eye color, body shape, clothing, architecture, accent, voice pitch, or other
traits that the chapter does not establish.

The goal is to identify reusable visual/audio reference entities for later MiniMax
H3 reference-to-video generation.

Character fields:
- stable_visual_description: identity-level traits likely to remain true.
- chapter_appearance: temporary wardrobe, injuries, dirt/wetness, disguise, carried
  items, age-state or other chapter-specific visible state.
- voice_description: only source-supported voice/delivery traits; empty if unknown.
- speaks: true only when the character actually speaks.
- reference_view_hints: choose views that would materially help preserve identity
  or reproduce likely shots. Allowed: {', '.join(CHARACTER_VIEWS)}.

Location fields:
- stable_visual_description: persistent layout/architecture/environment.
- chapter_state: temporary weather, lighting, damage, crowd, time-of-day, etc.
- reference_view_hints allowed: {', '.join(LOCATION_VIEWS)}.

Object fields:
- include only distinctive, recurring, plot-relevant, or visually important props.
- reference_view_hints allowed: {', '.join(OBJECT_VIEWS)}.

For view hints, be selective. A major character may justify face_front,
full_body_front, three_quarter and back_view; a major location may justify
wide_establishing, secondary_angle and key_detail. Minor entities usually need
only one view.

reference_priority:
- required: continuity would noticeably break without a reference.
- recommended: useful recurring consistency.
- optional: minor/background.

STRICT COMPACTNESS RULES:
- chunk_summary: max 450 characters.
- aliases: max 6 short items.
- distinguishing_features: max 6 items, max 120 characters each.
- reference_view_hints: max 5 for characters, 4 for locations, 3 for objects.
- evidence: MAXIMUM 3 items per entity, MAXIMUM 120 characters each.
- Evidence is only a compact factual anchor. NEVER reproduce long dialogue, long quotations, or whole sentences when a shorter anchor is enough.
- stable_visual_description/chapter_appearance/chapter_state: concise; do not narrate events.
- Keep stable_visual_description strictly persistent. Clothing, wounds, wetness, dirt, restraint state, carried gear, temporary exposure, and other scene-specific conditions belong in chapter_appearance/chapter_state, not the stable identity.
- Return the smallest JSON that fully captures continuity-relevant information.

Evidence must be brief and grounded in the supplied passage. Do not fabricate quotes.
""".strip()


MERGE_SYSTEM = """
Merge duplicate reference candidates extracted from overlapping chunks of ONE chapter.
Merge only candidates that clearly denote the same fictional character, location or
object. Preserve approximate first-appearance order. Do not merge merely similar
entities. Never invent missing visual/voice traits.

Combine reference_view_hints as the union of justified views. Keep the strongest
justified importance and reference_priority. Every output entity must list every
candidate_id it absorbed in source_candidate_ids.

Be compact: preserve at most 3 short evidence anchors per entity, never expand quotations, and keep descriptions concise. The merge output should normally be smaller than the combined input.
""".strip()


def extract_chunk(
    client: OpenAI,
    model: str,
    chapter_id: str,
    text: str,
    index: int,
    total: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    user = f"""Chapter ID: {chapter_id}
Passage chunk: {index}/{total}

--- BEGIN NOVEL PASSAGE ---
{text}
--- END NOVEL PASSAGE ---"""
    return chat_json(client, model, EXTRACT_SYSTEM, user, CHUNK_SCHEMA, args.temperature, args.max_tokens)


def combine_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {"chunk_summaries": [], "characters": [], "locations": [], "objects": []}
    counters = {"characters": 0, "locations": 0, "objects": 0}
    prefixes = {"characters": "C", "locations": "L", "objects": "O"}
    for chunk_no, result in enumerate(results, start=1):
        combined["chunk_summaries"].append(result.get("chunk_summary", ""))
        for kind in ("characters", "locations", "objects"):
            for raw in result.get(kind, []):
                counters[kind] += 1
                item = clean_entity(raw, kind)
                item["candidate_id"] = f"{prefixes[kind]}{counters[kind]:03d}_CHUNK{chunk_no:03d}"
                item["_order"] = counters[kind]
                combined[kind].append(item)
    return combined


def merge_candidates(
    client: OpenAI,
    model: str,
    chapter_id: str,
    combined: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    compact: dict[str, Any] = {"chunk_summaries": combined["chunk_summaries"]}
    for kind in ("characters", "locations", "objects"):
        compact[kind] = [{k: v for k, v in x.items() if k != "_order"} for x in combined[kind]]
    return chat_json(
        client,
        model,
        MERGE_SYSTEM,
        f"Chapter ID: {chapter_id}\n\nMerge this catalog:\n{json.dumps(compact, ensure_ascii=False, indent=2)}",
        MERGE_SCHEMA,
        min(args.temperature, 0.2),
        max(args.max_tokens, 6000),
    )


def assign_local_ids(merged: dict[str, Any], combined: dict[str, Any]) -> dict[str, Any]:
    candidate_order: dict[str, int] = {}
    for kind in ("characters", "locations", "objects"):
        for item in combined[kind]:
            candidate_order[item["candidate_id"]] = item["_order"]

    output: dict[str, Any] = {"chapter_summary": merged.get("chapter_summary", "")}
    for kind, prefix in (("characters", "CHAR"), ("locations", "LOC"), ("objects", "OBJ")):
        items = [clean_entity(x, kind) for x in merged.get(kind, [])]
        items.sort(
            key=lambda item: min(
                (candidate_order.get(x, 10**9) for x in item.get("source_candidate_ids", [])),
                default=10**9,
            )
        )
        final: list[dict[str, Any]] = []
        for i, item in enumerate(items, start=1):
            item.pop("source_candidate_ids", None)
            local_id = f"{prefix}_{i:03d}"
            final.append({"local_id": local_id, **item})
        output[kind] = final
    return output



def _hierarchical_merge_call_count(n: int, batch_size: int) -> int:
    n = max(1, n)
    if n == 1:
        return 1
    calls = 0
    while n > 1:
        n = (n + batch_size - 1) // batch_size
        calls += n
    return calls


def _merged_as_partial(merged: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_summary": merged.get("chapter_summary", ""),
        "characters": [{k: v for k, v in x.items() if k != "source_candidate_ids"} for x in merged.get("characters", [])],
        "locations": [{k: v for k, v in x.items() if k != "source_candidate_ids"} for x in merged.get("locations", [])],
        "objects": [{k: v for k, v in x.items() if k != "source_candidate_ids"} for x in merged.get("objects", [])],
    }


def hierarchical_merge_candidates(
    client: OpenAI, model: str, chapter_id: str, chunk_results: list[dict[str, Any]],
    args: argparse.Namespace, cache_dir: Path, op_index: int, progress_span: float,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Bound every chapter merge prompt by --merge-batch-size.

    Returns the final merged catalog plus the immediate combined catalog used to
    assign local IDs, and the next operation index.
    """
    level = list(chunk_results)
    round_no = 1
    if len(level) == 1:
        batches = [level]
    while True:
        next_level: list[dict[str, Any]] = []
        batch_count = (len(level) + args.merge_batch_size - 1) // args.merge_batch_size
        last_merged = None
        last_combined = None
        for batch_no, start in enumerate(range(0, len(level), args.merge_batch_size), start=1):
            batch = level[start:start + args.merge_batch_size]
            combined = combine_candidates(batch)
            key = hashlib.sha256((
                SCHEMA_VERSION + "\n" + model + "\n" + str(THINKING_ENABLED) + "\n" + CHAT_BACKEND + "\n" +
                json.dumps(combined, ensure_ascii=False, sort_keys=True, default=str)
            ).encode()).hexdigest()
            cache_path = cache_dir / f"merge_r{round_no:02d}_b{batch_no:03d}.json"
            merged = None
            if cache_path.exists() and not args.force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached.get("cache_key") == key:
                        merged = cached["result"]
                except Exception:
                    pass
            _progress_start_operation(op_index * progress_span, progress_span)
            if merged is None:
                print(f"  merging round {round_no}, batch {batch_no}/{batch_count} ({len(batch)} partial catalog(s))")
                merged = merge_candidates(client, model, chapter_id, combined, args)
                cache_path.write_text(json.dumps({"cache_key": key, "result": merged}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                _progress_finish_operation()
                if args.delay:
                    time.sleep(args.delay)
            else:
                _progress_advance((op_index + 1) * progress_span)
                print(f"  merge round {round_no}, batch {batch_no}/{batch_count}: cached")
            op_index += 1
            last_merged, last_combined = merged, combined
            next_level.append(_merged_as_partial(merged))
        if len(next_level) == 1:
            assert last_merged is not None and last_combined is not None
            return last_merged, last_combined, op_index
        level = next_level
        round_no += 1

def process_chapter(
    path: Path,
    out_dir: Path,
    client: OpenAI,
    model: str,
    args: argparse.Namespace,
) -> Path:
    chapter_id = slug(path.stem)
    out_path = out_dir / f"{chapter_id}_references.json"
    source_hash = sha256_file(path)

    if out_path.exists() and not args.force:
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            if old.get("schema_version") == SCHEMA_VERSION and old.get("source", {}).get("sha256") == source_hash:
                print(f"SKIP {path.name}: unchanged v2 output exists.")
                return out_path
        except Exception:
            pass

    text = read_chapter(path)
    chunks = split_chunks(text, max(3000, args.chunk_chars), max(0, args.overlap_paragraphs))
    merge_calls = _hierarchical_merge_call_count(len(chunks), args.merge_batch_size)
    progress_units = len(chunks) + merge_calls
    progress_span = 1.0 / max(1, progress_units)
    cache_dir = out_dir / ".cache" / chapter_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"{path.name}: {len(text):,} chars, {len(chunks)} chunk(s), {merge_calls} merge call(s)")
    chunk_results: list[dict[str, Any]] = []
    op_index = 0
    for i, chunk in enumerate(chunks, start=1):
        _progress_start_operation(op_index * progress_span, progress_span)
        cache_path = cache_dir / f"chunk_{i:03d}.json"
        cache_key = hashlib.sha256((SCHEMA_VERSION + "\n" + model + "\nthinking=" + str(THINKING_ENABLED) + "\nchat_backend=" + CHAT_BACKEND + "\n" + chunk).encode()).hexdigest()
        result = None
        if cache_path.exists() and not args.force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == cache_key:
                    result = cached["result"]
            except Exception:
                pass
        if result is None:
            print(f"  extracting chunk {i}/{len(chunks)}")
            result = extract_chunk(client, model, chapter_id, chunk, i, len(chunks), args)
            cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _progress_finish_operation()
            if args.delay:
                time.sleep(args.delay)
        else:
            _progress_advance((op_index + 1) * progress_span)
            print(f"  chunk {i}/{len(chunks)}: cached")
        op_index += 1
        chunk_results.append(result)

    merged, combined, op_index = hierarchical_merge_candidates(
        client, model, chapter_id, chunk_results, args, cache_dir, op_index, progress_span
    )
    _progress_advance(1.0)

    catalog = assign_local_ids(merged, combined)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "chapter_id": chapter_id,
        "source": {
            "file": path.name,
            "absolute_path": str(path.resolve()),
            "sha256": source_hash,
            "character_count": len(text),
        },
        "llm": {"base_url": args.base_url, "model": model, "thinking": THINKING_ENABLED, "chat_backend": CHAT_BACKEND},
        **catalog,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"  saved {out_path.name}: {len(payload['characters'])} characters, "
        f"{len(payload['locations'])} locations, {len(payload['objects'])} objects"
    )
    return out_path



def _cli_quote(value: Any) -> str:
    """Quote one argument for a copy/paste friendly shell command."""
    text = str(value)
    if not text:
        return '""'
    if re.search(r'[\\s"&|<>^()]', text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _format_command(parts: list[Any]) -> str:
    return " ".join(_cli_quote(x) for x in parts)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract v2 per-chapter novel reference catalogs.")
    p.add_argument("inputs", nargs="+", type=Path, help="Chapter file(s), directories, and/or * ? wildcard patterns. Matches are processed alphabetically.")
    p.add_argument("--out-dir", type=Path, default=Path("chapter_references"))
    p.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--api-key", default="lm-studio")
    p.add_argument("--model", default=None)
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Enable Qwen reasoning/thinking mode (/think).",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable Qwen reasoning/thinking mode (/no_think, default).",
    )
    p.set_defaults(thinking=False)
    p.add_argument(
        "--chat-backend",
        choices=["auto", "openai-chat", "qwen35-chatml"],
        default="auto",
        help=(
            "LLM transport for JSON calls. auto uses manual Qwen3.5 ChatML via "
            "/v1/completions for qwen3.5/qwen35 model IDs, otherwise OpenAI chat. "
            "qwen35-chatml is the recommended workaround when LM Studio ignores "
            "enable_thinking=false for Qwen3.5."
        ),
    )
    p.add_argument("--chunk-chars", type=int, default=5500)
    p.add_argument("--overlap-paragraphs", type=int, default=2)
    p.add_argument("--merge-batch-size", type=int, default=6, help="Maximum partial chunk catalogs per hierarchical merge call.")
    p.add_argument("--temperature", type=float, default=0.18)
    p.add_argument("--max-tokens", type=int, default=2200)
    p.add_argument(
        "--qwen35-max-output-tokens", "--max-output-tokens",
        dest="qwen35_max_output_tokens",
        type=int,
        default=2200,
        help=(
            "Safety cap for manual Qwen3.5 ChatML JSON completions. "
            "Streaming normally stops earlier as soon as a complete JSON object closes."
        ),
    )
    p.add_argument(
        "--qwen35-length-retries",
        type=int,
        default=2,
        help="Retry a Qwen3.5 JSON call compactly when output hits the token cap or parses as incomplete.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--force", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    global THINKING_ENABLED, CHAT_BACKEND, QWEN35_MAX_OUTPUT_TOKENS, QWEN35_LENGTH_RETRIES, _RUN_PROGRESS
    THINKING_ENABLED = bool(args.thinking)
    CHAT_BACKEND = args.chat_backend
    QWEN35_MAX_OUTPUT_TOKENS = max(256, int(args.qwen35_max_output_tokens))
    QWEN35_LENGTH_RETRIES = max(0, int(args.qwen35_length_retries))
    args.merge_batch_size = max(2, int(args.merge_batch_size))
    files = discover_inputs(args.inputs)
    if not files:
        print("ERROR: no supported chapter files found.", file=sys.stderr)
        return 2
    _RUN_PROGRESS = _RunProgress(len(files))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        lm = make_client(args.base_url, args.api_key)
        model = select_model(lm, args.model)
    except Exception as exc:
        print(f"ERROR connecting to LM Studio: {exc}", file=sys.stderr)
        return 1

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"LM Studio: {args.base_url}")
    print(f"Model: {model}")
    print(f"Thinking: {'enabled' if THINKING_ENABLED else 'disabled'}")
    resolved_backend = "qwen35-chatml" if _use_qwen35_chatml(model) else "openai-chat"
    print(f"Chat backend: {resolved_backend} (requested: {CHAT_BACKEND})")
    if resolved_backend == "qwen35-chatml":
        print(f"Qwen3.5 output cap: {QWEN35_MAX_OUTPUT_TOKENS} tokens (stream stops at complete JSON)")
        print(f"Qwen3.5 compact retries: {QWEN35_LENGTH_RETRIES}")
    print(f"Chapters: {len(files)}\n")
    failures = 0
    for input_index, path in enumerate(files, start=1):
        _progress_start_item(input_index, path.name)
        try:
            process_chapter(path, args.out_dir, lm, model, args)
        except Exception as exc:
            failures += 1
            print(f"ERROR {path}: {exc}", file=sys.stderr)
        finally:
            _progress_finish_item()
    print(f"\nCompleted: {len(files) - failures}/{len(files)} chapter(s).")
    if failures == 0:
        next_parts: list[Any] = [
            sys.executable,
            "02_consolidate_references.py",
            args.out_dir,
            "--out", "consolidated_references.json",
            "--asset-prompts-out", "reference_asset_prompts.txt",
            "--base-url", args.base_url,
            "--model", model,
            "--thinking" if THINKING_ENABLED else "--no-thinking",
            "--chat-backend", resolved_backend,
            "--max-output-tokens", QWEN35_MAX_OUTPUT_TOKENS,
            "--qwen35-length-retries", QWEN35_LENGTH_RETRIES,
        ]
        if args.api_key != "lm-studio":
            next_parts.extend(["--api-key", "YOUR_LM_STUDIO_API_KEY"])
        print("\nNext recommended command (Step 2/3):")
        print(_format_command(next_parts))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
