"""Standalone ComfyUI node for chapter-reference extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from . import util


# Match ComfyUI's usual seed behavior: a newly created node starts with a
# different seed, and the widget can randomize it after each execution.
_DEFAULT_SEED = secrets.randbits(32)
_CHAPTER_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def _saved_chapter_choices() -> list[str]:
    """Return chapter files previously uploaded to ComfyUI's input folder."""
    try:
        import folder_paths

        root = Path(folder_paths.get_input_directory()) / "minimax_h3_novel"
        if not root.is_dir():
            return [""]
        files = sorted(
            (path for path in root.iterdir()
             if path.is_file() and path.suffix.lower() in _CHAPTER_EXTENSIONS),
            key=lambda path: path.name.lower(),
        )
        return [f"minimax_h3_novel/{path.name}" for path in files] or [""]
    except Exception:
        # Keep the node importable outside ComfyUI (for tests and tooling).
        return [""]


def _default_output_dir() -> str:
    """Return a ComfyUI-managed location for saved chapter catalogs."""
    try:
        import folder_paths
        return str(Path(folder_paths.get_output_directory()) / "minimax_h3_novel" / "chapter_catalogs")
    except Exception:
        return "output/minimax_h3_novel/chapter_catalogs"


def _log(message: str) -> None:
    """Write immediately visible progress messages to the ComfyUI console."""
    print(f"[minimax_h3_novel] {message}", flush=True)


def _token_sequence_length(tokens: Any) -> Optional[int]:
    """Return the actual prompt length from ComfyUI's batched token payload.

    Generative ComfyUI tokenizers may return either a token batch directly or
    a dictionary whose first value is that batch.  Calling ``len(tokens)`` on
    the dictionary only reports the number of dictionary fields (often 1),
    which made the progress log incorrectly show ``prompt_tokens=1``.
    """
    if isinstance(tokens, dict):
        if not tokens:
            return 0
        tokens = next(iter(tokens.values()))

    shape = getattr(tokens, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])

    try:
        # ComfyUI normally returns [batch][token].
        return len(tokens[0]) if len(tokens) else 0
    except (TypeError, IndexError, KeyError):
        return None


def _interrupt_comfyui_processing() -> None:
    """Stop the rest of the current workflow after a fatal node error.

    Raising from a node reports the error, but ComfyUI may still visit other
    nodes (or later items of a list input) in the same execution.  ComfyUI's
    normal interrupt flag is checked before each such call.  Keep this helper
    optional so the node remains usable in tests and outside ComfyUI.
    """
    try:
        import comfy.model_management as model_management

        interrupt = getattr(model_management, "interrupt_current_processing", None)
        if callable(interrupt):
            interrupt(True)
            _log("Fatal node error: interrupt requested for the remaining workflow")
    except Exception as exc:
        # The original exception is more useful than an optional interrupt
        # integration failure, especially when running outside ComfyUI.
        _log(f"Could not request workflow interrupt: {exc}")


def _dump_invalid_response(raw: str, label: str, debug_dir: Optional[Path]) -> Optional[Path]:
    """Persist a failed model response so JSON failures can be diagnosed."""
    # ``out_dir`` can arrive from ComfyUI as a plain string even though the
    # node type annotation says Path.
    target_dir = Path(debug_dir) if debug_dir else None
    if target_dir is None:
        target_dir = Path(tempfile.gettempdir()) / "minimax_h3_novel_json_debug"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    path = target_dir / f"{safe_label}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}.json"
    path.write_text(raw, encoding="utf-8")
    return path


def _split_chapter_chunks(text: str, max_chars: int, overlap_paragraphs: int) -> list[str]:
    """Split prose into bounded chunks, including a single long paragraph.

    The canonical script preserves paragraphs, which is useful for CLI usage
    but can emit a paragraph far bigger than ``chunk_chars``.  Generative CLIP
    models have a much tighter practical prompt/output budget, so this node
    must treat the setting as an actual upper bound.
    """
    max_chars = max(1, int(max_chars))
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs or [text.strip()]:
        remaining = paragraph
        while len(remaining) > max_chars:
            # Prefer a sentence break, then whitespace, before a hard split.
            window = remaining[:max_chars + 1]
            boundaries = [
                window.rfind(". "), window.rfind("! "), window.rfind("? "),
                window.rfind("; "), window.rfind(", "), window.rfind(" "),
            ]
            cut = max(boundaries)
            cut = (cut + 1) if cut > 0 else max_chars
            units.append(remaining[:cut].strip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            units.append(remaining)

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        def joined_length(parts: list[str]) -> int:
            return sum(map(len, parts)) + max(0, len(parts) - 1) * 2

        if current and joined_length(current + [unit]) > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-max(0, overlap_paragraphs):]
            # Do not let overlap defeat the configured bound.
            while current and joined_length(current + [unit]) > max_chars:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _load_pipeline_script():
    """Return the bundled extraction implementation; never import CLI scripts."""
    from . import pipeline_compat
    return pipeline_compat


def _decode_json_with_clip(
    clip: Any,
    system: str,
    user: str,
    schema: dict,
    temperature: float,
    max_tokens: int,
    seed: int = 0,
    debug_dir: Optional[Path] = None,
    debug_label: str = "response",
) -> dict:
    """Generate and parse one structured response using ComfyUI's CLIP API."""
    methods = [getattr(clip, name, None) for name in ("tokenize", "generate", "decode")]
    if not all(callable(method) for method in methods):
        raise TypeError(
            "The connected CLIP is not generative. Use Load CLIP with a supported "
            "generative text model, not an ordinary SD1/SDXL CLIP encoder."
        )
    tokenize, generate, decode = methods
    requested_tokens = max(256, int(max_tokens))
    generation_started = time.perf_counter()
    last_error: Exception | None = None
    raw = ""
    generated_ids: Any = []
    # ComfyUI's CLIP generation API does not expose token streaming, so it
    # cannot stop at the closing JSON brace. Retry a capped/bad response using
    # an intentionally smaller catalog request instead.
    for attempt in range(3):
        retry_note = ""
        if attempt:
            retry_note = (
                "\n\nCRITICAL RETRY: The previous response did not close valid JSON. "
                "Return a MUCH SMALLER catalog: at most 3 characters, 2 locations, and 2 objects. "
                "Use concise phrases, empty arrays for low-value entities, and close the JSON early."
            )
        prompt = (
            "/no_think\n\n" + system
            + "\n\nReturn ONLY valid JSON. Do not use Markdown fences or commentary.\n"
            + user + retry_note + "\n\nRequired JSON schema:\n"
            + json.dumps(schema["schema"], ensure_ascii=False)
        )
        tokens = tokenize(prompt, skip_template=False, thinking=False)
        _log(
            f"JSON generation attempt {attempt + 1}/3: budget={requested_tokens} tokens, "
            f"prompt_tokens={_token_sequence_length(tokens) or 0}"
        )
        try:
            generated_ids = generate(
                tokens, do_sample=temperature > 0, max_length=requested_tokens,
                temperature=max(0.01, float(temperature)), top_p=0.9, seed=int(seed) + attempt,
            )
        except TypeError:
            try:
                generated_ids = generate(tokens, do_sample=temperature > 0, max_length=requested_tokens, seed=int(seed) + attempt)
            except TypeError:
                generated_ids = generate(tokens, do_sample=False, max_length=requested_tokens)
        try:
            generated_count = len(generated_ids)
        except (TypeError, AttributeError):
            generated_count = None
        if generated_count is not None:
            termination = "max_tokens (no stop token before the limit)" if generated_count >= requested_tokens else "stop token"
            _log(f"JSON generation ended: generated_tokens={generated_count}, termination={termination}")
        try:
            raw = decode(generated_ids)
            result = _load_pipeline_script().parse_json(raw)
            _log(
                f"JSON generation succeeded in {time.perf_counter() - generation_started:.1f}s "
                f"({len(raw):,} response chars, attempt {attempt + 1}/3)"
            )
            return result
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                _log(f"JSON generation attempt {attempt + 1}/3 failed: {exc}; retrying with a compact catalog")
                continue

    exc = last_error or RuntimeError("Generative CLIP returned no JSON response")
    try:
        raw = locals().get("raw", "")
        dump_path = None
        if raw:
            try:
                dump_path = _dump_invalid_response(raw, debug_label, debug_dir)
            except Exception as dump_exc:
                _log(f"Could not dump invalid JSON response: {dump_exc}")
        position = getattr(exc, "pos", None)
        context = ""
        if isinstance(position, int) and raw:
            start = max(0, position - 180)
            end = min(len(raw), position + 180)
            context = f"; around error: {raw[start:end]!r}"
        token_cap_note = ""
        try:
            generated_count = len(generated_ids)
            if generated_count >= requested_tokens:
                token_cap_note = (
                    f" The JSON was truncated at max_tokens={requested_tokens}. "
                    "Increase max_tokens and run again (or reduce chunk_chars)."
                )
        except (TypeError, AttributeError):
            pass
        _log(
            f"JSON generation failed after {time.perf_counter() - generation_started:.1f}s: "
            f"{exc}{token_cap_note}{context}"
        )
        if dump_path:
            _log(f"Invalid JSON response dumped to: {dump_path}")
    finally:
        _interrupt_comfyui_processing()
    message = f"Generative CLIP returned invalid JSON after 3 attempts: {exc}"
    if token_cap_note:
        message += token_cap_note
    raise RuntimeError(message) from exc


@dataclass
class ExtractChapterReferencesNode:
    """Extract chapter-local reference catalogs with a generative CLIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "chapter_paths": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Path to each chapter file or a folder containing chapter files; one path per line.",
                }),
                "saved_chapter": (_saved_chapter_choices(), {
                    "tooltip": "Previously uploaded chapter. Selecting it fills the chapter paths field.",
                }),
                "out_dir": ("STRING", {
                    "default": _default_output_dir(),
                    "tooltip": "Folder for per-chapter reference JSON files.",
                }),
                "chunk_chars": ("INT", {"default": 5500, "min": 1000, "max": 1000000}),
                "overlap_paragraphs": ("INT", {"default": 2, "min": 0, "max": 100}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                # The schema contains several per-entity descriptions.  A
                # dense chunk can exceed 4096 tokens before its JSON closes.
                "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768}),
            },
            "optional": {
                "seed": ("INT", {
                    "default": _DEFAULT_SEED,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "control_after_generate": "randomize",
                }),
            },
        }

    # Keep the large structured result on the pipeline-only port.  The STRING
    # port is a compact status view rather than a second full JSON copy.
    RETURN_TYPES = ("MINIMAX_CHAPTERS", "STRING")
    RETURN_NAMES = ("chapter_catalogs", "catalog_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(
        self, clip: Any, chapter_paths: Iterable[Path], out_dir: str, **params
    ) -> tuple[List[dict], str]:
        started = time.perf_counter()
        if not isinstance(out_dir, str):
            raise TypeError("out_dir must be a string")
        if not out_dir.strip():
            raise ValueError("out_dir must not be empty")
        output_dir = Path(out_dir.strip())
        pipeline = _load_pipeline_script()
        if not all(callable(getattr(clip, name, None)) for name in ("tokenize", "generate", "decode")):
            raise TypeError(
                "ExtractChapterReferencesNode requires a generative CLIP. "
                "Load a generative text model through Load CLIP; standard SD CLIP "
                "encoders only encode prompts and cannot perform extraction."
            )
        raw_paths = chapter_paths
        if (not raw_paths or (isinstance(raw_paths, str) and not raw_paths.strip())) and params.get("saved_chapter"):
            raw_paths = params["saved_chapter"]
        input_items = [Path(line.strip()) for line in raw_paths.splitlines() if line.strip()] if isinstance(raw_paths, str) else [Path(item) for item in raw_paths]
        paths = util.discover_inputs(input_items)
        _log(f"Starting extraction: {len(paths)} chapter file(s) discovered")
        if not paths:
            raise ValueError("No supported chapter files found")
        args = argparse.Namespace(
            chunk_chars=int(params.get("chunk_chars", 5500)), overlap_paragraphs=int(params.get("overlap_paragraphs", 2)),
            temperature=float(params.get("temperature", 0.35)), max_tokens=int(params.get("max_tokens", 8192)),
            seed=int(params.get("seed", _DEFAULT_SEED)),
        )
        results: List[dict] = []
        for chapter_number, path in enumerate(paths, start=1):
            text = util.read_chapter(path)
            chapter_id = path.stem
            chunks = _split_chapter_chunks(text, max(1000, args.chunk_chars), max(0, args.overlap_paragraphs))
            _log(
                f"Chapter {chapter_number}/{len(paths)} '{path.name}': "
                f"{len(text):,} chars, {len(chunks)} chunk(s), "
                f"chunk_chars={args.chunk_chars}, overlap={args.overlap_paragraphs}"
            )
            chunk_results = []
            for index, chunk in enumerate(chunks, start=1):
                _log(f"Chapter '{chapter_id}': extracting chunk {index}/{len(chunks)} ({len(chunk):,} chars)")
                user = f"""Chapter ID: {chapter_id}
Passage chunk: {index}/{len(chunks)}

--- BEGIN NOVEL PASSAGE ---
{chunk}
--- END NOVEL PASSAGE ---"""
                chunk_result = _decode_json_with_clip(
                    clip, pipeline.EXTRACT_SYSTEM, user, pipeline.CHUNK_SCHEMA,
                    args.temperature, args.max_tokens, args.seed,
                debug_dir=output_dir, debug_label=f"{chapter_id}_chunk_{index}",
                )
                chunk_results.append(chunk_result)
                _log(
                    f"Chapter '{chapter_id}': chunk {index}/{len(chunks)} complete — "
                    f"characters={len(chunk_result.get('characters', []))}, "
                    f"locations={len(chunk_result.get('locations', []))}, "
                    f"objects={len(chunk_result.get('objects', []))}"
                )
            combined = pipeline.combine_candidates(chunk_results)
            _log(
                f"Chapter '{chapter_id}': combined candidates — "
                f"characters={len(combined['characters'])}, locations={len(combined['locations'])}, "
                f"objects={len(combined['objects'])}"
            )
            compact = {"chunk_summaries": combined["chunk_summaries"]}
            for kind in ("characters", "locations", "objects"):
                compact[kind] = [{k: v for k, v in item.items() if k != "_order"} for item in combined[kind]]
            merge_user = f"Chapter ID: {chapter_id}\n\nMerge this catalog:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
            _log(f"Chapter '{chapter_id}': merging overlapping candidates")
            merged = _decode_json_with_clip(
                clip, pipeline.MERGE_SYSTEM, merge_user, pipeline.MERGE_SCHEMA,
                min(args.temperature, 0.2), max(args.max_tokens, 3000), args.seed + 1000,
                debug_dir=output_dir, debug_label=f"{chapter_id}_merge",
            )
            catalog = pipeline.assign_local_ids(merged, combined)
            _log(
                f"Chapter '{chapter_id}': merge complete — "
                f"characters={len(catalog['characters'])}, locations={len(catalog['locations'])}, "
                f"objects={len(catalog['objects'])}"
            )
            payload = {
                "schema_version": pipeline.SCHEMA_VERSION, "chapter_id": chapter_id,
                "source": {"file": path.name, "absolute_path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "character_count": len(text)},
                "llm": {"backend": "comfyui-clip", "model": type(clip).__name__, "thinking": False}, **catalog,
            }
            results.append(payload)
            if output_dir is not None:
                util.save_json(output_dir / f"{chapter_id}_references.json", payload)
                _log(f"Chapter '{chapter_id}': saved {output_dir / f'{chapter_id}_references.json'}")
            _log(f"Chapter {chapter_number}/{len(paths)} '{chapter_id}' complete")
        _log(f"Extraction complete: {len(results)} chapter(s) in {time.perf_counter() - started:.1f}s")
        return (results, util.catalog_summary(results))
