"""Utility helpers ported from the pipeline scripts for use by plugin nodes.

Keep these helpers small and focused so `nodes.py` can call them. They intentionally
mirror a subset of the pipeline scripts' behaviour (file discovery, reading,
and simple JSON helpers).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .path_access import input_path, output_path
from .path_access import loader_path as loader_path
from .source_text import read_source

CHAPTER_SCHEMA = "minimax-h3-novel-refs.chapter.v5"
REGISTRY_SCHEMA = "minimax-h3-novel-refs.consolidated.v5"
REFERENCE_TIMELINE_VERSION = "cinematic-neutral-references.v2"


def require_schema(data: Any, expected: str) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != expected:
        raise ValueError(f"Expected current schema {expected}; regenerate older outputs.")
    if expected == REGISTRY_SCHEMA and data.get("timeline_version") != REFERENCE_TIMELINE_VERSION:
        raise ValueError("Regenerate the chapter catalogs and reconsolidate references with the current appearance rules.")
    if expected == REGISTRY_SCHEMA and "character_enrichment" in data:
        raise ValueError("Reconsolidate references from the saved chapter catalogs and character visual profiles. "
                         "The previous character enrichment can change catalog visibility.")


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value)]


def discover_inputs(items: list[Path]) -> list[Path]:
    found: list[Path] = []
    for item in items:
        item = input_path(item)
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(item)
        elif item.is_dir():
            found.extend(
                input_path(p) for p in item.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
    return sorted(dict.fromkeys(found), key=lambda p: natural_key(p.name))


def read_chapter(path: Path) -> str:
    text, _ = read_source(path)
    if len(text) < 100:
        raise ValueError("Chapter is empty or too short after extraction.")
    return text


def load_json(path: Path) -> Any:
    path = output_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def catalog_summary(catalogs: Iterable[dict[str, Any]]) -> str:
    """Return a compact inspection string without duplicating full JSON on a port."""
    rows: list[str] = []
    for catalog in catalogs:
        rows.append(
            f"{catalog.get('chapter_id', '<unknown>')}: "
            f"{len(catalog.get('characters', []))} characters, "
            f"{len(catalog.get('locations', []))} locations, "
            f"{len(catalog.get('objects', []))} objects"
        )
    return "\n".join(rows) or "No chapter catalogs."


def registry_summary(registry: dict[str, Any]) -> str:
    """Return compact reference counts for a text preview node."""
    entities = registry.get("entities", [])
    counts = {kind: sum(item.get("entity_type") == kind for item in entities)
              for kind in ("character", "location", "object")}
    return (
        f"{len(registry.get('chapters', []))} chapters: "
        f"{counts['character']} characters, {counts['location']} locations, {counts['object']} objects\n"
        f"{len(registry.get('picture_assets', []))} picture briefs, "
        f"{len(registry.get('audio_assets', []))} audio briefs"
    )


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
