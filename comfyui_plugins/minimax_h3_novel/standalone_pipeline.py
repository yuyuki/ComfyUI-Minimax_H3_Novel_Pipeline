"""Bundled implementation shared by the standalone ComfyUI nodes."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import util

INPUT_SCHEMA = "minimax-h3-novel-refs.chapter.v2"
LEGACY_INPUT_SCHEMA = "minimax-h3-novel-refs.chapter.v1"
OUTPUT_SCHEMA = "minimax-h3-novel-refs.consolidated.v2"


def natural_key(value: str) -> list[Any]:
    return util.natural_key(value)


def _key(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), re.UNICODE))


def _priority(value: str) -> int:
    return {"optional": 0, "recommended": 1, "required": 2}.get(value, 1)


def _views(entity: dict[str, Any], args: Any) -> list[str]:
    kind = entity["entity_type"]
    defaults = {
        "character": ["face_front", "full_body_front", "three_quarter", "back_view"],
        "location": ["wide_establishing", "secondary_angle", "key_detail"],
        "object": ["hero_three_quarter", "detail_closeup"],
    }[kind]
    limit = getattr(args, f"max_{kind}_base_views")
    return list(dict.fromkeys(defaults + entity.get("reference_view_hints", [])))[:limit]


def consolidate(chapters: list[dict[str, Any]], args: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    registry: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    counters = {"character": 0, "location": 0, "object": 0}
    plural = {"character": "characters", "location": "locations", "object": "objects"}
    prefixes = {"character": "CHAR", "location": "LOC", "object": "OBJ"}
    for chapter in chapters:
        chapter_id = chapter["chapter_id"]
        for kind, collection in plural.items():
            for item in chapter.get(collection, []):
                names = [item.get("canonical_name", ""), *item.get("aliases", [])]
                existing = next((by_key[(kind, _key(name))] for name in names if (kind, _key(name)) in by_key), None)
                if existing is None:
                    counters[kind] += 1
                    existing = {"global_id": f"{prefixes[kind]}_{counters[kind]:03d}", "entity_type": kind,
                        "canonical_name": item.get("canonical_name", "Unnamed"), "aliases": [],
                        "stable_visual_description": item.get("stable_visual_description", ""),
                        "distinguishing_features": [], "voice_description": item.get("voice_description", ""),
                        "speaks": bool(item.get("speaks", False)), "importance": item.get("importance", "minor"),
                        "reference_priority": item.get("reference_priority", "recommended"), "reference_view_hints": [],
                        "chapter_appearances": [], "source_local_ids": []}
                    registry.append(existing)
                existing["aliases"] = list(dict.fromkeys(existing["aliases"] + [x for x in names if x and x != existing["canonical_name"]]))
                existing["distinguishing_features"] = list(dict.fromkeys(existing["distinguishing_features"] + item.get("distinguishing_features", [])))[:12]
                existing["reference_view_hints"] = list(dict.fromkeys(existing["reference_view_hints"] + item.get("reference_view_hints", [])))
                if _priority(item.get("reference_priority", "recommended")) > _priority(existing["reference_priority"]): existing["reference_priority"] = item["reference_priority"]
                existing["chapter_appearances"].append(chapter_id); existing["source_local_ids"].append({"chapter_id": chapter_id, "local_id": item.get("local_id", "")})
                for name in names:
                    if name: by_key[(kind, _key(name))] = existing
    registry.sort(key=lambda e: ({"character": 0, "location": 1, "object": 2}[e["entity_type"]], natural_key(e["global_id"])))
    pictures: list[dict[str, Any]] = []; audio: list[dict[str, Any]] = []
    for entity in registry:
        if _priority(entity["reference_priority"]) >= _priority(args.picture_threshold):
            for view in _views(entity, args):
                asset_id = f"PIC_{entity['global_id']}_{view.upper()}"
                pictures.append({"asset_id": asset_id, "linked_global_id": entity["global_id"], "canonical_name": entity["canonical_name"], "view_type": view, "variant": "base", "suggested_filename": asset_id.lower()+".png", "description": entity["stable_visual_description"], "generation_prompt": f"Reference image of {entity['canonical_name']}, {view.replace('_', ' ')}. {entity['stable_visual_description']}"})
        if entity["entity_type"] == "character" and entity.get("speaks") and _priority(entity["reference_priority"]) >= _priority(args.audio_threshold):
            asset_id = f"AUD_{entity['global_id']}_VOICE"; audio.append({"asset_id": asset_id, "linked_global_id": entity["global_id"], "canonical_name": entity["canonical_name"], "suggested_filename": asset_id.lower()+".wav", "description": entity.get("voice_description", ""), "generation_prompt": f"Voice reference for {entity['canonical_name']}. {entity.get('voice_description', '')}"})
    return registry, pictures, audio


def chapter_map(registry: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for entity in registry:
        for source in entity["source_local_ids"]: out.setdefault(source["chapter_id"], {})[source["local_id"]] = entity["global_id"]
    return out


def asset_index(registry: list[dict[str, Any]], pictures: list[dict[str, Any]], audio: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    return {e["global_id"]: {"pictures": [a["asset_id"] for a in pictures if a["linked_global_id"] == e["global_id"]], "audio": [a["asset_id"] for a in audio if a["linked_global_id"] == e["global_id"]]} for e in registry}


def write_asset_prompts(path: Path, pictures: list[dict[str, Any]], audio: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(f"{a['asset_id']}\n{a['generation_prompt']}" for a in [*pictures, *audio]), encoding="utf-8")


def process_chapter(path: Path, refs: dict[str, Any], _client: Any, _model: str, args: Any) -> dict[str, Any]:
    """Write a portable baseline H3 scene and its explicit reference order."""
    chapter_id = path.stem; text = util.read_chapter(path); target = args.out_dir / chapter_id; target.mkdir(parents=True, exist_ok=True)
    pictures = refs.get("picture_assets", [])[:args.max_pictures]; audio = refs.get("audio_assets", [])[:args.max_audio]
    subjects = []; picture_order = []
    for number, asset in enumerate(pictures, 1):
        subject, picture = f"<Subject {number}>", f"<Picture {number}>"
        subjects.append({"h3_subject_label": subject, "global_id": asset["linked_global_id"], "canonical_name": asset.get("canonical_name", ""), "pictures": [{"h3_picture_label": picture, "asset_id": asset["asset_id"], "view_type": asset.get("view_type", "")} ]})
        picture_order.append({"h3_picture_label": picture, "asset_id": asset["asset_id"], "subject_label": subject, "global_id": asset["linked_global_id"]})
    audio_order = [{"h3_audio_label": f"<Audio {i}>", "asset_id": a["asset_id"], "global_id": a["linked_global_id"], "canonical_name": a.get("canonical_name", "")} for i, a in enumerate(audio, 1)]
    bindings = {"subjects": subjects, "audio": audio_order, "picture_input_order": picture_order, "audio_input_order": audio_order, "unreferenced_visible_entities": []}
    definitions = "\n".join(f"{s['h3_subject_label']} {s['canonical_name']}: use {s['pictures'][0]['h3_picture_label']}." for s in subjects) or "No external visual references."
    prompt = "\n".join(["summary:", "[reference generation] Adapt this chapter into a cinematic beat.", "subject_definitions:", definitions, "audio_definitions:", "None.", "detailed_description:", f"[Shot 1] Faithfully depict this narrative beat: {text[:1800]}", "overall_soundscape:", "Natural scene ambience appropriate to the setting.", "non_diegetic_music:", "Restrained cinematic score only if it supports the scene.", ""])
    util.save_json(target / "scene_001_assets.json", bindings); (target / "scene_001_prompt.txt").write_text(prompt, encoding="utf-8")
    return {"chapter_id": chapter_id, "saved_prompt_count": 1, "outputs": [{"scene_id": "scene_001", "prompt_file": "scene_001_prompt.txt", "assets_file": "scene_001_assets.json"}]}
