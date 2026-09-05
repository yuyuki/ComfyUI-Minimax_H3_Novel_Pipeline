"""ComfyUI pipeline step3 generate implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import lmstudio_json as json_backend
from .lmstudio_json import chat_json, select_model as select_model, _is_comfy_interrupt

from .path_access import confined_path


# Qwen thinking control. Non-thinking is the default for this pipeline.


REFERENCE_SCHEMA = "minimax-h3-novel-refs.consolidated.v2"

SECTIONS = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
]
TASK_TYPES = {
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
}
VISIBLE_MARKERS = {
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
}
AUDIO_MARKERS = {"fully_copy", "partially_copy", "reference", "weak_reference"}

VIEW_REQUEST_ITEM = {
    "global_id": {"type": "string"},
    "view_types": {"type": "array", "items": {"type": "string"}},
    "prominence": {"type": "string", "enum": ["primary", "secondary", "background"]},
    "reason": {"type": "string"},
}
SCENE_ITEM = {
    "title": {"type": "string"},
    "source_excerpt": {"type": "string"},
    "visual_event": {"type": "string"},
    "location_global_id": {"type": "string"},
    "visible_entity_ids": {"type": "array", "items": {"type": "string"}},
    "speaking_entity_ids": {"type": "array", "items": {"type": "string"}},
    "reference_view_requests": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": VIEW_REQUEST_ITEM,
            "required": list(VIEW_REQUEST_ITEM),
            "additionalProperties": False,
        },
    },
    "dialogue_present": {"type": "boolean"},
    "adaptation_notes": {"type": "string"},
}
SCENE_SCHEMA = {
    "name": "chapter_video_scenes_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": SCENE_ITEM,
                    "required": list(SCENE_ITEM),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scenes"],
        "additionalProperties": False,
    },
}
PROMPT_SCHEMA = {
    "name": "minimax_h3_prompt_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"prompt_text": {"type": "string"}},
        "required": ["prompt_text"],
        "additionalProperties": False,
    },
}

H3_RULES = r"""
Write ONE MiniMax H3 full-reference/reference-to-video prompt.

The prompt MUST contain exactly these six top-level sections, in this order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Language:
- write all six sections in English;
- preserve source language only for dialogue/lyrics inside <d> and text visibly
  present in the scene.

Reference semantics:
- <Subject N> is reusable visible content abstracted from reference assets.
- ONE <Subject N> MAY be defined by MULTIPLE reference assets. When that happens,
  combine the sources in ONE subject definition and explain what each asset provides.
- <Picture N> is an input reference image. In this workflow pictures are identity,
  appearance, environment, or object references. Therefore cite them inside the
  corresponding <Subject N> definition; do NOT create standalone Picture definition
  lines unless the caller explicitly marks a picture as a concrete keyframe anchor.
- <Video N> is reserved for source-video editing/continuation/temporal structure.
- <Audio N> is an audio signal used for copying/reference, e.g. voice timbre.
- Once assigned, a label keeps the same meaning throughout the prompt.

subject_definitions:
- define each supplied <Subject N> exactly once;
- if a subject has several pictures, cite ALL of them in that same subject definition;
- explicitly say what each view contributes, e.g. face identity, full-body proportions,
  rear silhouette, location layout, secondary angle, or detail structure;
- define supplied <Audio N> items and bind voice references to the corresponding
  <Subject N> (Sx) when that subject actually speaks.

summary:
- one short English paragraph;
- start with applicable task types chosen only from:
  reference generation, keyframe completion, video editing, video continuation,
  audio reuse, audio reference;
- combine multiple types with " + " and do not duplicate a type;
- do not introduce labels not already defined.

retention_analysis:
- one line per tracked Subject and Audio reference in this workflow;
- because Picture inputs here only define Subjects, do not add separate Picture
  retention lines;
- Subject marker must be one of fully_preserved, partially_preserved,
  attribute_transfer, weak_reference;
- Audio marker must be one of fully_copy, partially_copy, reference, weak_reference;
- never put speaker IDs such as (S1) here.

detailed_description:
- primary playback-order description;
- generation tasks normally target 350-500 English words;
- write 1-2 visual/cinematic style sentences before [Shot 1];
- [Shot 1] has NO timestamp;
- every later shot starts exactly: [Shot N] At MM:SS.mmm, 
- timestamps are cut times and must fit the requested duration;
- describe composition/framing, referenced visible traits and positions,
  environment/lighting, actions/state changes, camera movement, current ambience/SFX,
  and where the reference applies;
- at the first clear appearance of a Subject, state its visible referenced traits,
  frame position and action;
- use the same Subject label even when a different reference view of that Subject is
  especially relevant to a later angle. Do NOT create a new Subject just because the
  camera sees the same character from the back or side.

Speech:
- assign stable (S1), (S2), ... in order of first actual vocal event;
- a referenced visible speaker is written <Subject N> (Sx);
- dialogue/lyrics must be inside <d>[Language] Exact words.</d>;
- preserve source dialogue faithfully and never invent dialogue;
- internal thought is not audible speech unless the source explicitly makes it
  narration/speech;
- off-screen speech keeps its speaker ID;
- use <scenetrans>/<cutoff> only when truly needed.

Sound:
- overall_soundscape summarizes ambience and physical/diegetic sounds;
- non_diegetic_music describes audience-only score, including instrumentation,
  tempo, mood and dynamics, or N/A;
- do not repeat complete dialogue in sound summaries.

Adaptation discipline:
- stay faithful to the supplied excerpt;
- do not invent plot events, named characters, lore, dialogue or distinctive props;
- compress for duration without changing the core event;
- favor visually legible action over exposition.
""".strip()


@dataclass
class ViewRequest:
    global_id: str
    view_types: list[str]
    prominence: str
    reason: str


@dataclass
class Scene:
    title: str
    source_excerpt: str
    visual_event: str
    location_global_id: str
    visible_entity_ids: list[str]
    speaking_entity_ids: list[str]
    reference_view_requests: list[ViewRequest]
    dialogue_present: bool
    adaptation_notes: str


@dataclass
class Validation:
    ok: bool
    errors: list[str]
    word_count: int


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value)]


def slug(value: str, max_len: int = 64) -> str:
    value = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("._")
    return (value[:max_len] or "scene").rstrip("._")


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
        raise ValueError(f"Unsupported input type: {path.suffix}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if len(text) < 100:
        raise ValueError("Chapter is empty or too short after extraction.")
    return text


def split_chunks(text: str, max_chars: int, overlap_paragraphs: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text]
    out: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        add = len(para) + (2 if current else 0)
        if current and current_len + add > max_chars:
            out.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(x) for x in current) + max(0, len(current) - 1) * 2
        current.append(para)
        current_len += add
    if current:
        out.append("\n\n".join(current))
    return out


def make_client(base_url: str, api_key: str, *, http_client=None) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=300.0, max_retries=2, http_client=http_client)


def entity_index(refs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["global_id"]: e for e in refs.get("entities", [])}


def picture_assets_by_entity(refs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for asset in refs.get("picture_assets", []):
        out.setdefault(asset["linked_global_id"], []).append(asset)
    return out


def chapter_catalog(refs: dict[str, Any], chapter_id: str) -> list[dict[str, Any]]:
    entities = entity_index(refs)
    assets = picture_assets_by_entity(refs)
    gids = set(refs.get("chapter_entity_map", {}).get(chapter_id, {}).values())
    if not gids:
        gids = {e["global_id"] for e in entities.values() if chapter_id in e.get("chapters_seen", [])}

    out = []
    for gid in sorted(gids, key=natural_key):
        e = entities.get(gid)
        if not e:
            continue
        variation = next((x for x in e.get("chapter_variations", []) if x.get("chapter_id") == chapter_id), {})
        available = []
        for a in assets.get(gid, []):
            if a.get("variant") in {"base", chapter_id}:
                available.append(
                    {
                        "view_type": a.get("view_type", ""),
                        "variant": a.get("variant", "base"),
                        "asset_id": a.get("asset_id", ""),
                    }
                )
        out.append(
            {
                "global_id": gid,
                "entity_type": e["entity_type"],
                "canonical_name": e["canonical_name"],
                "aliases": e.get("aliases", []),
                "stable_visual_description": e.get("stable_visual_description", ""),
                "chapter_visual_state": variation.get("visual_state", ""),
                "distinguishing_features": e.get("distinguishing_features", []),
                "voice_description": e.get("voice_description", ""),
                "available_picture_views": available,
            }
        )
    return out


PLAN_SYSTEM = """
Select short, visually coherent scenes from a novel passage for video adaptation.
Stay faithful to the source. Do not invent dialogue or plot events.

Use global IDs only from the supplied chapter entity catalog.
- visible_entity_ids: catalogued characters/locations/objects actually visible.
- speaking_entity_ids: catalogued characters who actually speak in the excerpt.
- location_global_id: catalogued main location or empty string.
- reference_view_requests: for important visible entities, select ONLY view_type
  values actually listed in available_picture_views. Request views that materially
  help the planned camera angles.

View-selection examples:
- close facial dialogue: face_front and/or three_quarter;
- full-body action: full_body_front + three_quarter;
- walking away/rear framing: back_view plus a face/full-body identity view if useful;
- profile framing: profile plus face_front if available;
- location establishing shot: wide_establishing;
- reverse/alternate camera angle: secondary_angle or reverse_angle;
- insert on distinctive architecture/prop: key_detail/detail_closeup.

Do not request every available image by default. The later binding stage has a finite
reference budget. Mark the main character/location as primary, supporting entities as
secondary, and incidental entities as background.

source_excerpt must contain enough exact prose to adapt the moment and retain exact
source dialogue wording when dialogue matters. Prefer action or strong visual/emotional
beats with a legible beginning/change/result. Avoid pure exposition and duplicate scenes
caused by chunk overlap.
""".strip()


def plan_scenes(
    client: OpenAI,
    model: str,
    chapter_id: str,
    chunk: str,
    index: int,
    total: int,
    catalog: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Scene]:
    user = f"""Chapter: {chapter_id}
Chunk: {index}/{total}
Target clip duration: approximately {args.duration:g} seconds
Maximum scenes from this chunk: {args.scenes_per_chunk}

CHAPTER ENTITY CATALOG:
{json.dumps(catalog, ensure_ascii=False, indent=2)}

--- BEGIN NOVEL PASSAGE ---
{chunk}
--- END NOVEL PASSAGE ---"""
    data = chat_json(client, model, PLAN_SYSTEM, user, SCENE_SCHEMA, 0.24, 6500)
    valid_ids = {e["global_id"] for e in catalog}
    view_lookup = {
        e["global_id"]: {x["view_type"] for x in e.get("available_picture_views", [])}
        for e in catalog
    }

    scenes: list[Scene] = []
    for raw in data.get("scenes", [])[:args.scenes_per_chunk]:
        visible = [x for x in raw["visible_entity_ids"] if x in valid_ids]
        speaking = [x for x in raw["speaking_entity_ids"] if x in valid_ids]
        loc = raw["location_global_id"] if raw["location_global_id"] in valid_ids else ""
        if loc and loc not in visible:
            visible.insert(0, loc)

        requests: list[ViewRequest] = []
        seen_req: set[str] = set()
        for req in raw.get("reference_view_requests", []):
            gid = req.get("global_id", "")
            if gid not in valid_ids or gid in seen_req:
                continue
            valid_views = [v for v in req.get("view_types", []) if v in view_lookup.get(gid, set())]
            if not valid_views:
                continue
            requests.append(
                ViewRequest(
                    global_id=gid,
                    view_types=list(dict.fromkeys(valid_views)),
                    prominence=req.get("prominence", "secondary"),
                    reason=req.get("reason", "").strip(),
                )
            )
            seen_req.add(gid)

        scenes.append(
            Scene(
                title=raw["title"].strip(),
                source_excerpt=raw["source_excerpt"].strip(),
                visual_event=raw["visual_event"].strip(),
                location_global_id=loc,
                visible_entity_ids=visible,
                speaking_entity_ids=speaking,
                reference_view_requests=requests,
                dialogue_present=bool(raw["dialogue_present"]),
                adaptation_notes=raw["adaptation_notes"].strip(),
            )
        )
    return scenes


def fingerprint(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def dedupe_scenes(scenes: list[Scene], threshold: float = 0.72) -> list[Scene]:
    kept: list[Scene] = []
    fps: list[set[str]] = []
    for scene in scenes:
        fp = fingerprint(scene.source_excerpt)
        if any(jaccard(fp, old) >= threshold for old in fps):
            continue
        kept.append(scene)
        fps.append(fp)
    return kept


def default_view_order(entity_type: str) -> list[str]:
    if entity_type == "character":
        return ["face_front", "three_quarter", "full_body_front", "back_view", "profile", "expression_closeup", "costume_detail"]
    if entity_type == "location":
        return ["wide_establishing", "secondary_angle", "reverse_angle", "key_detail", "interior_zone", "exterior_approach"]
    return ["hero_three_quarter", "detail_closeup", "side_profile", "scale_context"]


def best_asset_for_view(
    assets: list[dict[str, Any]],
    view_type: str,
    chapter_id: str,
) -> dict[str, Any] | None:
    exact = [a for a in assets if a.get("view_type") == view_type]
    if not exact:
        return None
    return next((a for a in exact if a.get("variant") == chapter_id), None) or next(
        (a for a in exact if a.get("variant") == "base"), exact[0]
    )


def available_assets_for_entity(
    refs: dict[str, Any],
    global_id: str,
    chapter_id: str,
) -> list[dict[str, Any]]:
    assets = [a for a in refs.get("picture_assets", []) if a.get("linked_global_id") == global_id]
    # Keep only base or chapter-specific variants; if both exist for same view, chapter variant wins later.
    return [a for a in assets if a.get("variant") in {"base", chapter_id}]


def audio_asset_for(refs: dict[str, Any], global_id: str) -> dict[str, Any] | None:
    return next((a for a in refs.get("audio_assets", []) if a.get("linked_global_id") == global_id), None)


def request_map(scene: Scene) -> dict[str, ViewRequest]:
    return {r.global_id: r for r in scene.reference_view_requests}


def prominence_score(value: str) -> int:
    return {"primary": 0, "secondary": 1, "background": 2}.get(value, 1)


def build_bindings(
    refs: dict[str, Any],
    scene: Scene,
    chapter_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    entities = entity_index(refs)
    requests = request_map(scene)

    ordered_visible = list(dict.fromkeys(
        ([scene.location_global_id] if scene.location_global_id else []) + scene.visible_entity_ids
    ))

    candidates: list[dict[str, Any]] = []
    unreferenced: list[dict[str, Any]] = []
    for order_index, gid in enumerate(ordered_visible):
        entity = entities.get(gid)
        if not entity:
            continue
        available = available_assets_for_entity(refs, gid, chapter_id)
        if not available:
            unreferenced.append(
                {
                    "global_id": gid,
                    "entity_type": entity["entity_type"],
                    "canonical_name": entity["canonical_name"],
                    "visual_description": entity.get("stable_visual_description", ""),
                }
            )
            continue

        req = requests.get(gid)
        requested_views = req.view_types if req else []
        if not requested_views:
            available_view_set = {a.get("view_type") for a in available}
            requested_views = [v for v in default_view_order(entity["entity_type"]) if v in available_view_set]

        selected_assets: list[dict[str, Any]] = []
        seen_asset_ids: set[str] = set()
        for view in requested_views:
            asset = best_asset_for_view(available, view, chapter_id)
            if asset and asset["asset_id"] not in seen_asset_ids:
                selected_assets.append(asset)
                seen_asset_ids.add(asset["asset_id"])

        # Fallback if requested views were missing/stale.
        if not selected_assets:
            ordered_available = sorted(
                available,
                key=lambda a: (
                    0 if a.get("variant") == chapter_id else 1,
                    default_view_order(entity["entity_type"]).index(a.get("view_type"))
                    if a.get("view_type") in default_view_order(entity["entity_type"])
                    else 999,
                ),
            )
            selected_assets = ordered_available[:1]

        candidates.append(
            {
                "global_id": gid,
                "entity": entity,
                "prominence": req.prominence if req else ("primary" if gid == scene.location_global_id else "secondary"),
                "reason": req.reason if req else "Default reference coverage for a visible entity.",
                "order_index": order_index,
                "assets": selected_assets[: max(1, args.max_pictures_per_subject)],
            }
        )

    candidates.sort(key=lambda x: (prominence_score(x["prominence"]), x["order_index"]))

    # Allocate at least one image to as many important entities as the total budget permits.
    allocated: dict[str, list[dict[str, Any]]] = {}
    picture_count = 0
    for c in candidates:
        if picture_count >= args.max_pictures:
            break
        if c["assets"]:
            allocated[c["global_id"]] = [c["assets"][0]]
            picture_count += 1

    # Round-robin additional views, prioritizing primary then secondary subjects.
    level = 1
    while picture_count < args.max_pictures:
        added = False
        for c in candidates:
            gid = c["global_id"]
            if gid not in allocated:
                continue
            if level < len(c["assets"]) and len(allocated[gid]) < args.max_pictures_per_subject:
                allocated[gid].append(c["assets"][level])
                picture_count += 1
                added = True
                if picture_count >= args.max_pictures:
                    break
        if not added:
            break
        level += 1

    subjects: list[dict[str, Any]] = []
    picture_input_order: list[dict[str, Any]] = []
    picture_index = 0
    for subject_index, c in enumerate([x for x in candidates if x["global_id"] in allocated], start=1):
        entity = c["entity"]
        subject_label = f"<Subject {subject_index}>"
        bound_pictures = []
        for asset in allocated[c["global_id"]]:
            picture_index += 1
            picture_label = f"<Picture {picture_index}>"
            bound = {
                "h3_picture_label": picture_label,
                "asset_id": asset["asset_id"],
                "suggested_filename": asset.get("suggested_filename", ""),
                "view_type": asset.get("view_type", ""),
                "variant": asset.get("variant", "base"),
                "description": asset.get("description", ""),
            }
            bound_pictures.append(bound)
            picture_input_order.append(
                {
                    "h3_picture_label": picture_label,
                    "asset_id": asset["asset_id"],
                    "suggested_filename": asset.get("suggested_filename", ""),
                    "subject_label": subject_label,
                    "global_id": c["global_id"],
                    "canonical_name": entity["canonical_name"],
                    "view_type": asset.get("view_type", ""),
                    "variant": asset.get("variant", "base"),
                }
            )
        subjects.append(
            {
                "h3_subject_label": subject_label,
                "global_id": c["global_id"],
                "entity_type": entity["entity_type"],
                "canonical_name": entity["canonical_name"],
                "prominence": c["prominence"],
                "selection_reason": c["reason"],
                "stable_visual_description": entity.get("stable_visual_description", ""),
                "distinguishing_features": entity.get("distinguishing_features", []),
                "pictures": bound_pictures,
            }
        )

    subject_by_gid = {s["global_id"]: s["h3_subject_label"] for s in subjects}
    audio: list[dict[str, Any]] = []
    audio_input_order: list[dict[str, Any]] = []
    for gid in scene.speaking_entity_ids:
        if len(audio) >= args.max_audio:
            break
        asset = audio_asset_for(refs, gid)
        if not asset:
            continue
        entity = entities.get(gid, {})
        label = f"<Audio {len(audio) + 1}>"
        item = {
            "h3_audio_label": label,
            "asset_id": asset["asset_id"],
            "suggested_filename": asset.get("suggested_filename", ""),
            "global_id": gid,
            "canonical_name": entity.get("canonical_name", asset.get("canonical_name", "")),
            "linked_subject_label": subject_by_gid.get(gid, ""),
            "reference_description": asset.get("description", ""),
            "voice_description": entity.get("voice_description", ""),
        }
        audio.append(item)
        audio_input_order.append(
            {
                "h3_audio_label": label,
                "asset_id": asset["asset_id"],
                "suggested_filename": asset.get("suggested_filename", ""),
                "global_id": gid,
                "canonical_name": item["canonical_name"],
            }
        )

    allocated_gids = set(allocated)
    for c in candidates:
        if c["global_id"] not in allocated_gids:
            entity = c["entity"]
            unreferenced.append(
                {
                    "global_id": c["global_id"],
                    "entity_type": entity["entity_type"],
                    "canonical_name": entity["canonical_name"],
                    "visual_description": entity.get("stable_visual_description", ""),
                    "reason": "Per-clip picture budget exhausted.",
                }
            )

    return {
        "subjects": subjects,
        "audio": audio,
        "picture_input_order": picture_input_order,
        "audio_input_order": audio_input_order,
        "unreferenced_visible_entities": unreferenced,
        "binding_note": (
            "Several picture_input_order entries may point to the same Subject. "
            "Attach pictures in <Picture N> order and audio in <Audio N> order."
        ),
    }


def generate_prompt(
    client: OpenAI,
    model: str,
    scene: Scene,
    bindings: dict[str, Any],
    duration: float,
    args: argparse.Namespace,
) -> str:
    has_audio = bool(bindings["audio"])
    expected_prefix = "reference generation" + (" + audio reference" if has_audio else "")
    system = H3_RULES + """

The caller supplies an exact per-clip binding table. Obey it exactly:
- do not renumber Subject, Picture or Audio labels;
- do not invent extra reference labels;
- every subject definition must cite ALL pictures assigned to that subject;
- multiple pictures assigned to one subject are complementary views of the SAME
  entity, not separate subjects;
- state what each picture contributes to that one subject;
- unreferenced visible entities may appear as ordinary prose without Subject labels;
- do not claim an audio reference exists unless it is in the audio bindings.
"""
    user = f"""TARGET DURATION: {duration:g} seconds.
All cut timestamps must be <= {duration:.3f} seconds.
EXPECTED SUMMARY PREFIX: [{expected_prefix}]

SCENE
Title: {scene.title}
Visual event: {scene.visual_event}
Dialogue present in source: {scene.dialogue_present}
Adaptation notes: {scene.adaptation_notes}

EXACT PER-CLIP BINDINGS
{json.dumps(bindings, ensure_ascii=False, indent=2)}

--- BEGIN SOURCE EXCERPT ---
{scene.source_excerpt}
--- END SOURCE EXCERPT ---

Normally use 2-4 shots, but fit the actual action. Preserve source dialogue exactly
if used. Do not vocalize internal thoughts. detailed_description normally targets
350-500 English words; dialogue-heavy material prioritizes fitting the actual spoken
timeline. Return only the six-section H3 prompt inside prompt_text, with no Markdown.
"""
    data = chat_json(client, model, system, user, PROMPT_SCHEMA, args.temperature, args.max_tokens)
    return re.sub(r"<think>.*?</think>", "", data["prompt_text"], flags=re.S | re.I).strip()


def section_body(prompt: str, name: str) -> str:
    idx = SECTIONS.index(name)
    start_match = re.search(rf"(?mi)^\s*{re.escape(name)}\s*:\s*$", prompt)
    if not start_match:
        return ""
    start = start_match.end()
    if idx == len(SECTIONS) - 1:
        return prompt[start:].strip()
    next_match = re.search(rf"(?mi)^\s*{re.escape(SECTIONS[idx + 1])}\s*:\s*$", prompt[start:])
    end = start + next_match.start() if next_match else len(prompt)
    return prompt[start:end].strip()


def timestamp_seconds(mm: str, ss: str, mmm: str) -> float:
    return int(mm) * 60 + int(ss) + int(mmm) / 1000.0


def validate_prompt(prompt: str, bindings: dict[str, Any], duration: float) -> Validation:
    errors: list[str] = []
    positions: list[int] = []
    for sec in SECTIONS:
        matches = list(re.finditer(rf"(?mi)^\s*{re.escape(sec)}\s*:\s*$", prompt))
        if len(matches) != 1:
            errors.append(f"Section {sec!r} must appear exactly once; found {len(matches)}.")
        positions.append(matches[0].start() if matches else -1)
    if all(p >= 0 for p in positions) and positions != sorted(positions):
        errors.append("The six sections are not in the required order.")

    summary = section_body(prompt, "summary")
    prefix = re.match(r"^\[([^\]]+)\]", summary)
    if not prefix:
        errors.append("summary must begin with a bracketed task-type prefix.")
    else:
        parts = [x.strip() for x in prefix.group(1).split("+")]
        if len(parts) != len(set(parts)):
            errors.append("summary has a duplicate task type.")
        if any(x not in TASK_TYPES for x in parts):
            errors.append("summary contains an invalid task type.")
        expected = ["reference generation"] + (["audio reference"] if bindings["audio"] else [])
        if parts != expected:
            errors.append(f"summary task prefix should be [{' + '.join(expected)}].")

    detailed = section_body(prompt, "detailed_description")
    word_count = len(re.findall(r"\b[\w'-]+\b", detailed))
    nums = [int(x) for x in re.findall(r"\[Shot\s+(\d+)\]", detailed)]
    if not nums:
        errors.append("No [Shot N] markers found.")
    elif nums != list(range(1, len(nums) + 1)):
        errors.append(f"Shot numbering is not sequential: {nums}.")
    if re.search(r"\[Shot 1\]\s+At\s+", detailed):
        errors.append("[Shot 1] must not have a timestamp.")
    later = re.findall(r"\[Shot\s+(\d+)\]\s+At\s+(\d{2}):(\d{2})\.(\d{3}),", detailed)
    if len(later) != max(0, len(nums) - 1):
        errors.append("Every shot after Shot 1 must begin '[Shot N] At MM:SS.mmm, '.")
    previous = -1.0
    for n, mm, ss, mmm in later:
        t = timestamp_seconds(mm, ss, mmm)
        if t <= previous:
            errors.append("Shot cut timestamps must strictly increase.")
        if t > duration + 1e-6:
            errors.append(f"Shot {n} cut {t:.3f}s exceeds target duration {duration:.3f}s.")
        previous = t
    if word_count < 330:
        errors.append(f"detailed_description is short ({word_count} words; normal target 350-500).")
    elif word_count > 540:
        errors.append(f"detailed_description is long ({word_count} words; normal target 350-500).")

    for match in re.finditer(r"<d>(.*?)</d>", prompt, flags=re.S):
        body = match.group(1).strip()
        if not re.match(r"^\[[^\]]+\]\s+.+", body, flags=re.S):
            errors.append("Each <d> dialogue span must begin with [Language].")
            break
    if "```" in prompt:
        errors.append("Prompt contains a Markdown fence.")

    defs = section_body(prompt, "subject_definitions")
    retention = section_body(prompt, "retention_analysis")

    expected_subjects = {s["h3_subject_label"] for s in bindings["subjects"]}
    expected_pictures = {
        p["h3_picture_label"]
        for s in bindings["subjects"]
        for p in s["pictures"]
    }
    expected_audio = {a["h3_audio_label"] for a in bindings["audio"]}

    used_subjects = set(re.findall(r"<Subject\s+\d+>", prompt))
    used_pictures = set(re.findall(r"<Picture\s+\d+>", prompt))
    used_audio = set(re.findall(r"<Audio\s+\d+>", prompt))

    if used_subjects - expected_subjects:
        errors.append("Unbound Subject labels: " + ", ".join(sorted(used_subjects - expected_subjects)))
    if used_pictures - expected_pictures:
        errors.append("Unbound Picture labels: " + ", ".join(sorted(used_pictures - expected_pictures)))
    if used_audio - expected_audio:
        errors.append("Unbound Audio labels: " + ", ".join(sorted(used_audio - expected_audio)))

    for subject in bindings["subjects"]:
        label = subject["h3_subject_label"]
        line_match = re.search(rf"(?mi)^\s*{re.escape(label)}(?:\s|$).*$", defs)
        if not line_match:
            errors.append(f"{label} is missing from subject_definitions.")
            continue
        line = line_match.group(0)
        for picture in subject["pictures"]:
            plabel = picture["h3_picture_label"]
            if plabel not in line:
                errors.append(f"{label} definition must cite assigned {plabel}.")

    for audio in bindings["audio"]:
        if audio["h3_audio_label"] not in defs:
            errors.append(f"{audio['h3_audio_label']} is missing from subject_definitions.")

    # In this workflow pictures only define subjects; standalone Picture analysis is unwanted.
    if re.search(r"(?m)^\s*<Picture\s+\d+>.*?:\s*(?:fully_preserved|partially_preserved|attribute_transfer|weak_reference)\s*-", retention):
        errors.append("retention_analysis should track the Subject, not separate Picture lines, for identity/view references.")

    if re.search(r"\(S\d+\)", retention):
        errors.append("retention_analysis must not contain speaker IDs.")

    for line in retention.splitlines():
        stripped = line.strip()
        match = re.match(
            r"^(<Subject\s+\d+>|<Video\s+\d+>|<Audio\s+\d+>).*?:\s*([a-z_]+)\s*-",
            stripped,
        )
        if match:
            label, marker = match.groups()
            allowed = AUDIO_MARKERS if label.startswith("<Audio") else VISIBLE_MARKERS
            if marker not in allowed:
                errors.append(f"Invalid retention marker {marker!r} for {label}.")

    return Validation(ok=not errors, errors=errors, word_count=word_count)


def repair_prompt(
    client: OpenAI,
    model: str,
    prompt: str,
    scene: Scene,
    bindings: dict[str, Any],
    validation: Validation,
    duration: float,
    args: argparse.Namespace,
) -> str:
    system = H3_RULES + """

You are repairing a malformed prompt. Fix the listed validation errors while
preserving the source event and exact binding table. Several Picture labels may
belong to the same Subject; never split one entity into multiple Subjects merely
because it has several views. Return only prompt_text JSON.
"""
    user = f"""TARGET DURATION: {duration:g}s

VALIDATION ERRORS:
{chr(10).join('- ' + x for x in validation.errors)}

BINDINGS:
{json.dumps(bindings, ensure_ascii=False, indent=2)}

SOURCE:
--- BEGIN ---
{scene.source_excerpt}
--- END ---

CURRENT PROMPT:
--- BEGIN ---
{prompt}
--- END ---"""
    data = chat_json(client, model, system, user, PROMPT_SCHEMA, min(args.temperature, 0.16), args.max_tokens)
    return re.sub(r"<think>.*?</think>", "", data["prompt_text"], flags=re.S | re.I).strip()


def save_scene(
    chapter_dir: Path,
    index: int,
    scene: Scene,
    bindings: dict[str, Any],
    prompt: str,
    validation: Validation,
) -> dict[str, Any]:
    chapter_dir = chapter_dir.resolve()
    stem = f"scene_{index:03d}_{slug(scene.title)}"
    prompt_path = confined_path(chapter_dir / f"{stem}_prompt.txt", chapter_dir)
    assets_path = confined_path(chapter_dir / f"{stem}_assets.json", chapter_dir)
    source_path = confined_path(chapter_dir / f"{stem}_source.txt", chapter_dir)
    prompt_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")
    assets_path.write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_path.write_text(
        f"TITLE: {scene.title}\n\nVISUAL EVENT:\n{scene.visual_event}\n\n"
        f"ADAPTATION NOTES:\n{scene.adaptation_notes}\n\nSOURCE EXCERPT:\n{scene.source_excerpt}\n",
        encoding="utf-8",
    )
    return {
        "index": index,
        "title": scene.title,
        "prompt_file": prompt_path.name,
        "assets_file": assets_path.name,
        "source_file": source_path.name,
        "valid": validation.ok,
        "validation_errors": validation.errors,
        "detailed_description_words": validation.word_count,
        "picture_count": len(bindings["picture_input_order"]),
        "subject_count": len(bindings["subjects"]),
        "audio_count": len(bindings["audio"]),
    }


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    return {
        "title": scene.title,
        "source_excerpt": scene.source_excerpt,
        "visual_event": scene.visual_event,
        "location_global_id": scene.location_global_id,
        "visible_entity_ids": scene.visible_entity_ids,
        "speaking_entity_ids": scene.speaking_entity_ids,
        "reference_view_requests": [r.__dict__ for r in scene.reference_view_requests],
        "dialogue_present": scene.dialogue_present,
        "adaptation_notes": scene.adaptation_notes,
    }


def scene_from_dict(data: dict[str, Any]) -> Scene:
    return Scene(
        title=data["title"],
        source_excerpt=data["source_excerpt"],
        visual_event=data["visual_event"],
        location_global_id=data["location_global_id"],
        visible_entity_ids=list(data["visible_entity_ids"]),
        speaking_entity_ids=list(data["speaking_entity_ids"]),
        reference_view_requests=[ViewRequest(**x) for x in data.get("reference_view_requests", [])],
        dialogue_present=bool(data["dialogue_present"]),
        adaptation_notes=data["adaptation_notes"],
    )


def process_chapter(
    path: Path,
    refs: dict[str, Any],
    client: OpenAI,
    model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    chapter_id = slug(path.stem)
    text = read_chapter(path)
    catalog = chapter_catalog(refs, chapter_id)
    if not catalog:
        print(f"WARNING: no consolidated entity mapping for {chapter_id}.", file=sys.stderr)

    chunks = split_chunks(text, max(3000, args.chunk_chars), max(0, args.overlap_paragraphs))
    chapter_dir = confined_path(chapter_id, args.out_dir)
    chapter_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = confined_path(chapter_dir / ".cache", chapter_dir)
    cache_dir.mkdir(exist_ok=True)

    print(f"{path.name}: {len(chunks)} planning chunk(s)")
    scenes: list[Scene] = []
    for i, chunk in enumerate(chunks, start=1):
        cache_key = hashlib.sha256(
            (
                REFERENCE_SCHEMA + "\n" + model + "\nthinking=" + str(json_backend.THINKING_ENABLED) + "\nchat_backend=" + json_backend.CHAT_BACKEND + "\n" + str(args.duration) + "\n" +
                refs.get("source_digest", "") + "\n" +
                json.dumps(catalog, ensure_ascii=False, sort_keys=True) + "\n" + chunk
            ).encode()
        ).hexdigest()
        cache_path = confined_path(cache_dir / f"plan_{i:03d}.json", chapter_dir)
        chunk_scenes = None
        if cache_path.exists() and not args.force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == cache_key:
                    chunk_scenes = [scene_from_dict(x) for x in cached["scenes"]]
            except Exception:
                pass
        if chunk_scenes is None:
            print(f"  planning chunk {i}/{len(chunks)}")
            chunk_scenes = plan_scenes(client, model, chapter_id, chunk, i, len(chunks), catalog, args)
            cache_path.write_text(
                json.dumps({"cache_key": cache_key, "scenes": [scene_to_dict(x) for x in chunk_scenes]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if args.delay:
                time.sleep(args.delay)
        else:
            print(f"  planning chunk {i}/{len(chunks)}: cached")
        scenes.extend(chunk_scenes)

    scenes = dedupe_scenes(scenes)
    if args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]
    print(f"  selected {len(scenes)} scene(s)")

    entries: list[dict[str, Any]] = []
    for i, scene in enumerate(scenes, start=1):
        print(f"  [{i}/{len(scenes)}] {scene.title}")
        bindings = build_bindings(refs, scene, chapter_id, args)
        if not bindings["subjects"] and not bindings["audio"]:
            reason = "No reference asset available for this full-reference scene."
            entries.append({"index": i, "title": scene.title, "skipped": True, "reason": reason})
            print(f"    skipped: {reason}")
            continue

        prompt_key = hashlib.sha256(
            (
                REFERENCE_SCHEMA + "\n" + model + "\nthinking=" + str(json_backend.THINKING_ENABLED) + "\nchat_backend=" + json_backend.CHAT_BACKEND + "\n" + str(args.duration) + "\n" +
                scene.source_excerpt + "\n" + json.dumps(bindings, ensure_ascii=False, sort_keys=True)
            ).encode()
        ).hexdigest()
        prompt_cache = confined_path(cache_dir / f"prompt_{i:03d}.json", chapter_dir)
        prompt = None
        if prompt_cache.exists() and not args.force:
            try:
                cached = json.loads(prompt_cache.read_text(encoding="utf-8"))
                if cached.get("cache_key") == prompt_key:
                    prompt = cached["prompt"]
            except Exception:
                pass

        if prompt is None:
            prompt = generate_prompt(client, model, scene, bindings, args.duration, args)

        validation = validate_prompt(prompt, bindings, args.duration)
        repairs = 0
        while not validation.ok and repairs < args.repair_attempts:
            repairs += 1
            print(f"    repair {repairs}/{args.repair_attempts}: {'; '.join(validation.errors[:3])}")
            prompt = repair_prompt(client, model, prompt, scene, bindings, validation, args.duration, args)
            validation = validate_prompt(prompt, bindings, args.duration)

        prompt_cache.write_text(
            json.dumps({"cache_key": prompt_key, "prompt": prompt}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        entry = save_scene(chapter_dir, i, scene, bindings, prompt, validation)
        entry["repair_attempts_used"] = repairs
        entries.append(entry)
        if validation.ok:
            print(
                f"    OK — {entry['subject_count']} subjects, {entry['picture_count']} pictures, "
                f"{entry['audio_count']} audio, {validation.word_count} detailed words"
            )
        else:
            print("    saved with warnings: " + "; ".join(validation.errors[:3]))
        if args.delay:
            time.sleep(args.delay)

    saved = [x for x in entries if x.get("prompt_file")]
    if saved:
        blocks = []
        for x in saved:
            p = confined_path(x["prompt_file"], chapter_dir).read_text(encoding="utf-8").rstrip()
            blocks.append(f"========== SCENE {x['index']:03d}: {x['title']} ==========\n\n{p}")
        confined_path(chapter_dir / "all_prompts.txt", chapter_dir).write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")

    manifest = {
        "chapter_id": chapter_id,
        "source_file": str(path),
        "model": model,
        "duration_seconds": args.duration,
        "scene_count": len(scenes),
        "saved_prompt_count": len(saved),
        "outputs": entries,
    }
    confined_path(chapter_dir / "manifest.json", chapter_dir).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


