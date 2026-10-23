"""Chapter-wide spatial staging from consolidated cinematic timelines.

All sequences in the chapter are supplied to every planning call, allowing future
facts to constrain earlier placement without pulling future actions backward.
"""
from __future__ import annotations

import json
from typing import Any

from . import progress, lmstudio_pipeline, util
from .lmstudio_json import chat_json
from .run_output import stage_output

_ENTITY = {
    "entity_id": {"type": "string"},
    "kind": {"type": "string", "enum": ["character", "object", "location"]},
    "world_position": {"type": "string"},
    "relative_to": {"type": "string"},
    "orientation": {"type": "string"},
    "state": {"type": "string"},
    "screen_position": {"type": "string"},
    "evidence": {"type": "string", "enum": ["explicit", "inferred", "unknown"]},
    "source_sequence": {"type": "integer"},
}
_FIELDS = {
    "location_id": {"type": "string"},
    "location_layout": {"type": "string"},
    "camera_position": {"type": "string"},
    "camera_direction": {"type": "string"},
    "camera_movement": {"type": "string"},
    "camera_axis": {"type": "string"},
    "initial_entities": {"type": "array", "items": {"type": "object", "properties": _ENTITY, "required": list(_ENTITY), "additionalProperties": False}},
    "final_entities": {"type": "array", "items": {"type": "object", "properties": _ENTITY, "required": list(_ENTITY), "additionalProperties": False}},
    "justified_changes": {"type": "array", "items": {"type": "string"}},
    "warnings": {"type": "array", "items": {"type": "string"}},
}
_SCHEMA = {
    "name": "chapter_spatial_sequence_v2", "strict": True,
    "schema": {"type": "object", "properties": _FIELDS,
               "required": list(_FIELDS), "additionalProperties": False},
}
_SYSTEM = """You are a screenplay spatial-continuity supervisor.
Study the ENTIRE ordered chapter timeline, not just the adjacent sequence.
Return staging ONLY for target_sequence, considering explicit facts anywhere
in the chapter, prior established positions, and future constraints.
Look forward AND backward: if a later sequence states a character is near a
fixed prop and no intervening movement occurs, place them there from the
earlier sequence. NEVER import later actions into earlier sequences.
Distinguish world-space (fixed in the setting), person-relative and
screen-space (dependent on camera). Camera movements or cuts can change screen
placement without moving an actor. Preserve immutable location geometry,
object anchoring, possession, facing and interactions; do not teleport actors.
Return initial and final states including important offscreen persistent
entities. A change requires an event in the target sequence; record its cause
in justified_changes. When placement is not supportable, use 'unknown', not
fabricated coordinates. Use consolidated global IDs where available; never
pretend an ambiguous name is a certain match. Every inferred position must
identify the supporting source_sequence; explicit facts must cite their
actual sequence. Flag contradictions and missing evidence in warnings.
Respect consolidated designs unless the source contradicts them.
Do not invent events, props, dialogue or camera cuts. Keep descriptions
concise and in the source language. Return only the requested JSON."""

def _verify(data: dict[str, Any], sequence_id: int) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Sequence {sequence_id}: spatial output is not an object.")
    for field in _FIELDS:
        if field not in data:
            raise ValueError(f"Sequence {sequence_id}: missing {field} from spatial output.")
    for name in ("initial_entities", "final_entities"):
        if not isinstance(data[name], list):
            raise ValueError(f"Sequence {sequence_id}: {name} must be a list.")
        for entity in data[name]:
            if not isinstance(entity, dict) or not all(k in entity for k in _ENTITY):
                raise ValueError(f"Sequence {sequence_id}: invalid {name} entity.")
    for name in ("warnings", "justified_changes"):
        if not isinstance(data[name], list) or not all(isinstance(x, str) for x in data[name]):
            raise ValueError(f"Sequence {sequence_id}: invalid {name}.")

class SpatialContinuityNode:
    """Replace the former no-input compatibility placeholder with real planning."""

    DESCRIPTION = ("Plan physical placement of characters, objects, locations and camera "
                   "across ALL consolidated cinematic sequences. Save a per-sequence JSON "
                   "with evidence and explicit uncertainty; does not rewrite source scenes.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "consolidated_references": ("MINIMAX_REGISTRY",),
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 5000, "min": 1024, "max": 100000}),
            "out_dir": ("STRING", {"default": "spatial_continuity",
                                   "tooltip": "Relative folder inside the current timestamped run."}),
        }}

    RETURN_TYPES = ("MINIMAX_SPATIAL_CONTINUITY", "STRING", "STRING")
    RETURN_NAMES = ("spatial_continuity", "spatial_json", "summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @progress.node_progress
    def run(self, consolidated_references: dict[str, Any], lmstudio_config: dict[str, Any],
            temperature: float = 0.1, max_tokens: int = 5000,
            out_dir: str = "spatial_continuity") -> dict[str, Any]:
        util.require_schema(consolidated_references, util.REGISTRY_SCHEMA)
        timelines = consolidated_references.get("chapter_timelines")
        if not isinstance(timelines, dict) or not timelines:
            raise ValueError("Consolidated references have no chapter_timelines.")
        if not isinstance(lmstudio_config, dict):
            raise TypeError("Connect LM Studio Configuration.")
        output = stage_output(lmstudio_config, out_dir)
        output.mkdir(parents=True, exist_ok=True)
        pipeline = lmstudio_pipeline.load("generate")
        client, model = lmstudio_pipeline.make_client_and_model(
            pipeline, str(lmstudio_config["api_url"]), lmstudio_config)
        chapters: dict[str, Any] = {}
        tasks = [(chapter_id, timeline) for chapter_id, timeline in timelines.items()]
        with client:
            for chapter_id, timeline in progress.steps(tasks):
                sequences = timeline.get("sequences", [])
                if not isinstance(sequences, list):
                    raise ValueError(f"{chapter_id}: malformed sequences.")
                sequence_ids = [seq.get("sequence") for seq in sequences]
                if any(not isinstance(s, int) for s in sequence_ids) or len(set(sequence_ids)) != len(sequence_ids):
                    raise ValueError(f"{chapter_id}: invalid or duplicate sequence IDs.")
                ordered = sorted(sequences, key=lambda seq: seq["sequence"])
                planned: list[dict[str, Any]] = []
                for sequence in ordered:
                    lmstudio_pipeline.comfy_interrupt_check()
                    seq_id = sequence["sequence"]
                    request = {
                        "target_sequence": seq_id,
                        "whole_chapter_timeline": ordered,
                        "entity_registry": consolidated_references.get("entities", []),
                        "chapter_references": timeline.get("references", {}),
                        "visual_designs": consolidated_references.get("visual_designs", {}),
                        "previous_final_state": planned[-1]["final_entities"] if planned else [],
                        "previous_camera": planned[-1]["camera"] if planned else {},
                    }
                    answer = chat_json(client, model, _SYSTEM,
                                       json.dumps(request, ensure_ascii=False),
                                       _SCHEMA, float(temperature), int(max_tokens))
                    _verify(answer, seq_id)
                    planned.append({
                        "sequence": seq_id,
                        "location_id": answer["location_id"],
                        "location_layout": answer["location_layout"],
                        "camera": {key: answer[key] for key in
                                   ("camera_position", "camera_direction", "camera_movement", "camera_axis")},
                        "initial_entities": answer["initial_entities"],
                        "final_entities": answer["final_entities"],
                        "justified_changes": answer["justified_changes"],
                        "warnings": answer["warnings"],
                    })
                chapters[chapter_id] = {"sequences": planned}
        payload = {
            "schema_version": "minimax-spatial-continuity.v2",
            "source_digest": consolidated_references.get("source_digest", ""),
            "chapters": chapters,
        }
        util.save_json(output / "spatial_continuity.json", payload)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        count = sum(len(c["sequences"]) for c in chapters.values())
        summary = f"Spatial continuity: {len(chapters)} chapters, {count} sequences; saved to {output / 'spatial_continuity.json'}"
        return {"ui": {"text": [summary]}, "result": (payload, rendered, summary)}
