"""Chapter-wide spatial staging from consolidated cinematic timelines.

All sequences in the chapter are supplied to every planning call, allowing future
facts to constrain earlier placement without pulling future actions backward.
"""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator
from typing import Any

from . import progress, lmstudio_pipeline, util
from .lmstudio_json import chat_json
from .run_output import stage_output
from .editable_schemas import SCHEMA_DIR, export_schema, validate_document

SPATIAL_SCHEMA = json.loads((SCHEMA_DIR / "spatial_continuity.schema.json").read_text(encoding="utf-8"))
_ENTITY = SPATIAL_SCHEMA["$defs"]["entity"]["properties"]
# LM Studio receives a flat, self-contained per-sequence response schema.
_FIELDS = {
    **{k: v for k, v in SPATIAL_SCHEMA["$defs"]["sequence"]["properties"].items()
       if k not in ("sequence", "camera", "initial_entities", "final_entities")},
    **SPATIAL_SCHEMA["$defs"]["camera"]["properties"],
    **{k: {"type": "array", "items": SPATIAL_SCHEMA["$defs"]["entity"]}
       for k in ("initial_entities", "final_entities")},
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
The target sequence's source and adaptation are authoritative for actions and
visibility. first_sequence/first_phase mark the first source observation, not
physical creation. Earlier placement inferred from later evidence must remain
inferred and offscreen until the source reveals it. Never label future evidence
as an explicit current fact. Keep distinct instances (such as a lost torch and
its replacement, or a suspension rope and a torch line) distinct in state prose.
Return initial and final states including important offscreen persistent
entities. A change requires an event in the target sequence; record its cause
in justified_changes. When placement is not supportable, use 'unknown', not
fabricated coordinates. Use consolidated global IDs where available; never
pretend an ambiguous name is a certain match.
Use each matched entity's exact consolidated canonical_name; use an empty
canonical_name when entity_id has no matching consolidated global_id.
Every inferred position must identify the supporting source_sequence; explicit facts must cite their
actual sequence. Flag contradictions and missing evidence in warnings.
Respect consolidated designs unless the source contradicts them.
Do not invent events, props, dialogue or camera cuts. Keep descriptions
concise and in the source language. Return only the requested JSON."""

def _verify(data: dict[str, Any], sequence_id: int) -> None:
    errors = list(Draft202012Validator(_SCHEMA["schema"]).iter_errors(data))
    if errors:
        raise ValueError(f"Sequence {sequence_id}: invalid spatial output: {errors[0].message}")
    for entity in data["initial_entities"] + data["final_entities"]:
        if entity["evidence"] == "explicit" and entity["source_sequence"] > sequence_id:
            raise ValueError(
                f"Sequence {sequence_id}: spatial output treats future evidence as current fact "
                f"for {entity['entity_id']} (source sequence {entity['source_sequence']}); "
                "rerun Spatial Continuity. Earlier placement must be inferred, not explicit; "
                "future actions and revelations must not happen now."
            )


def validate_plan(payload: dict[str, Any], registry: dict[str, Any]) -> None:
    """Reject stale, incomplete or obsolete plans before generation starts."""
    if not isinstance(payload, dict) or payload.get("schema_version") != "minimax-spatial-continuity.v2":
        raise ValueError("Invalid spatial_continuity. Connect the current Spatial Continuity output.")
    validate_document(payload, SPATIAL_SCHEMA, "invalid spatial output")
    if payload.get("source_digest") != registry.get("source_digest", ""):
        raise ValueError("Spatial continuity is stale; rerun it with the current consolidation.")
    chapters = payload.get("chapters")
    timelines = registry.get("chapter_timelines", {})
    names = {entity["global_id"]: entity["canonical_name"] for entity in registry.get("entities", [])}
    if not isinstance(chapters, dict) or set(chapters) != set(timelines):
        raise ValueError("Spatial continuity must cover every consolidated chapter.")
    for cid, timeline in timelines.items():
        chapter = chapters[cid]
        plans = chapter.get("sequences") if isinstance(chapter, dict) else None
        expected = [s["sequence"] for s in timeline["sequences"]]
        if (not isinstance(plans, list) or any(not isinstance(p, dict) for p in plans)
                or [p.get("sequence") for p in plans] != sorted(expected)):
            raise ValueError(f"{cid}: spatial continuity must cover every sequence once, in order.")
        for plan in plans:
            camera = plan.get("camera")
            if not isinstance(camera, dict):
                raise ValueError(f"{cid}: invalid spatial camera.")
            answer = {k: v for k, v in plan.items() if k not in ("sequence", "camera")}
            answer.update(camera)
            _verify(answer, plan["sequence"])
            for entity in answer["initial_entities"] + answer["final_entities"]:
                if entity["canonical_name"] != names.get(entity["entity_id"], ""):
                    raise ValueError(f"{cid}: spatial canonical_name does not match consolidated references.")
                if entity["evidence"] != "unknown" and entity["source_sequence"] not in expected:
                    raise ValueError(f"{cid}: spatial evidence cites an unknown sequence.")


class SpatialContinuityNode:
    """Plan spatial continuity using the complete consolidated chapter."""

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
        names = {entity["global_id"]: entity["canonical_name"]
                 for entity in consolidated_references.get("entities", [])}
        tasks = [(chapter_id, timeline) for chapter_id, timeline in timelines.items()]
        with client:
            for chapter_id, timeline in progress.steps(tasks):
                sequences = timeline.get("sequences") if isinstance(timeline, dict) else None
                if not isinstance(sequences, list) or not sequences or any(not isinstance(s, dict) for s in sequences):
                    raise ValueError(f"{chapter_id}: malformed sequences.")
                sequence_ids = [seq.get("sequence") for seq in sequences]
                if any(type(s) is not int or s < 1 for s in sequence_ids) or len(set(sequence_ids)) != len(sequence_ids):
                    raise ValueError(f"{chapter_id}: invalid or duplicate sequence IDs.")
                ordered = sorted(sequences, key=lambda seq: seq["sequence"])
                planned: list[dict[str, Any]] = []
                for sequence in ordered:
                    lmstudio_pipeline.comfy_interrupt_check()
                    seq_id = sequence["sequence"]
                    request = {
                        "target_sequence": seq_id,
                        "chapter_id": chapter_id,
                        "whole_chapter_timeline": ordered,
                        "entity_registry": consolidated_references.get("entities", []),
                        "chapter_references": timeline.get("references", {}),
                        "visual_designs": consolidated_references.get("visual_designs", {}),
                        "previous_final_state": planned[-1]["final_entities"] if planned else [],
                        "previous_camera": planned[-1]["camera"] if planned else {},
                    }
                    for attempt in range(3):
                        answer = chat_json(client, model, _SYSTEM,
                                           json.dumps(request, ensure_ascii=False),
                                           _SCHEMA, float(temperature), int(max_tokens))
                        try:
                            _verify(answer, seq_id)
                        except ValueError as error:
                            if attempt == 2:
                                raise
                            request["correction_required"] = str(error)
                        else:
                            break
                    for entity in answer["initial_entities"] + answer["final_entities"]:
                        entity["canonical_name"] = names.get(entity["entity_id"], "")
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
            "$schema": "spatial_continuity.schema.json",
            "schema_version": "minimax-spatial-continuity.v2",
            "source_digest": consolidated_references.get("source_digest", ""),
            "chapters": chapters,
        }
        validate_plan(payload, consolidated_references)
        export_schema(output, "spatial_continuity")
        util.save_json(output / "spatial_continuity.json", payload)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        count = sum(len(c["sequences"]) for c in chapters.values())
        summary = f"Spatial continuity: {len(chapters)} chapters, {count} sequences; saved to {output / 'spatial_continuity.json'}"
        return {"ui": {"text": [summary]}, "result": (payload, rendered, summary)}
