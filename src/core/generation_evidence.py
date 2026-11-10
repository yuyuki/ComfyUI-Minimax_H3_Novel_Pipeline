"""Deterministic upstream evidence snapshots for H3; never calls a model."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .generation_requests import atomic_json, digest
from .lmstudio_pipeline import comfy_interrupt_check
from .path_access import confined_path


VERSION = 1


def prepare(refs, chapter_id, sequences, spatial_plan, period, cache_dirs, *,
            catalog_for, context_for, dependencies, force=False):
    """Reuse only exact upstream inputs, carry forward and return detached data.

    Callers supply resolved/validated timeline and spatial inputs. Source text,
    selected phases, character states and inferred staging remain separate.
    Model settings deliberately do not participate in this deterministic key.
    """
    comfy_interrupt_check()
    sequence_ids = [row["sequence"] for row in sequences]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("Duplicate sequences in H3 evidence inputs.")
    spatial = ({row["sequence"]: row for row in spatial_plan["chapters"][chapter_id]["sequences"]}
               if spatial_plan is not None else {})
    if spatial_plan is not None:
        if (set(spatial) != set(sequence_ids)
                or len(spatial) != len(spatial_plan["chapters"][chapter_id]["sequences"])):
            raise ValueError("H3 spatial evidence does not cover the selected chapter sequences exactly once.")
        if spatial_plan.get("source_digest", "") != refs.get("source_digest", ""):
            raise ValueError("H3 spatial evidence is stale; rerun Spatial Continuity.")
    dependencies = {**dependencies, "evidence_compiler": sha256(Path(__file__).read_bytes()).hexdigest()}
    key = digest([VERSION, refs, chapter_id, sequences, spatial_plan, period, dependencies])
    target = confined_path("evidence_bundle.json", cache_dirs[0])
    reasons = []
    for directory in cache_dirs if not force else []:
        try:
            record = json.loads(confined_path("evidence_bundle.json", directory).read_text(encoding="utf-8"))
            if record["version"] != VERSION or record["key"] != key:
                reasons.append("dependencies_changed")
                continue
            bundle = record["bundle"]
            if record["bundle_digest"] != digest(bundle) or not _valid(bundle, chapter_id, sequence_ids):
                reasons.append("invalid_bundle")
                continue
        except FileNotFoundError:
            reasons.append("missing")
            continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            reasons.append("invalid_bundle")
            continue
        comfy_interrupt_check()
        if directory != cache_dirs[0]:
            atomic_json(target, record)
        return deepcopy(bundle), {"status": "cached", "digest": record["bundle_digest"]}

    rows = []
    for sequence in sequences:
        comfy_interrupt_check()
        number = sequence["sequence"]
        catalog = catalog_for(refs, chapter_id, number)
        context = context_for(refs, chapter_id, number)
        # Each occurrence retains chapter/sequence/phase provenance, even when
        # two phases contain exactly the same text. No summary or inferred state.
        phases = [{"id": digest([chapter_id, number, phase]), "chapter_id": chapter_id,
                   "sequence": number, "phase": phase, "text": text,
                   "origin": "resolved_cinematic_adaptation"}
                  for phase, text in sequence["adaptation"].items()]
        rows.append({"sequence": deepcopy(sequence), "catalog": deepcopy(catalog),
                     "character_context": deepcopy(context), "phases": phases,
                     "spatial_plan": deepcopy(spatial.get(number))})
    bundle = {
        "chapter_id": chapter_id, "source_digest": refs.get("source_digest", ""),
        "upstream_digest": digest(refs), "period_instruction": period,
        "sequences": rows,
        "references": {name: deepcopy(refs.get(name, []))
                       for name in ("entities", "picture_assets", "audio_assets")},
        "chapter_entity_map": deepcopy(refs.get("chapter_entity_map", {}).get(chapter_id, {})),
        "visual_designs": deepcopy({key: value for key, value in refs.get("visual_designs", {}).items()
                                    if key != "character_profiles"}),
    }
    if not _valid(bundle, chapter_id, sequence_ids):
        raise ValueError("Invalid compiled H3 evidence bundle.")
    record = {"version": VERSION, "key": key, "bundle": bundle, "bundle_digest": digest(bundle)}
    comfy_interrupt_check()
    atomic_json(target, record)
    return deepcopy(bundle), {"status": "compiled", "digest": record["bundle_digest"],
                              "reason": "force" if force else ",".join(dict.fromkeys(reasons))}


def _valid(bundle, chapter_id, sequence_ids):
    if not isinstance(bundle, dict) or bundle.get("chapter_id") != chapter_id:
        return False
    rows = bundle.get("sequences")
    if not isinstance(rows, list) or [row["sequence"]["sequence"] for row in rows] != sequence_ids:
        return False
    if not isinstance(bundle.get("period_instruction"), str) or not isinstance(bundle.get("visual_designs"), dict):
        return False
    if not isinstance(bundle.get("chapter_entity_map"), dict):
        return False
    refs = bundle.get("references", {})
    if any(not isinstance(refs.get(key), list) for key in ("entities", "picture_assets", "audio_assets")):
        return False
    for row in rows:
        if (not isinstance(row["catalog"], list) or any(not isinstance(item, dict) for item in row["catalog"])
                or not isinstance(row["phases"], list)
                or not (row["character_context"] is None or isinstance(row["character_context"], dict))
                or not (row["spatial_plan"] is None or isinstance(row["spatial_plan"], dict))):
            return False
        number = row["sequence"]["sequence"]
        expected = [{"id": digest([chapter_id, number, phase]), "chapter_id": chapter_id,
                     "sequence": number, "phase": phase, "text": text,
                     "origin": "resolved_cinematic_adaptation"}
                    for phase, text in row["sequence"]["adaptation"].items()]
        if row["phases"] != expected:
            return False
    return True
