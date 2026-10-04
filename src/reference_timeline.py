"""Lossless temporal storage, separate from model-consolidated reference identity."""
from copy import deepcopy

from .cinematic_references import PHASES, TEMPORAL_VERSION, parse_chapters
from . import util


def validate_catalogs(chapters):
    """Reject legacy catalogs and ambiguous temporal addresses before any model work."""
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("Supply a non-empty list of cinematic chapter catalogs.")
    seen = set()
    for chapter in chapters:
        if not isinstance(chapter, dict):
            raise ValueError("Each chapter catalog must be an object.")
        util.require_schema(chapter, util.CHAPTER_SCHEMA)
        cid = chapter.get("chapter_id")
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            raise ValueError("Chapter IDs must be non-empty and unique.")
        seen.add(cid)
        if chapter.get("timeline_version") != TEMPORAL_VERSION:
            raise ValueError("Re-extract chapter catalogs with the cinematic timeline format.")
        parse_chapters([chapter])
        sequences = {str(s["sequence"]) for s in chapter["sequences"]}
        local_ids = set()
        for kind in ("characters", "locations", "objects"):
            if not isinstance(chapter.get(kind), list):
                raise ValueError(f"{cid}: {kind} must be a list.")
            for entity in chapter[kind]:
                if not isinstance(entity, dict):
                    raise ValueError(f"{cid}: references must be objects.")
                lid = entity.get("local_id")
                if not isinstance(lid, str) or not lid or lid in local_ids:
                    raise ValueError(f"{cid}: local IDs must be non-empty and unique.")
                local_ids.add(lid)
                states = entity.get("state_by_sequence")
                if not isinstance(states, dict) or not states:
                    raise ValueError(f"{cid}/{lid}: state_by_sequence is required.")
                for sequence, phases in states.items():
                    if sequence not in sequences or not isinstance(phases, dict) or not phases:
                        raise ValueError(f"{cid}/{lid}: invalid sequence address.")
                    for phase, observations in phases.items():
                        if phase not in PHASES or not isinstance(observations, list) or not observations:
                            raise ValueError(f"{cid}/{lid}: invalid phase observations.")
                        if not all(isinstance(o, dict) for o in observations):
                            raise ValueError(f"{cid}/{lid}: observations must be objects.")


def merge_timeline(entity, timeline):
    """Identity merges append every observation, including conflicting evidence."""
    target = entity.setdefault("timeline", {})
    for chapter, sequences in timeline.items():
        destination = target.setdefault(chapter, {})
        for sequence, phases in sequences.items():
            for phase, observations in phases.items():
                destination.setdefault(sequence, {}).setdefault(phase, []).extend(deepcopy(observations))
        target[chapter] = {
            sequence: {phase: destination[sequence][phase] for phase in PHASES if phase in destination[sequence]}
            for sequence in sorted(destination, key=int)
        }
    refresh_first_occurrence(entity)


def refresh_first_occurrence(entity, chapter_order=None):
    timeline = entity["timeline"]
    if chapter_order is not None:
        entity["timeline"] = timeline = {cid: timeline[cid] for cid in chapter_order if cid in timeline}
    first = {}
    for cid, sequences in timeline.items():
        sequence, phase = min((int(s), PHASES.index(p)) for s, phases in sequences.items()
                              for p, observations in phases.items() if observations)
        first[cid] = {"first_sequence": sequence, "first_phase": PHASES[phase]}
    entity["first_occurrence_by_chapter"] = first
    if first:
        cid = next(iter(first))
        entity.update(first_chapter_id=cid, **first[cid])


def state_at(entity, chapter_id, sequence, phase):
    """Exact observations only; absence is unknown, never future or carried state."""
    if type(sequence) is not int or sequence < 1 or phase not in PHASES:
        raise ValueError("State lookup requires a positive sequence and an explicit phase.")
    return deepcopy(entity["timeline"].get(chapter_id, {}).get(str(sequence), {}).get(phase, []))
