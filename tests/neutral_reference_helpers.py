"""Explicit neutral reference fixtures for current-contract pipeline tests."""
from copy import deepcopy

from minimax_h3_novel_pipeline.core import reference_visuals


def observation(item, text="Fixture evidence.", kind="characters"):
    result = deepcopy(item)
    result.setdefault("presence", "visible")
    result.setdefault("visual_traits", [])
    if not result["visual_traits"] and kind != "characters":
        result["visual_traits"] = [{"attribute": "construction", "description": "A rough wooden surface.",
                                    "scope": "intrinsic", "evidence": text[:120]}]
    return result


def candidates(result, text):
    return {**result, **{kind: [observation(item, text, kind) for item in result.get(kind, [])]
                        for kind in ("characters", "locations", "objects")}}


def entity(item, chapters=("chapter",), sequences=(1, 2, 3)):
    result = deepcopy(item)
    common = [{"attribute": "appearance", "description": result.get("stable_visual_description") or "A rough wooden surface.",
               "scope": "intrinsic", "evidence": "Fixture evidence."}]
    result["visual_reference"] = {"common_traits": common, "scenes": {
        cid: {str(s): {"presence": "visible", "traits": [], "excluded_attributes": []} for s in sequences} for cid in chapters}}
    if not result.get("timeline"):
        result["timeline"] = {cid: {str(s): {"source": [{"presence": "visible", "visual_traits": deepcopy(common)}]}
                                   for s in sequences} for cid in chapters}
    return result


def catalog(value):
    result = deepcopy(value)
    for kind in ("characters", "locations", "objects"):
        for item in result[kind]:
            first = next(iter(item["state_by_sequence"].values()))["source"]
            if first and kind != "characters" and item.get("stable_visual_description"):
                first[0].setdefault("visual_traits", [{"attribute": "appearance",
                    "description": item["stable_visual_description"], "scope": "intrinsic", "evidence": "Fixture evidence."}])
            for entry in item["state_by_sequence"].values():
                entry["source"] = [observation(o, kind=kind) for o in entry["source"]]
            item["visual_reference"] = reference_visuals.compile_reference(
                {result["chapter_id"]: item["state_by_sequence"]}, [], item["local_id"])
    return result
