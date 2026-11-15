"""Source-addressed neutral appearance, independent from narrative actions."""
from copy import deepcopy
from hashlib import sha256
import json
import re


VERSION = "neutral-visual-reference.v1"
PRESENCE = ("visible", "offscreen", "mentioned")
TRAIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "attribute": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
        "scope": {"type": "string", "enum": ["intrinsic", "scene"]},
        "evidence": {"type": "string", "minLength": 1},
    },
    "required": ["attribute", "description", "scope", "evidence"],
}


def attribute_key(value):
    return re.sub(r"[^\w]+", "_", value.casefold()).strip("_")


def validate_observation(observation):
    if observation.get("presence") not in PRESENCE or not isinstance(observation.get("visual_traits"), list):
        raise ValueError("Neutral visual observations are required; re-extract chapter references.")
    for trait in observation["visual_traits"]:
        if (not isinstance(trait, dict) or set(trait) != set(TRAIT_SCHEMA["required"])
                or trait["scope"] not in ("intrinsic", "scene")
                or not all(isinstance(trait[k], str) and trait[k].strip() for k in TRAIT_SCHEMA["required"])
                or not attribute_key(trait["attribute"])):
            raise ValueError("Invalid neutral visual trait; re-extract chapter references.")


def validate_evidence(result, source_input):
    """Appearance evidence must belong to this scene, never the chapter context."""
    text = source_input["text"]
    for kind in ("characters", "locations", "objects"):
        for observation in result.get(kind, []):
            validate_observation(observation)
            for trait in observation["visual_traits"]:
                if trait["evidence"] not in text:
                    raise ValueError("Visual trait evidence must quote the local passage, never another scene.")


def compile_reference(timeline, diagnostics, identity):
    """Reuse only unambiguous intrinsic traits; never infer presence or carry state."""
    intrinsic = {}
    observations = []
    for chapter, sequences in timeline.items():
        for sequence, entry in sorted(sequences.items(), key=lambda pair: int(pair[0])):
            for observation in entry["source"]:
                validate_observation(observation)
                observations.append((chapter, sequence, observation))
                for trait in observation["visual_traits"]:
                    if trait["scope"] == "intrinsic":
                        key = attribute_key(trait["attribute"])
                        values = intrinsic.setdefault(key, {})
                        values.setdefault(trait["description"].casefold(), deepcopy(trait))
    common = {key: next(iter(values.values())) for key, values in intrinsic.items() if len(values) == 1}
    for key, values in intrinsic.items():
        if len(values) > 1:
            diagnostics.append({"stage": "neutral_visual_traits", "entity_id": identity,
                                "attribute": key, "status": "incomplete",
                                "message": "Conflicting intrinsic descriptions were not propagated across scenes.",
                                "source_descriptions": [v["description"] for v in values.values()]})
    scenes = {}
    for chapter, sequence, observation in observations:
        state = scenes.setdefault(chapter, {}).setdefault(sequence, {"presence": "mentioned", "traits": [], "excluded_attributes": []})
        if PRESENCE.index(observation["presence"]) < PRESENCE.index(state["presence"]):
            state["presence"] = observation["presence"]
        # Conflicting intrinsic observations remain local, never a guessed global fact.
        for trait in observation["visual_traits"]:
            key = attribute_key(trait["attribute"])
            if trait["scope"] == "scene" or key not in common:
                if trait not in state["traits"]:
                    state["traits"].append(deepcopy(trait))
    for chapter, sequences in scenes.items():
        for sequence, state in sequences.items():
            by_key = {}
            for trait in state["traits"]:
                by_key.setdefault(attribute_key(trait["attribute"]), []).append(trait)
            # An explicit local override is not a conflict with the intrinsic baseline.
            for key, values in by_key.items():
                local = [v for v in values if v["scope"] == "scene"]
                if local:
                    by_key[key] = local
            conflicts = {k for k, values in by_key.items()
                         if len({v["description"].casefold() for v in values}) > 1}
            if conflicts:
                diagnostics.append({"stage": "neutral_visual_traits", "entity_id": identity,
                                    "chapter_id": chapter, "sequence": int(sequence), "status": "incomplete",
                                    "message": "Ambiguous local traits omitted from image appearance.",
                                    "attributes": sorted(conflicts)})
                state["excluded_attributes"] = sorted(conflicts)
            state["traits"] = [v for key, values in by_key.items() if key not in conflicts for v in values]
    return {"common_traits": list(common.values()), "scenes": scenes}


def descriptions(traits):
    return " ".join(dict.fromkeys(t["description"].strip() for t in traits))


def traits_at(entity, chapter, sequence):
    reference = entity["visual_reference"]
    state = reference["scenes"].get(chapter, {}).get(str(sequence))
    if state is None or state["presence"] != "visible":
        return None
    traits = {attribute_key(t["attribute"]): t for t in reference["common_traits"]
              if attribute_key(t["attribute"]) not in state["excluded_attributes"]}
    traits.update({attribute_key(t["attribute"]): t for t in state["traits"]})
    return list(traits.values())


def appearance_at(entity, chapter, sequence):
    traits = traits_at(entity, chapter, sequence)
    return None if traits is None else descriptions(traits)


def variant_id(appearance):
    return "state_" + sha256(appearance.encode("utf-8")).hexdigest()[:16]


def validate_registry(registry):
    for entity in registry.get("entities", []):
        validate_reference(entity.get("visual_reference"))
    entities = {e["global_id"]: e for e in registry.get("entities", [])}
    for asset in registry.get("picture_assets", []):
        if not isinstance(asset.get("scene_bindings"), dict):
            raise ValueError("Scene-addressed image references are required; reconsolidate references.")
        entity = entities.get(asset.get("linked_global_id"))
        if entity is None:
            raise ValueError("Image reference has no established entity; reconsolidate references.")
        for chapter, sequences in asset["scene_bindings"].items():
            if (not isinstance(sequences, list) or not sequences
                    or any(type(s) is not int or s < 1 for s in sequences)
                    or len(sequences) != len(set(sequences))
                    or any(appearance_at(entity, chapter, s) is None for s in sequences)):
                raise ValueError("Image reference must address visible entity occurrences; reconsolidate references.")
            for sequence in sequences:
                key = signature(traits_at(entity, chapter, sequence))
                common_key = signature(entity["visual_reference"]["common_traits"])
                expected = "base" if key == common_key else variant_id(key)
                if asset.get("variant") != expected:
                    raise ValueError("Image reference state does not match its scene; reconsolidate references.")


def validate_reference(reference):
    if not isinstance(reference, dict) or set(reference) != {"common_traits", "scenes"}:
        raise ValueError("Neutral visual references are required; re-extract and reconsolidate references.")
    if not isinstance(reference["common_traits"], list) or not isinstance(reference["scenes"], dict):
        raise ValueError("Invalid neutral visual reference.")
    validate_observation({"presence": "visible", "visual_traits": reference["common_traits"]})
    if any(t["scope"] != "intrinsic" for t in reference["common_traits"]):
        raise ValueError("Common visual appearance must contain intrinsic traits only.")
    for sequences in reference["scenes"].values():
        if not isinstance(sequences, dict):
            raise ValueError("Visual reference scenes must be addressed by chapter and sequence.")
        for sequence, state in sequences.items():
            if (not isinstance(sequence, str) or not re.fullmatch(r"[1-9][0-9]*", sequence)
                    or not isinstance(state, dict) or set(state) != {"presence", "traits", "excluded_attributes"}
                    or not isinstance(state["excluded_attributes"], list)
                    or not all(isinstance(k, str) and k for k in state["excluded_attributes"])):
                raise ValueError("Invalid scene-addressed visual appearance.")
            validate_observation({"presence": state["presence"], "visual_traits": state["traits"]})


def scene_assets(assets, chapter, sequence):
    return [asset for asset in assets if sequence in asset["scene_bindings"].get(chapter, [])]


def signature(traits):
    return json.dumps(sorted({(attribute_key(t["attribute"]), t["description"].casefold()) for t in traits}), ensure_ascii=False)
