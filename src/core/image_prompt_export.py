"""Deterministic Qwen-Image text exports; never generates media or calls an LLM."""
from __future__ import annotations

import re

from . import util
from .path_access import confined_path

FOLDERS = {"character": "characters", "location": "places", "object": "objects"}


def export_image_prompts(registry, output):
    output = util.output_path(output)
    designs = {item["global_id"]: item for item in registry.get("visual_designs", {}).get("entities", [])}
    grouped = {}
    for asset in registry.get("picture_assets", []):
        grouped.setdefault(asset["linked_global_id"], []).append(asset)
    records, texts = [], []
    for entity in registry.get("entities", []):
        gid = entity["global_id"]
        assets = grouped.get(gid, [])
        if not assets:
            continue
        added = designs.get(gid, {}).get("added_details", {})
        description = entity.get("stable_visual_description", "")
        record = {"global_id": gid, "entity_type": entity["entity_type"],
                  "canonical_name": entity["canonical_name"], "source_description": description,
                  "added_details": added, "image_style": registry.get("image_style", ""),
                  "prompts": [{key: asset.get(key, "") for key in (
                      "asset_id", "view_type", "variant", "description", "generation_prompt", "suggested_filename",
                  )} for asset in assets]}
        lines = [f"{gid} — {entity['canonical_name']}", f"Source-supported description: {description}",
                 "Added design details (adaptation choices; editable in visual_designs.json):",
                 *(f"- {trait}: {value}" for trait, value in added.items())]
        if not added:
            lines.append("None recorded.")
        for prompt in record["prompts"]:
            lines.extend(["", f"=== {prompt['asset_id']} | {prompt['view_type']} | {prompt['variant']} ===",
                          f"Description: {prompt['description']}", f"Suggested image: {prompt['suggested_filename']}",
                          "COPY-PASTE PROMPT:", prompt["generation_prompt"]])
        text = "\n".join(lines).strip() + "\n"
        name = re.sub(r"[^\w-]+", "_", entity["canonical_name"], flags=re.UNICODE).strip("_")[:60] or "entity"
        # Validate IDs rather than sanitizing them into potentially colliding filenames.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", gid):
            raise ValueError("Invalid entity ID for image export.")
        target = confined_path(f"{FOLDERS[entity['entity_type']]}/{gid}_{name}.txt", output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        record["text_file"] = target.relative_to(output).as_posix()
        records.append(record)
        texts.append(text)
    util.save_json(output / "image_prompts.json", {"schema_version": "minimax-h3-image-prompts.v1", "entities": records})
    return records, "\n\n".join(texts)
