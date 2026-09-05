"""Select one generated scene and expose its exact H3 reference order."""
from __future__ import annotations

import json
from typing import Any


class SelectH3SceneNode:
    """Select a generated scene for MiniMax H3 Reference to Video.

    The image and audio IDs are intentionally returned in the exact order used
    by ``<Picture N>`` and ``<Audio N>`` in the prompt.  Attach the matching
    media to MiniMax H3 in that order; the node never fabricates media from
    the textual briefs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("MINIMAX_PROMPTS",),
                "chapter_index": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "scene_index": ("INT", {"default": 1, "min": 1, "max": 100000}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_SCENE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "h3_scene",
        "prompt",
        "image_asset_ids_in_h3_order",
        "audio_asset_ids_in_h3_order",
        "binding_json",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(
        self, prompts: dict[str, Any], chapter_index: int, scene_index: int
    ) -> tuple[dict[str, Any], str, str, str, str]:
        if not isinstance(prompts, dict) or not isinstance(prompts.get("chapters"), list):
            raise TypeError("prompts must be the output of Generate H3 Prompts.")
        chapter_position = int(chapter_index) - 1
        chapters = prompts["chapters"]
        if not 0 <= chapter_position < len(chapters):
            raise ValueError(f"chapter_index {chapter_index} is outside 1..{len(chapters)}.")

        chapter = chapters[chapter_position]
        entries = chapter.get("outputs", [])
        entry = next((item for item in entries if item.get("index") == int(scene_index)), None)
        if entry is None:
            available = [str(item.get("index")) for item in entries]
            raise ValueError(
                f"scene_index {scene_index} was not generated for chapter {chapter.get('chapter_id', chapter_index)!r}. "
                f"Available scene indexes: {', '.join(available) or 'none'}."
            )
        if not entry.get("prompt_text"):
            raise ValueError(
                f"Scene {scene_index} was skipped and has no H3 prompt: {entry.get('reason', 'unknown reason')}"
            )

        bindings = entry.get("bindings", {})
        scene = {
            "schema_version": "minimax-h3-scene.v1",
            "chapter_id": chapter.get("chapter_id", ""),
            "scene_index": entry["index"],
            "title": entry.get("title", ""),
            "prompt": entry["prompt_text"],
            "picture_asset_ids": list(entry.get("picture_asset_ids", [])),
            "audio_asset_ids": list(entry.get("audio_asset_ids", [])),
            "bindings": bindings,
        }
        return (
            scene,
            scene["prompt"],
            "\n".join(scene["picture_asset_ids"]),
            "\n".join(scene["audio_asset_ids"]),
            json.dumps(bindings, ensure_ascii=False, indent=2),
        )
