"""Compatibility exports and ComfyUI mappings for the MiniMax H3 nodes."""
from __future__ import annotations

from .consolidate_references import ConsolidateReferencesNode
from .extract_chapter_references import ExtractChapterReferencesNode
from .generate_h3_prompts import GenerateH3PromptsNode
from .load_chapter_catalogs import LoadChapterCatalogsNode
from .load_consolidated_references import LoadConsolidatedReferencesNode
from .lmstudio_config import LMStudioConfigurationNode
from .select_h3_scene import SelectH3SceneNode


NODE_CLASS_MAPPINGS = {
    "LMStudioConfigurationNode": LMStudioConfigurationNode,
    "ExtractChapterReferencesNode": ExtractChapterReferencesNode,
    "LoadChapterCatalogsNode": LoadChapterCatalogsNode,
    "LoadConsolidatedReferencesNode": LoadConsolidatedReferencesNode,
    "ConsolidateReferencesNode": ConsolidateReferencesNode,
    "GenerateH3PromptsNode": GenerateH3PromptsNode,
    "SelectH3SceneNode": SelectH3SceneNode,
}

__all__ = [
    "ExtractChapterReferencesNode",
    "LMStudioConfigurationNode",
    "LoadChapterCatalogsNode",
    "LoadConsolidatedReferencesNode",
    "ConsolidateReferencesNode",
    "GenerateH3PromptsNode",
    "SelectH3SceneNode",
    "NODE_CLASS_MAPPINGS",
]
