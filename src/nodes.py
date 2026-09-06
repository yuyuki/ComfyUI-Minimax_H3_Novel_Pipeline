"""Compatibility exports and ComfyUI mappings for the MiniMax H3 nodes."""
from __future__ import annotations

from .consolidate_references import ConsolidateReferencesNode
from .chapter_selection import SelectChaptersNode
from .extract_chapter_references import ExtractChapterReferencesNode
from .generate_h3_prompts import GenerateH3PromptsNode
from .load_chapter_catalogs import LoadChapterCatalogsNode
from .load_consolidated_references import LoadConsolidatedReferencesNode
from .lmstudio_config import LMStudioConfigurationNode


NODE_CLASS_MAPPINGS = {
    "LMStudioConfigurationNode": LMStudioConfigurationNode,
    "SelectChaptersNode": SelectChaptersNode,
    "ExtractChapterReferencesNode": ExtractChapterReferencesNode,
    "LoadChapterCatalogsNode": LoadChapterCatalogsNode,
    "LoadConsolidatedReferencesNode": LoadConsolidatedReferencesNode,
    "ConsolidateReferencesNode": ConsolidateReferencesNode,
    "GenerateH3PromptsNode": GenerateH3PromptsNode,
}

__all__ = [
    "ExtractChapterReferencesNode",
    "LMStudioConfigurationNode",
    "SelectChaptersNode",
    "LoadChapterCatalogsNode",
    "LoadConsolidatedReferencesNode",
    "ConsolidateReferencesNode",
    "GenerateH3PromptsNode",
    "NODE_CLASS_MAPPINGS",
]
