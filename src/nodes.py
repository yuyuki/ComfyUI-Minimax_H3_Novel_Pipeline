"""Compatibility exports and ComfyUI mappings for the MiniMax H3 nodes."""
from __future__ import annotations

from .consolidate_references import ConsolidateReferencesNode
from .chapter_selection import SelectChaptersNode
from .cinematic_chapter_adapter import CinematicChapterAdapterNode
from .extract_chapter_references import ExtractChapterReferencesNode
from .generate_h3_prompts import GenerateH3PromptsNode
from .load_chapter_catalogs import LoadChapterCatalogsNode
from .load_consolidated_references import LoadConsolidatedReferencesNode
from .lmstudio_config import LMStudioConfigurationNode
from .spatial_continuity import SpatialContinuityNode
from .narrative_nodes import NarrativeContinuityNode, NovelCinematicSimplifierNode


NODE_CLASS_MAPPINGS = {
    "CinematicChapterAdapterNode": CinematicChapterAdapterNode,
    "NarrativeContinuityNode": NarrativeContinuityNode,
    "NovelCinematicSimplifierNode": NovelCinematicSimplifierNode,
    "LMStudioConfigurationNode": LMStudioConfigurationNode,
    "SelectChaptersNode": SelectChaptersNode,
    "ExtractChapterReferencesNode": ExtractChapterReferencesNode,
    "LoadChapterCatalogsNode": LoadChapterCatalogsNode,
    "LoadConsolidatedReferencesNode": LoadConsolidatedReferencesNode,
    "ConsolidateReferencesNode": ConsolidateReferencesNode,
    "GenerateH3PromptsNode": GenerateH3PromptsNode,
    "SpatialContinuityNode": SpatialContinuityNode,
}

__all__ = [
    "CinematicChapterAdapterNode",
    "NarrativeContinuityNode",
    "NovelCinematicSimplifierNode",
    "ExtractChapterReferencesNode",
    "LMStudioConfigurationNode",
    "SelectChaptersNode",
    "LoadChapterCatalogsNode",
    "LoadConsolidatedReferencesNode",
    "ConsolidateReferencesNode",
    "GenerateH3PromptsNode",
    "SpatialContinuityNode",
    "NODE_CLASS_MAPPINGS",
]
