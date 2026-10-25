"""Node exports and ComfyUI mappings for the MiniMax H3 nodes."""
from __future__ import annotations

from .character_visual_profile_generator import CharacterVisualProfileGeneratorNode
from .consolidate_references import ConsolidateReferencesNode
from .consolidate_character_profiles import ConsolidateCharacterProfilesNode
from .character_evidence_extractor import CharacterEvidenceExtractorNode
from .chapter_selection import SelectChaptersNode
from .cinematic_chapter_adapter import CinematicChapterAdapterNode
from .extract_chapter_references import ExtractChapterReferencesNode
from .generate_h3_prompts import GenerateH3PromptsNode
from .load_character_profiles import LoadCharacterProfilesNode
from .load_character_evidence import LoadCharacterEvidenceNode
from .load_character_visual_profiles import LoadCharacterVisualProfilesNode
from .load_chapter_catalogs import LoadChapterCatalogsNode
from .load_cinematic_chapters import LoadCinematicChaptersNode
from .load_consolidated_references import LoadConsolidatedReferencesNode
from .lmstudio_config import LMStudioConfigurationNode
from .spatial_continuity import SpatialContinuityNode
from .load_spatial_continuity import LoadSpatialContinuityNode


NODE_CLASS_MAPPINGS = {
    "LoadCharacterVisualProfilesNode": LoadCharacterVisualProfilesNode,
    "CharacterVisualProfileGeneratorNode": CharacterVisualProfileGeneratorNode,
    "ConsolidateCharacterProfilesNode": ConsolidateCharacterProfilesNode,
    "LoadCharacterEvidenceNode": LoadCharacterEvidenceNode,
    "LoadCharacterProfilesNode": LoadCharacterProfilesNode,
    "CharacterEvidenceExtractorNode": CharacterEvidenceExtractorNode,
    "CinematicChapterAdapterNode": CinematicChapterAdapterNode,
    "LMStudioConfigurationNode": LMStudioConfigurationNode,
    "SelectChaptersNode": SelectChaptersNode,
    "ExtractChapterReferencesNode": ExtractChapterReferencesNode,
    "LoadChapterCatalogsNode": LoadChapterCatalogsNode,
    "LoadCinematicChaptersNode": LoadCinematicChaptersNode,
    "LoadConsolidatedReferencesNode": LoadConsolidatedReferencesNode,
    "ConsolidateReferencesNode": ConsolidateReferencesNode,
    "GenerateH3PromptsNode": GenerateH3PromptsNode,
    "SpatialContinuityNode": SpatialContinuityNode,
    "LoadSpatialContinuityNode": LoadSpatialContinuityNode,
}

__all__ = [
    "LoadCharacterVisualProfilesNode",
    "CharacterVisualProfileGeneratorNode",
    "ConsolidateCharacterProfilesNode",
    "LoadCharacterEvidenceNode",
    "LoadCharacterProfilesNode",
    "CharacterEvidenceExtractorNode",
    "CinematicChapterAdapterNode",
    "ExtractChapterReferencesNode",
    "LMStudioConfigurationNode",
    "SelectChaptersNode",
    "LoadChapterCatalogsNode",
    "LoadCinematicChaptersNode",
    "LoadConsolidatedReferencesNode",
    "ConsolidateReferencesNode",
    "GenerateH3PromptsNode",
    "SpatialContinuityNode",
    "LoadSpatialContinuityNode",
    "NODE_CLASS_MAPPINGS",
]
