export const pipeline = {
  name: "MiniMax H3 Novel Pipeline",
  workflow: [
    "SelectChaptersNode",
    "CinematicChapterAdapterNode",
    "ExtractChapterReferencesNode",
    "ConsolidateReferencesNode",
    "GenerateH3PromptsNode",
  ],
  characterProfilesWorkflow: [
    "CharacterEvidenceExtractorNode",
    "ConsolidateCharacterProfilesNode",
    "CharacterVisualProfileGeneratorNode",
  ],
  characterProfilesTarget: "ConsolidateReferencesNode",
  nodes: [
    {
      id: "LMStudioConfigurationNode",
      name: "LM Studio Configuration",
      category: "Setup",
      purpose: "Shares the model and connection settings used by the AI processing nodes.",
    },
    {
      id: "SelectChaptersNode",
      name: "Select Chapters",
      category: "Core workflow",
      purpose: "Picks chapter files or a folder to process.",
    },
    {
      id: "CinematicChapterAdapterNode",
      name: "Cinematic Chapter Adapter",
      category: "Core workflow",
      purpose: "Turns chapter text into ordered, reviewable scene outlines.",
    },
    {
      id: "LoadCinematicChaptersNode",
      name: "Load Cinematic Chapters",
      category: "Load and resume",
      purpose: "Loads saved scene outlines for another workflow run.",
    },
    {
      id: "ExtractChapterReferencesNode",
      name: "Extract Chapter References",
      category: "Core workflow",
      purpose: "Finds characters, places, objects and story details in chapters.",
    },
    {
      id: "LoadChapterCatalogsNode",
      name: "Load Chapter Catalogs",
      category: "Load and resume",
      purpose: "Loads saved chapter-by-chapter reference results.",
    },
    {
      id: "ConsolidateReferencesNode",
      name: "Consolidate References",
      category: "Core workflow",
      purpose: "Combines chapter references into a shared registry and creates image/audio briefs.",
    },
    {
      id: "LoadConsolidatedReferencesNode",
      name: "Load Consolidated References",
      category: "Load and resume",
      purpose: "Loads a saved registry to continue without repeating consolidation.",
    },
    {
      id: "GenerateH3PromptsNode",
      name: "Generate H3 Prompts",
      category: "Core workflow",
      purpose: "Builds scene prompts and reference instructions for MiniMax H3.",
    },
    {
      id: "SpatialContinuityNode",
      name: "Spatial Continuity",
      category: "Continuity",
      purpose: "Plans character, object, location and camera positions across scenes.",
    },
    {
      id: "LoadSpatialContinuityNode",
      name: "Load Spatial Continuity",
      category: "Continuity",
      purpose: "Loads and checks an edited spatial-continuity plan.",
    },
    {
      id: "CharacterEvidenceExtractorNode",
      name: "Character Evidence Extractor",
      category: "Character profiles",
      purpose: "Collects quoted, chapter-linked evidence about character appearance.",
    },
    {
      id: "LoadCharacterEvidenceNode",
      name: "Load Character Evidence",
      category: "Character profiles",
      purpose: "Loads saved or manually edited character evidence.",
    },
    {
      id: "ConsolidateCharacterProfilesNode",
      name: "Consolidate Character Profiles",
      category: "Character profiles",
      purpose: "Groups evidence into source-linked character profiles.",
    },
    {
      id: "LoadCharacterProfilesNode",
      name: "Load Character Profiles",
      category: "Character profiles",
      purpose: "Loads and validates saved character profiles.",
    },
    {
      id: "CharacterVisualProfileGeneratorNode",
      name: "Character Visual Profile Generator",
      category: "Character profiles",
      purpose:
        "Creates permanent physical designs, then uses LM Studio to review possible duplicate identities and update the same files.",
    },
    {
      id: "LoadCharacterVisualProfilesNode",
      name: "Load Character Visual Profiles",
      category: "Character profiles",
      purpose: "Loads saved physical character profiles for Consolidate References.",
    },
  ],
};

export const categories = [...new Set(pipeline.nodes.map((node) => node.category))];

export function getPipelineOverview() {
  return {
    name: pipeline.name,
    totalNodes: pipeline.nodes.length,
    categories: categories.map((name) => ({
      name,
      count: pipeline.nodes.filter((node) => node.category === name).length,
    })),
    workflow: pipeline.workflow.map((id) => pipeline.nodes.find((node) => node.id === id)),
    characterProfilesWorkflow: pipeline.characterProfilesWorkflow.map((id) =>
      pipeline.nodes.find((node) => node.id === id),
    ),
    characterProfilesTarget: findNode(pipeline.characterProfilesTarget),
  };
}

export function searchNodes(query) {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  return pipeline.nodes.filter((node) =>
    [node.id, node.name, node.category, node.purpose].some((value) =>
      value.toLocaleLowerCase().includes(normalizedQuery),
    ),
  );
}

export function findNode(nodeId) {
  return pipeline.nodes.find((node) => node.id === nodeId);
}
