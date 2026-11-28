#!/usr/bin/env python3
"""Generate a deterministic architecture map from the active Python package."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src"
OUTPUT_PATH = ROOT / "docs" / "architecture.md"


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    path: str
    dependencies: tuple[str, ...]
    public_symbols: tuple[str, ...]
    comfyui_nodes: tuple[tuple[str, str], ...] = ()


def _module_name(path: Path, source_dir: Path) -> str:
    relative = path.relative_to(source_dir).with_suffix("")
    parts = relative.parts[:-1] if path.name == "__init__.py" else relative.parts
    return ".".join(parts) or "package_init"


def _internal_dependencies(tree: ast.AST, module_names: set[str], package: str) -> tuple[str, ...]:
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        parts = package.split(".") if package else []
        if node.level > 1:
            parts = parts[:-(node.level - 1)]
        base = ".".join([*parts, *([node.module] if node.module else [])])
        candidates = [".".join(filter(None, (base, alias.name))) for alias in node.names]
        matched = {candidate for candidate in candidates if candidate in module_names}
        dependencies.update(matched)
        if not matched and base in module_names:
            dependencies.add(base)
    return tuple(sorted(dependencies))


def _registration_entries(tree: ast.AST, name: str) -> dict[str, ast.expr]:
    """Read literal registration dictionaries without importing the plugin."""
    entries: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        value = None
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
            and node.func.attr == "update"
            and len(node.args) == 1
        ):
            value = node.args[0]
        if isinstance(value, ast.Dict):
            for key, item in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    entries[key.value] = item
    return entries


def _comfyui_display_names(tree: ast.AST) -> dict[str, str]:
    classes = _registration_entries(tree, "NODE_CLASS_MAPPINGS")
    labels = _registration_entries(tree, "NODE_DISPLAY_NAME_MAPPINGS")
    names: dict[str, str] = {}
    for node_id, value in classes.items():
        class_name = value.attr if isinstance(value, ast.Attribute) else value.id if isinstance(value, ast.Name) else None
        label = labels.get(node_id)
        if class_name:
            names[class_name] = label.value if isinstance(label, ast.Constant) and isinstance(label.value, str) else node_id
    return names


def scan_modules(source_dir: Path = SOURCE_DIR) -> list[ModuleInfo]:
    paths = sorted(source_dir.rglob("*.py"))
    module_names = {_module_name(path, source_dir) for path in paths}
    trees = {path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in paths}
    display_names = _comfyui_display_names(trees[source_dir / "__init__.py"]) if source_dir / "__init__.py" in trees else {}
    modules: list[ModuleInfo] = []
    for path in paths:
        tree = trees[path]
        public_symbols = sorted(
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
        )
        modules.append(
            ModuleInfo(
                name=_module_name(path, source_dir),
                path=path.relative_to(ROOT).as_posix(),
                dependencies=_internal_dependencies(
                    tree, module_names, ".".join(path.relative_to(source_dir).parts[:-1])
                ),
                public_symbols=tuple(public_symbols),
                comfyui_nodes=tuple(
                    (node.name, display_names[node.name])
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name in display_names
                ),
            )
        )
    return modules


def _mermaid_id(name: str) -> str:
    return "m_" + "".join(character if character.isalnum() else "_" for character in name)


def _mermaid_label(label: str) -> str:
    return label.replace("&", "#38;").replace('"', "#quot;").replace("<", "#60;").replace(">", "#62;").replace("\n", "<br/>")


def render_architecture(modules: list[ModuleInfo]) -> str:
    display_names = dict(node for module in modules for node in module.comfyui_nodes)

    def node_label(class_name: str, fallback: str) -> str:
        return _mermaid_label(display_names.get(class_name, fallback))

    lines = [
        "# Architecture map",
        "",
        "> Generated by `python tools/generate_architecture.py`. Do not edit this file manually.",
        "",
        "This map covers the active Python package in `src/`.",
        "`src/nodes/` contains the ComfyUI nodes and their exports; `src/core/` contains shared logic and bundled JSON schemas. `src/__init__.py` registers the plugin and local HTTP routes.",
        "",
        "## Pipeline flow",
        "",
        "```mermaid",
        "flowchart TD",
        f'    selector["{node_label("SelectChaptersNode", "Select Chapters")}"] --> '
        f'adapter["{node_label("CinematicChapterAdapterNode", "Cinematic Chapter Adapter")}"]',
        '    config --> adapter',
        f'    selector --> evidence["{node_label("CharacterEvidenceExtractorNode", "Character Evidence Extractor")}"]',
        '    config --> evidence',
        '    evidence --> evidence_json["Exact chapter quotations, positions and fingerprints"]',
        f'    evidence_json --> evidence_loader["{node_label("LoadCharacterEvidenceNode", "Load Character Evidence")}"]',
        f'    evidence --> profiles["{node_label("ConsolidateCharacterProfilesNode", "Consolidate Character Profiles")}"]',
        '    evidence_loader --> profiles',
        '    original_chapters["Original confined chapter files"] --> profiles',
        '    config --> profiles',
        f'    profiles --> visual_profiles["{node_label("CharacterVisualProfileGeneratorNode", "Character Visual Profile Generator")}"]',
        '    config --> visual_profiles',
        '    selector --> visual_profiles',
        '    visual_profiles --> visual_json["Selected permanent physical profiles JSON"]',
        '    visual_profiles --> global_designs["Global visual_designs.json"]',
        '    global_designs --> consolidate',
        '    visual_profiles --> consolidate',
        '    original_chapters --> consolidate',
        f'    visual_json --> visual_loader["{node_label("LoadCharacterVisualProfilesNode", "Load Character Visual Profiles")}"]',
        '    visual_loader --> consolidate',
        '    profiles --> profiles_json["Source-grounded editable character profiles JSON"]',
        f'    profiles_json --> profiles_loader["{node_label("LoadCharacterProfilesNode", "Load Character Profiles")}"]',
        '    profiles_loader --> visual_profiles',
        '    adapter --> adapted["Per-chapter sequence/source/adaptation JSON"]',
        '    adapted --> extract',
        f'    adapted --> loader["{node_label("LoadCinematicChaptersNode", "Load Cinematic Chapters")}"]',
        '    loader --> extract',
        f'    config["{node_label("LMStudioConfigurationNode", "LM Studio Configuration")}"] --> '
        f'extract["{node_label("ExtractChapterReferencesNode", "Extract Chapter References")}"]',
        f'    extract --> consolidate["{node_label("ConsolidateReferencesNode", "Consolidate References")}"]',
        '    consolidate --> timeline["Canonical source observations: chapter / sequence"]',
        '    consolidate --> designs["Prepare stable visual designs"]',
        f'    designs --> generate["{node_label("GenerateH3PromptsNode", "Generate H3 Prompts")}"]',
        '    consolidate --> generate',
        '    adapted --> generate',
        '    loader --> generate',
        '    config --> generate',
        '    generate --> review["H3 validation, camera processing, temporal review/repair"]',
        "```",
        "",
        "The node wrappers coordinate filesystem access, configuration snapshots and run outputs. `cinematic_references` extracts references from each sequence's `adaptation.initialState` and `adaptation.event` using shared `pipeline_step1_extract` helpers; `pipeline_step2_consolidate` and `pipeline_step3_generate` implement consolidation and H3 generation.",
        "H3 caches use exact inputs, prompts, schemas, effective model settings and explicit per-stage behavior versions in `generation_requests.CACHE_CONTRACTS`; cache identity never reads Python source files or inspects their AST. Increment affected contracts when shared helpers, transport or validation change cache semantics. Final prompt checkpoints include every stage contract, while exact requests and draft/format/camera artifacts track their own dependencies. Evidence compiler and cache record-format changes use the respective module's `VERSION`. Integrity checks, current validation, complete review coverage and chapter-local read-only resume remain required.",
        "H3 repairs request strict JSON patches for affected section bodies, grouping related errors in the same section. Exact assigned image IDs can be translated locally to Picture labels. Each proposed merge must resolve its targeted format errors without introducing new ones; rejected patches retain the previous wording and provide feedback for bounded retries. Continuity evidence routes semantic patches to cited prompt sections, and the complete continuity review must run again before approval.",
        "`character_profile_batches` separates bounded identity and precise-trait requests with resumable per-request caches. `character_profile_document` validates editable profiles, selections, sources and direct identity links against the bundled schema. `character_visual_profiles` deterministically projects permanent physical facts for source-selected chapters. The character documents keep source paths, normalized text fingerprints and exact quotations with positions. `character_visual_integration` verifies the sources and joins permanent physical facts to registry characters by a unique name or alias. The first populated chapter initializes the registry directly from its catalog, and the duplicate audit still runs. H3 uses the catalog's sequence-specific presence and appearance for bindings and the resolved adaptation for staging.",
        "The standalone cinematic adapter uses `cinematic_adaptation` and the shared LM Studio JSON transport. Its duration-aware ordered scenes feed extraction and H3 generation directly; generation preserves one scene per sequence. Catalogs and registries retain temporal observations.",
        "Generate H3 Prompts produces prompts for scenes with empty asset bindings. By default, later sequences in each chapter use an opening image supplied directly in MiniMax. The export identifies the previous sequence to use when the action continues, while the current adaptation governs explicit transitions. Opening images stay outside the numbered reference bindings. Spatial Continuity remains optional.",
        "Consolidation plans reusable location angles from shared geometry and scene framing, with approved appearance for fixed landmarks visible in the assigned scenes. It reviews each view for scenery-only framing and falls back to a general view with an incomplete diagnostic when review fails. Spatial Continuity uses canonical registry location IDs. H3 reports missing location coverage and incomplete reviews in scene diagnostic JSON files.",
        "### Evidence by registered node",
        "",
        "| Node | Evidence role |",
        "|---|---|",
        "| Character Evidence Extractor | Resolves numbered citations, checks uncovered appearance cues, and audits enduring traits. |",
        "| Consolidate Character Profiles | Uses local naming evidence and cross-chapter identity review, then checks trait chronology. |",
        "| Character Visual Profile Generator | Projects stable traits for selected chapters, including characters named there under a verified alias. |",
        "| Cinematic Chapter Adapter | Selects source boundaries, continuity evidence, event ranges and passages supporting review findings. |",
        "| Extract Chapter References | Selects occurrence and trait evidence from initialState/event and updates an ordered chapter reference dictionary. |",
        "| Consolidate References | Selects and audits cross-chapter identity evidence and reviews scene-supported landmarks against approved appearance. With character profiles, selects evidence for casting, temporary states and chronology. |",
        "| Generate H3 Prompts | Consumes resolved reference evidence and existing source pointers. Dialogue and prompts are authored content. |",
        "| Spatial Continuity | Uses existing sequence-level evidence addresses for staging and camera geometry. |",
        "| Load Character Evidence | Loads resolved quotations and their source locations. |",
        "| Load Character Profiles | Loads resolved evidence references and chronology. |",
        "| Load Character Visual Profiles | Loads resolved visual facts and their links to character evidence. |",
        "| Load Cinematic Chapters | Loads resolved source excerpts and editable adaptation choices. |",
        "| Load Chapter Catalogs | Loads and validates resolved local evidence. |",
        "| Load Consolidated References | Loads consolidated references and their resolved evidence. |",
        "| Load Spatial Continuity | Loads staging and sequence-level evidence addresses. |",
        "| Select Chapters | Selects inputs. |",
        "| Provider Configuration | Supplies model settings. |",
        "",
        "`source_text` creates request-scoped passage IDs with exact text and character offsets. Character evidence, casting and chronology, adaptation planning and review, and both levels of reference identity reconciliation resolve selected passages in Python. Character identity and trait decisions use evidence record IDs. Saved citations preserve exact quotations and provenance. Request caches bind selections to their inputs and passage contract.",
        "The adapter estimates each action, utterance or reaction separately in a single structured response per source window. Python packs the timed beats into sequences under the eight-second ceiling, reconstructs exact source spans, applies state changes and inherits each ending. A batch semantic audit cites original passages and triggers bounded corrections. Source-bound checkpoints resume committed windows, and the sidecar records incomplete checks and performance.",
        "`narrative_validation` reviews adaptation fidelity and transitions before targeted sequence repairs. When a full review and its focused fidelity check exhaust their output budget, indexed event groups must verify full coverage. Reference extraction reads `initialState` followed by `event` with their resolved names. Each request selects local evidence and can reuse an ID from the chapter dictionary. New objects and locations require a local visual detail. The dictionary grows from validated, independently cached extraction requests. `chapter_reference_identity` audits provisional identities and possible duplicates using bounded passage evidence. Recoverable failures leave unconfirmed identities separate with incomplete diagnostics. Cross-chapter reconciliation and optional whole-book character profile matching remain in consolidation.",
        "The adapter reviews the exact exported fields, including a separate narration-language verdict for each field and the evidence needed by reference extraction. Reviews compare the current sequence with the committed predecessor and use following source to check the boundary. Repairs preserve untouched fields for field-scoped findings and recheck the resulting scene before commit. Unresolved source ambiguity stays in a diagnostic sidecar. Editable defaults receive their own review. Node-local per-chapter checkpoints preserve drafts, accepted scenes, approvals and learned window sizes for resume_run into a new timestamped run.",
        "",
        "`reference_timeline` validates cinematic catalogs, merges observations without rewriting them, computes first sequence occurrences and provides exact source-observation lookup. Identity reconciliation and duplicate audits preserve temporal observations separately. `reference_visuals` compiles common intrinsic appearance and scene-local visual traits with explicit presence. A cached structured review groups confirmed equivalent intrinsic wording without rewriting source observations. Conflicting descriptions remain local with incomplete diagnostics; local states are never carried through unobserved scenes.",
        "Chapter reference catalogs store initialState/event observations in `state_by_sequence[sequence].source`. Each local ID maps to a consolidated entity and retains its sequence addresses. Object references use intrinsic appearance and material. Character references combine sequence wardrobe with independent physical profiles. Location references contain durable scenery and sequence-specific backdrop conditions. Equal appearances share assets with explicit `scene_bindings`, and H3 selects the assets for its exact sequence. Actions and narrative orientation remain in adaptation. Chapter and consolidated contracts are v5 with timeline revision `cinematic-neutral-references.v2`. Older outputs require regeneration. See [neutral visual references](neutral_visual_references.md) for the contract and validation cases. `editable_schemas` validates and exports the visual design schema. Extraction exports the chapter catalog schema.",
        "",
        "## Internal module dependencies",
        "",
        "Only modules with incoming or outgoing internal dependency edges are shown. Labels include registered ComfyUI node names.",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    connected = {module.name for module in modules if module.dependencies}
    connected.update(dependency for module in modules for dependency in module.dependencies)
    for module in modules:
        if module.name in connected:
            label = "<br/>".join(_mermaid_label(name) for name in [module.name, *(label for _, label in module.comfyui_nodes)])
            lines.append(f'    {_mermaid_id(module.name)}["{label}"]')
    for module in modules:
        for dependency in module.dependencies:
            lines.append(f"    {_mermaid_id(module.name)} --> {_mermaid_id(dependency)}")
    lines.extend(["```", "", "## Module inventory", "", "| Module | Internal dependencies | Public classes/functions |", "|---|---|---|"])
    for module in modules:
        dependencies = ", ".join(f"`{name}`" for name in module.dependencies) or "—"
        symbols = ", ".join(f"`{name}`" for name in module.public_symbols) or "—"
        lines.append(f"| `{module.path}` | {dependencies} | {symbols} |")
    lines.extend(["", f"Modules: **{len(modules)}**. Internal dependency edges: **{sum(len(module.dependencies) for module in modules)}**.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed architecture map is stale.")
    args = parser.parse_args()
    rendered = render_architecture(scan_modules())
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            parser.error("docs/architecture.md is stale; run python tools/generate_architecture.py")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
