# Example workflow

Node implementations live in `src/nodes/`; shared pipeline code and schemas live
in `src/core/`. This internal organization preserves node IDs and socket types,
so the example workflow uses the same wiring.

To try the standalone **Cinematic Chapter Adapter**, connect **Select Chapters**
to its `chapter_selection` input and **LM Studio Configuration** to its
`lmstudio_config` input, then queue. The adapter is an output node, so it needs
no downstream connection. It saves one `NNN_<chapter>.cinematic.json` object per
chapter under the run's `cinematic_chapters/` directory, with top-level
`chapter_name` (the source filename without its extension) and `sequences`.
Each sequence has `sequence`, `source`,
and an `adaptation` object containing `initialState`,
`event`, and `endingState` strings. `event` preserves all ordered events, numbered
and separated by newlines. Its `saved_files` output lists the files;
`cinematic_chapters` exposes the chapter records and sequence arrays in memory.
Review the generated adaptations, then connect `cinematic_chapters` to
**Extract Chapter References** (replacing its former `chapter_selection` input).

To resume from those saved adaptations, replace the adapter with **Load Cinematic
Chapters**. Set `cinematic_path=20260911153042/cinematic_chapters` (using your
previous run's timestamp), or select a single `*.cinematic.json` file in that
folder. Connect the loader's `cinematic_chapters` output to Extract's
`cinematic_chapters` input and keep LM Studio Configuration connected to Extract.
The loader validates saved chapters and loads folders in natural filename order.
Its paths start at `output/minimax_h3_novel` and must stay within that root.

1. Add **LM Studio Configuration**, enter the server URL, and set the
   matching **Trusted API URL** and **API Key** in
   ComfyUI Settings â†’ MiniMax H3 Novel â†’ LM Studio.
2. Add **Select Chapters**, **Extract Chapter References**, **Consolidate
   References** and **Generate H3 Prompts**. Connect configuration to all three.
   Optionally turn on `enable_spatial_continuity` in **Generate H3 Prompts** to enable
   automatic staging and model checks of consistency between scenes. No JSON input
   is needed. Define layout constraints to the location's `added_details.layout` in `visual_designs.json` and
   import that file through Consolidate.
   Generate checks final prompts after camera refinement, corrects contradictions
   within `repair_attempts`, and records unresolved issues as validation warnings.
   The existing visual designs are carried by `consolidated_references` and reused
   automatically. Review `spatial_continuity` in the chapter's existing
   `manifest.json` for inferred staging and prompt checks; no extra JSON file
   is created by the toggle alone.
   For a reusable chapter-wide plan, connect **Consolidate References** and
   **LM Studio Configuration** to **Spatial Continuity**, then its
   `spatial_continuity` output to **Generate H3 Prompts**. This exports
   `spatial_continuity.json` and provides a `spatial_json` text output.
   The node also copies `spatial_continuity.schema.json` beside it and adds a
   relative `$schema` link for editor validation. Keep both files together when
   editing a copy. Load Spatial Continuity validates against the bundled schema.
   Each sequence is planned with the entire chapter visible, including distant
   future constraints, initial/final entity positions, camera, evidence and warnings.
   Generate uses this plan during scene planning and prompt generation, and reviews
   final prompts even with the toggle off. Rerun Spatial Continuity after consolidation changes.
   To edit the plan manually, save a corrected JSON file inside
   `output/minimax_h3_novel` (renamed copies are supported). Add **Load Spatial
   Continuity**, set `spatial_path` to that file or its folder, and connect the
   same `consolidated_references` used by Generate. Connect the loader's
   `spatial_continuity` output to Generate instead of the planning node.
   The loader preserves edits, validates the v2 structure and source digest,
   and reloads changed file contents on the next execution without LM Studio.
   Keep chapter/sequence IDs and `source_digest` intact; edit positions, camera,
   states and evidence. A folder path looks for `spatial_continuity.json`.
   Set `duration` in Generate H3 Prompts to the length of each scene. Set
   `max_shots` to your desired limit (1 by default; a shot needs at least 2.5s).
   Optionally describe the camera axis and movement in `camera_direction`.
   Enable `refine_camera` for a final LM Studio camera-only pass after prompt
   validation. Check `camera_warnings` if the extra description is rejected.
   `repair_attempts` also limits camera-only correction retries after a rejection.
3. Choose the chapters once in Select Chapters, then connect its
   `chapter_selection` output to Cinematic Chapter Adapter and Generate. Connect the
   adapter’s `cinematic_chapters` output to Extract. Share configuration with the adapter. Connect Extract's
   `chapter_catalogs` to Consolidate, then
   Consolidate's `consolidated_references` to Generate. Each of these four processing
   nodes can terminate the workflow and displays its own text preview after execution;
   no separate Preview Text node is required.
4. In Consolidate, choose `image_style` (default **realistic photographic**) and
   reference image prompts include every consolidated character, place and object.
   Use `asset_batch_size=4` as the initial setting for the Qwen3.5 9B model.
5. Queue the workflow. Open the new timestamp folder shown in configuration status.
   Copy each view's prompt from `references/image_prompts/` into your Qwen-Image-2512
   workflow, or use Generate's `image_prompt_text` output. The same export is saved
   under `h3_prompts/image_prompts/`. Generate and review one image per view.
   Consolidate merges repeated facts into shared English appearance prose before
   assembling the views. To fix prompts from an older run, use Load Chapter Catalogs
   and rerun Consolidate, then Generate with the new registry.
6. Use the desired chapter and scene entry from Generate's `prompts` payload
   with your MiniMax H3 Reference to Video node. Generate/load the media from
   the registry's briefs and attach it in the entry's image/audio asset-ID order.
   Alternatively, open the scene's `*_prompt.txt` (or chapter `all_prompts.txt`),
   attach the named references in the listed order, and copy the text below
   `COPY-PASTE PROMPT:` into H3. Check any validation warnings before use.

To resume, replace Extract with **Load Chapter Catalogs**, or replace
Extract and Consolidate with **Load Consolidated References**. Generation
still needs the original chapter text and LM Studio configuration.

All three stages save to their `out_dir`. The pipeline produces no video
references and does not generate images or audio itself.

Set each stage's output budget with its own `max_tokens`; configuration no
longer applies a second cap. Old configuration widgets migrate on workflow
load. Set extraction passage size with Extract's `chunk_chars`; the shared
`qwen35_safe_chunk_chars` field is removed. Paragraph order is preserved within each phase; phases never overlap.

Optional `default_temporal_context` on **Extract Chapter References** accepts a date
or era, such as `1925` or `France, late 19th century`. Leave it empty when unknown.
Explicit narrative timing and flashbacks take precedence. This is user context,
not source evidence. It is saved in chapter catalogs and consolidated timelines,
preserved by the loaders, and used for missing visual design details and H3 prompts.
After changing it, rerun extraction, consolidation and generation; extraction caches
are invalidated automatically.
Restart ComfyUI and refresh the browser after updating the node package.
For JSON failures, check the ComfyUI console's `LLM stream` lines for character
counts, `finish_reason` and `local_stop`. These diagnostics omit generated text.

Chapter paths are relative to ComfyUI's input directory, for example
`minimax_h3_novel/chapter_01.txt`. Upload or copy external chapters there.
Every queue execution creates a shared `yyyyMMddHHMMSS` folder, using local time,
under `output/minimax_h3_novel`. Output subfolders are relative to that run: use
`chapter_catalogs` for extraction, `references` for consolidation and
`h3_prompts` for generation. Loader paths include the previous timestamp: for example,
`catalog_path=20260911153042/chapter_catalogs` or
`consolidated_path=20260911153042/references/consolidated_references.json`. Absolute paths
must stay within the corresponding root; `..` and links escaping it are rejected.

To compare settings with results, open `extract_configuration.json` in
`chapter_catalogs`, `consolidate_configuration.json` in `references`, or
`generate_configuration.json` in `h3_prompts`. Each snapshot includes the shared
LM Studio controls, resolved model, node settings, inputs and run timestamps.
When `status` is `completed`, `outputs` lists result paths relative to the snapshot
and SHA-256 hashes. `started` indicates an incomplete execution. API keys are omitted.
Change settings in the nodes; these JSON files are records, not editable presets.


Compact retries preserve entity capacities and visual features, allowing 500 characters for stable descriptions and 350 for chapter appearance/state. Cached results from the older retry policy are regenerated automatically.

To edit invented appearance details, open the previous run's
`references/visual_designs.json`. Each entity has its ID, name, type, a source-facts
snapshot and an editable `added_details` object:

```json
"added_details": {
  "hair": "Short copper hair.",
  "default_outfit": "A plain charcoal linen tunic."
}
```

Change or remove traits in that object; leave IDs, names and types intact. Use
Load Chapter Catalogs for the same novel, set Consolidate's `visual_designs_path`
to that edited file, and queue again. Outputs go into a fresh run. Imported designs
are checked against the current source facts; contradictions must be corrected.
An empty object removes all additions for an entity. Omitted entities are designed
anew. The selected style affects all new reference prompts. Loading a consolidated
registry preserves its already generated style and prompts.

Edit `link_to_local_id` directly in the extracted chapter catalog, using the exported
`chapter_references.schema.json` for editor completion. Use another entry's `local_id`
only for the same entity, in the same chapter and category; use `null` for main entries
or uncertain identity. Targets must exist; self-links, chains and cycles are rejected.
Load the edited catalogs with Load Chapter Catalogs before running Consolidate.
Linked entries are grouped before model reconciliation and keep all source addresses
and phase observations. The model may reconcile these groups across chapters.
Sequences remain unchanged; no states are propagated between phases. Base image prompts
are prepared for every consolidated entity using stable identity facts only.
Regenerate visual designs when identity edits change global IDs.

For each entity, the text export labels source description and added design details
separately, followed by individual copy-paste prompts for each base-reference view.
`image_prompts.json` contains asset IDs, descriptions and prompts for automation.
Style/design choices do not modify extracted novel facts. Review generated images
for consistency before using them as H3 references.

Consolidation audits registries above `audit_max_entities` using likely-duplicate clusters instead of skipping the audit. `audit_similarity` (0.68) and `audit_cluster_size` (24) control matching and batch size; `no_audit` still disables auditing. Clustering is heuristic and may miss duplicates across groups.

Only current v4 chapter catalogs and registries are accepted; consolidation requires cinematic phase timelines. Regenerate older outputs and recreate configuration nodes: the legacy backend selector was removed. The package contains only the ComfyUI pipeline; standalone CLI and fallback implementations are removed.

Generate gives each scene one continuous shot lasting the full `duration`.
Increase `scenes_per_chunk` (and `max_scenes` if capped) to allow more separate
moments. Rerun Generate to replace earlier compressed scene plans and prompts.

## Cinematic preprocessing

Use **Cinematic Chapter Adapter** or **Load Cinematic Chapters** for the
cinematic extraction workflow. Generate uses selected source chapters and the
current consolidated registry, with optional spatial continuity.
Recreate older configuration nodes and use relative `out_dir` subfolders;
old widget layouts, narrative bundles and absolute output paths are unsupported.

### Extraction timeline contract

Each chapter is independent. Sequence IDs must be positive and strictly increasing
in the supplied array; extraction never sorts them. `initialState` is true at the
start, `event` describes changes during the sequence, and `endingState` is true
only after completion. Each phase is extracted separately without future text.
`source` is retained as supporting evidence, never used to override adaptation.

Catalogs retain the complete ordered `sequences`, plus each reference's
`first_sequence`, `first_phase`, and `state_by_sequence`. Each sequence entry maps
phase names to lists of observations (including descriptions, states and evidence).
Missing phases mean no observation, not absence or automatic state inheritance.
Repeated unambiguous names/aliases share a character or location identity; uncertain
identities stay separate. Objects reuse exact canonical names; shared generic aliases
never merge distinct physical instances. The local-link review can connect supported
synonyms. Object names must distinguish functions and attachments when needed.
Character-level `speaks` is true if any phase contains speech; individual phase
flags are preserved. Established aliases accumulate without rewriting earlier phases.
Changing state is never merged into an unqualified chapter appearance/state.
Consolidation separates global stable identity from each entity's
`timeline[chapter_id][sequence][phase]` observation lists. Identity reconciliation
and duplicate audits receive only stable fields; deterministic merges preserve
all observations and their evidence, even when aliases resolve to one identity.
`chapter_timelines` retains the original chapter records and ordered sequences.
Chapter IDs must be unique; sequence numbers are local to their chapter.

`first_occurrence_by_chapter` records each chapter's `first_sequence` and
`first_phase`, computed by numeric sequence and phase rank. Entity-level
`first_chapter_id`, `first_sequence`, and `first_phase` use supplied chapter order.
The Python `reference_timeline.state_at(entity, chapter_id, sequence, phase)` helper
returns only observations at that exact address (an empty list means unknown).
Read the preceding sequence's `endingState` separately for continuity; it never
overwrites the next explicit `initialState`. No state is propagated automatically.

Chapter-wide variations and the `no_variants` control are removed. Reference
image briefs use stable identity only, so later possession, disguise or damage
cannot become a chapter-wide image default. Sequence-specific image generation
requires a consumer that explicitly selects a temporal address. Regenerate old
catalogs and registries; no legacy format conversion is provided.

Extraction's `chunk_chars` divides paragraphs only within a phase. The obsolete
`overlap_paragraphs` and `merge_batch_size` controls are removed. Phase calls use
`max_tokens` and the shared JSON retries; `force` bypasses timeline caches.
Each uncached extraction is semantically reviewed against its phase and earlier
identity context, with at most two targeted corrections. Persistent doubts are
saved in `narrative_review.extraction_issues`; results and warnings are cached. Adapter reviews also check source detail
coverage and transitions; `correction_attempts` bounds targeted sequence repairs.
After bounded repairs/reconstruction, persistent doubts produce two or three full
scene alternatives in `narrative_review.issues[].proposals`, with the first
`selected: true`. No narrative disagreement blocks execution and there is no
`accepted` flag. Each proposal includes a reason and complete `adaptation` with
`initialState`, `event`, and `endingState`; the selected one governs the scene.
Set another proposal's `selected` to true and the others to false in the original
`.cinematic.json`, then reload with **Load Cinematic Chapters** and rerun extraction,
consolidation, spatial continuity (if used), and generation. This refreshes reference
observations, assets and prompts together. Missing selections default to the first;
multiple selections use the first selected proposal, with a console notice. File
changes are detected on the next run. Alternatives remain in saved review metadata
but only the resolved adaptation is sent to downstream narrative model calls.
Malformed data or technical failures still stop the workflow.
Reviews use additional LM Studio requests and remain model judgments, not a proof
of narrative fidelity. Regenerate adaptations and catalogs from previous runs.
Original-source Generate inputs remain unchanged; this change provides temporal
reference metadata, not a replacement for Generate's narrative continuity input.

### Generation from consolidated cinematic timelines

Generate H3 Prompts uses the registry's `chapter_timelines`, preserving sequence
order and `initialState` / `event` / `endingState` observations. Connect the same
Select Chapters output used by Cinematic Chapter Adapter; selected paths must
match the registry's source-file provenance. Chapter IDs come from the registry.
Missing or ambiguous matches require reconsolidating the intended chapters.
Generation does not reread or rechunk the raw prose. `scenes_per_chunk` limits
scenes per cinematic sequence; `max_scenes` limits scenes per chapter. Text chunk
size and paragraph-overlap controls have been removed from Generate.

Each saved scene records its sequence ID. Reference state is scoped to that exact
sequence and phase, with no automatic carry-forward. Stable visual designs and
image styles remain part of the consolidated assets and image-prompt export.


H3 reference generation allows at most 9 images and 3 audio references per scene.
The defaults are 8 images and 3 audio references; the per-subject image budget
cannot exceed 9 and remains subject to the total image budget. Audio references
are selected only when an image reference is available.
Scene duration is restricted to 4–15 seconds. The pipeline's 2.5-second minimum
shot length permits at most 6 shots at 15 seconds; shorter scenes automatically
use a lower shot cap. This pacing rule is internal, not an H3 shot-count limit.
max_scenes defaults to 0 (all scenes per chapter).
Temperature, token budget and repair attempts control LM Studio, not H3.
Actual reference audio files must each be 2–15 seconds and total no more than
15 seconds. This prompt-only pipeline cannot validate media duration or file size.
Limits: https://www.minimax.io/news/minimax-h3-open-source
