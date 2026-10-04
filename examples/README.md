# Example workflow

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
   is created. Existing **Spatial Continuity** connections also enable the pass,
   even with the toggle off; disconnect the node to disable it.
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
   keep `image_asset_scope=all entities` to include every character, place and object.
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

For a two-pass identity review, enable Consolidate's `links_only`. It saves
`references/reference_links.json` and both `*.schema.json` files, and blocks
downstream generation. Keep the schemas beside the editable files for completion
and error highlighting in a compatible JSON editor.

In `reference_links.json`, find the sound's address using the `entities` list.
The same review handles all contextual references: confirm `identity` / `same_as`
between “l'homme avec la torche” and the established person, between “la crevasse
à Delphes” and “la crevasse sombre”, or between “la corde” and “le filin”, when
the passage establishes one entity. Use null sequence/phase for identity links.
For “la paroi rocheuse” describing the crevasse, classify the fragment as
`manifestation` and confirm an `attribution` with `relation: "describes"` to the
crevasse at the fragment's exact sequence/phase. This preserves the description
on the location without creating a separate asset. Objects support the same flow.
Keep genuinely independent places/objects as `entity`; leave uncertain targets unresolved.

Keep `classification: "manifestation"` for a sound such as “les cris”; retain
`entity` for an actual unnamed person. For an attribution, edit the character
target and set `status` to `confirmed` only when you want to impose that decision:

```json
{
  "kind": "attribution",
  "source": {"chapter_id": "chapter", "local_id": "CHAR_002"},
  "target": {"chapter_id": "chapter", "local_id": "CHAR_003"},
  "sequence": 1,
  "phase": "event",
  "relation": "emitted_by",
  "status": "confirmed",
  "reason": "Attribution reviewed by the operator.",
  "evidence": ["Des cris retentissaient au-dessus de lui."]
}
```

These addresses are illustrative: use the generated IDs and actual evidence.
Leave uncertain authors `unresolved` (a `null` target is allowed); sounds still
produce no character asset. `identity` links use `same_as` and null sequence/phase.
Confirm to merge; reject to prevent a merge. `relation` links describe relationships
without merging their endpoints. Only confirmed links are applied, and the model
cannot undo manual decisions. Classification is applied even for unresolved links.

Set `reference_links_path` to the edited file, disable `links_only`, and rerun
Consolidate with the same catalogs. Schemas and source addresses are checked before
model calls. Regenerate links after changing catalogs; regenerate visual designs
after identity edits that change global IDs. Install the updated requirements and
restart ComfyUI to load the new inputs and JSON Schema validator.

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
Repeated unambiguous names/aliases share an identity; uncertain identities stay separate.
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
Original-source Generate inputs remain unchanged; this change provides temporal
reference metadata, not a replacement for Generate's narrative continuity input.
