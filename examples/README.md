# Example workflow

1. Add **LM Studio Configuration**, enter the server URL, and set the
   matching **Trusted API URL** and **API Key** in
   ComfyUI Settings → MiniMax H3 Novel → LM Studio.
2. Add **Select Chapters**, **Extract Chapter References**, **Consolidate
   References** and **Generate H3 Prompts**. Connect configuration to all three.
   Optionally connect **Spatial Continuity** to **Generate H3 Prompts** and enter
   fixed geometry in its JSON field (for example `{"tablet.wall":"right wall"}`).
3. Choose the chapters once in Select Chapters, then connect its
   `chapter_selection` output to Extract and Generate. Connect Extract's
   `chapter_catalogs` to Consolidate, then
   Consolidate's `consolidated_references` to Generate. Optionally connect
   Consolidate's `registry_summary` to a Preview Text node for chapter, entity
   and asset-brief counts.
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
`qwen35_safe_chunk_chars` field is removed. Paragraphs and overlap are preserved.
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

Extraction uses hierarchical merges (`merge_batch_size`, default 2) and caches each merge batch for resuming. The default `max_tokens` is 8192 per extraction/merge call. Update these controls in existing workflows to adopt the new defaults. The final catalog must still fit the output budget; dense catalogs may need more output tokens and a larger context window. For crowded passages, reduce `chunk_chars` to avoid the per-passage limits of 6 characters, 4 locations and 6 objects. Enable `force` to regenerate cached results.

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

For each entity, the text export labels source description and added design details
separately, followed by individual copy-paste prompts for each view and variant.
`image_prompts.json` contains asset IDs, descriptions and prompts for automation.
Style/design choices do not modify extracted novel facts. Review generated images
for consistency before using them as H3 references.

Consolidation audits registries above `audit_max_entities` using likely-duplicate clusters instead of skipping the audit. `audit_similarity` (0.68) and `audit_cluster_size` (24) control matching and batch size; `no_audit` still disables auditing. Clustering is heuristic and may miss duplicates across groups.

Only current v3 chapter catalogs and registries are accepted. Regenerate older outputs and recreate configuration nodes: the legacy backend selector was removed. The package contains only the ComfyUI pipeline; standalone CLI and fallback implementations are removed.

Generate gives each scene one continuous shot lasting the full `duration`.
Increase `scenes_per_chunk` (and `max_scenes` if capped) to allow more separate
moments. Rerun Generate to replace earlier compressed scene plans and prompts.
