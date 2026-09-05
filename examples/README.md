# Example workflow

1. Add **LM Studio Configuration**, enter the server URL/model, and set the
   API key in ComfyUI Settings → MiniMax H3 Novel → LM Studio.
2. Add **Extract Chapter References**, **Consolidate References** and
   **Generate H3 Prompts**. Connect configuration to all three.
3. Choose a chapter in Extract and Generate, or enter the same chapter paths
   in both. Connect Extract's `chapter_catalogs` to Consolidate, then
   Consolidate's `consolidated_references` to Generate.
4. Use the desired chapter and scene entry from Generate's `prompts` payload
   with your MiniMax H3 Reference to Video node. Generate/load the media from
   the registry's briefs and attach it in the entry's image/audio asset-ID order.

To resume, replace Extract with **Load Chapter Catalogs**, or replace
Extract and Consolidate with **Load Consolidated References**. Generation
still needs the original chapter text and LM Studio configuration.

All three stages save to their `out_dir`. The pipeline produces no video
references and does not generate images or audio itself.

Chapter paths are relative to ComfyUI's input directory, for example
`minimax_h3_novel/chapter_01.txt`. Upload or copy external chapters there.
Output and loader paths are relative to `output/minimax_h3_novel`: use
`chapter_catalogs` for extraction, `references` for consolidation and
`h3_prompts` for generation. Resume with `catalog_path=chapter_catalogs` or
`consolidated_path=references/consolidated_references.json`. Absolute paths
must stay within the corresponding root; `..` and links escaping it are rejected.

Extraction uses hierarchical merges (`merge_batch_size`, default 6) and caches each merge batch for resuming. This limits partial catalogs per call; dense catalogs can still require a larger context window. Enable `force` to regenerate cached results.

Consolidation audits registries above `audit_max_entities` using likely-duplicate clusters instead of skipping the audit. `audit_similarity` (0.68) and `audit_cluster_size` (24) control matching and batch size; `no_audit` still disables auditing. Clustering is heuristic and may miss duplicates across groups.

Only current v3 chapter catalogs and registries are accepted. Regenerate older outputs and recreate configuration nodes: the legacy backend selector was removed. The package contains only the ComfyUI pipeline; standalone CLI and fallback implementations are removed.
