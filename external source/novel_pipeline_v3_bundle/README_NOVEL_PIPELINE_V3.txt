NOVEL PIPELINE V3 — scalable visual references + lore knowledge graph + H3

Requirements
------------
pip install openai pypdf

Keep novel_pipeline_common.py in the same directory as the 01L/02L lore scripts.
LM Studio default API URL: http://127.0.0.1:1234/v1

Recommended source layout
-------------------------
Use one chapter per .txt/.md/.pdf file. Wildcards * and ? are expanded by the scripts
and all matched inputs are processed case-insensitively in alphabetical order.

A) VISUAL REFERENCE PIPELINE (scalable v3)
------------------------------------------
1. Extract chapter visual references with bounded hierarchical chapter merging:

python 01_extract_chapter_references_v3.0.0.py "chapters/*.md" --out-dir chapter_references

Useful option:
  --merge-batch-size 6

2. Consolidate global visual registry. Small registries use the old whole-registry
audit; large registries automatically use bounded likely-duplicate clusters:

python 02_consolidate_references_v3.0.0.py "chapter_references/*_references.json" --out consolidated_references.json --asset-prompts-out reference_asset_prompts.txt

Useful options:
  --audit-max-entities 120
  --audit-similarity 0.68
  --audit-cluster-size 24

B) LORE / KNOWLEDGE-GRAPH PIPELINE
----------------------------------
1L. Extract chapter lore in chunks and recursively merge bounded batches:

python 01L_extract_chapter_lore_v1.0.0.py "chapters/*.md" --out-dir chapter_lore

Defaults:
  --chunk-chars 7000
  --merge-batch-size 6

Each chapter output includes:
  entities
  atomic facts
  durable relationships
  events
  terminology
  short evidence anchors
  temporal/state information

2L. Reconcile chapter entities into a global lore registry, aggregate atomic facts and
relationships, preserve provenance, and run a scalable clustered duplicate audit:

python 02L_consolidate_lore_v1.0.0.py "chapter_lore/*_lore.json" --out consolidated_lore.json

Important output sections:
  entities
  facts
  relationships
  events
  terminology
  chapter_summaries
  unresolved_mentions
  source_chapters
  statistics

C) H3 GENERATION WITH OPTIONAL LORE
------------------------------------
Without lore (works like v2.6, plus v3 compatibility):

python 03_generate_h3_prompts_v3.0.0.py "chapters/*.md" --references consolidated_references.json --out-dir h3_prompts --duration 8

With lore enrichment:

python 03_generate_h3_prompts_v3.0.0.py "chapters/*.md" --references consolidated_references.json --lore consolidated_lore.json --out-dir h3_prompts --duration 8

The H3 script injects a bounded source-derived chapter lore slice for identity,
relationship and world-continuity context. The source excerpt remains authoritative for
visible action, dialogue and timing. Default lore-context cap:
  --lore-context-chars 12000

SCALABILITY NOTES
-----------------
* 01 visual v3 no longer performs one potentially huge final chapter merge. It recursively
  merges batches of partial catalogs.
* 01L lore uses the same bounded hierarchical merge strategy.
* 02 visual v3 no longer disables duplicate auditing when the registry exceeds 120
  entities. It switches to bounded similarity clusters.
* 02L lore builds a global entity registry and rewrites graph references after audited
  entity merges.
* Evidence is deliberately stored as short anchors rather than large quotations.
* For multi-volume series, keep chapter filenames alphabetically sortable (e.g.
  01_001..., 01_002..., 02_001...) so processing chronology is deterministic.

FILES
-----
novel_pipeline_common.py
01_extract_chapter_references_v3.0.0.py
02_consolidate_references_v3.0.0.py
01L_extract_chapter_lore_v1.0.0.py
02L_consolidate_lore_v1.0.0.py
03_generate_h3_prompts_v3.0.0.py
README_NOVEL_PIPELINE_V3.txt
