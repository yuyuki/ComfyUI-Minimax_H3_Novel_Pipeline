# MiniMax H3 Novel Pipeline

ComfyUI nodes that extract novel reference catalogs, consolidate characters,
locations and objects across chapters, and generate MiniMax H3 scene prompts.
All language-model work runs through LM Studio's OpenAI-compatible API.
The nodes produce text, JSON and media briefs; images and audio are generated
or loaded separately in ComfyUI.

See [CHANGELOG.md](CHANGELOG.md) for the summarized commit history.

The registry publishing workflow sends this file as the new version's
`changelog` using Comfy CLI's `--changelog-file` option. To release, update the
changelog and bump `project.version` in `pyproject.toml`, then push to `main`.
Each release receives the complete changelog snapshot; previously published
versions are not updated.

## Cinematic Chapter Adapter

Connect **Select Chapters â†’ Cinematic Chapter Adapter** and connect
**LM Studio Configuration** to the adapter. Queue this standalone output node
to adapt the selected chapters using the Qwen or Mistral model selected by the
shared configuration. No image/video model is loaded or configured by this node.

Each chapter is saved to
`output/minimax_h3_novel/<timestamp>/cinematic_chapters/NNN_<chapter>.cinematic.json`
as a chapter object:
`{ "chapter_name": "chapter", "sequences": [{ "sequence": 1, "source": "...", "adaptation": { "initialState": "...", "event": "...", "endingState": "..." } }] }`.
`chapter_name` is the source filename without its extension and appears once at
the top level. Numbering restarts at 1 for each chapter. `source` preserves the text returned
by the existing chapter reader (PDF text extraction and whitespace cleanup still
apply). `adaptation` contains three strings: `initialState`, `event` (all ordered
events, numbered and separated by newlines), and `endingState`.
Descriptions stay in the source language if supported
by the LM Studio model; the prompt requires dialogue to remain verbatim.

The adapter requests cinematic normalization, preserving story, visual details,
identities, props, places and causality without excessive summarization. It
separates future actions from the initial state and passes the preceding final
state into the next passage. Source coverage and response fields are checked;
semantic fidelity and temporal correctness still require reviewing model output.
`chunk_chars` bounds each input passage; `max_tokens` controls its output budget.
If responses are truncated, reduce `chunk_chars` or increase `max_tokens`.
`correction_attempts` retries invalid fields or missing/rewritten source excerpts.
If validation still fails, passages longer than 1000 characters are automatically
split into smaller requests with the same correction budget per request. Recovery
preserves exact source coverage and carries final state through each smaller
passage; invalid responses at 1000 characters or fewer still stop the run.

`cinematic_chapters` returns a list of chapter records with `source_file`,
`saved_file`, `chapter_name` and `sequences`; `saved_files` lists the JSON paths for inspection.
A separate `cinematic_adapter_configuration.json` records non-secret run settings.
Connect `cinematic_chapters` to **Extract Chapter References**. Extraction no longer accepts
`chapter_selection`; reconnect existing workflows through the adapter.

To reuse saved adaptations, add **Load Cinematic Chapters**, set `cinematic_path`
to a previous run's folder (for example `20260911153042/cinematic_chapters`) or
one `*.cinematic.json` file, and connect its `cinematic_chapters` output to
Extract's `cinematic_chapters` input. Folder loading uses natural filename order
and validates the saved chapter structure. Paths are relative to
`output/minimax_h3_novel`; absolute paths must stay inside that root.
The loader needs no LM Studio configuration; keep configuration connected to Extract.

## Installation

Requires Python 3.10 or newer, ComfyUI, and an LM Studio server with a loaded
model. From your ComfyUI directory:

```sh
git clone https://github.com/yuyuki/minimax_h3_novel_pipeline.git custom_nodes/minimax_h3_novel_pipeline
python -m pip install -r custom_nodes/minimax_h3_novel_pipeline/requirements.txt
```

Use the Python interpreter that runs ComfyUI (including its embedded Python
when using a portable installation), then restart ComfyUI and refresh the
browser. Install the entire repository: the root `__init__.py`, `src/` and
`web/` directories are all needed for a source checkout.

Runtime dependencies are `openai>=1.0,<3`, `httpx>=0.27,<1` and `pypdf`.
The OpenAI SDK range preserves compatibility with the HTTPX transport used
by the nodes. PDF reading uses `pypdf`; text and Markdown do not need it.

## Architecture analysis

The generated [architecture map](docs/architecture.md) shows the three-stage
pipeline flow, internal Python-module dependencies and each module's public
classes and functions. It is built with Python's standard-library AST parser,
so it adds no ComfyUI runtime dependency or external service.

Regenerate it after changing imports or public symbols:

```sh
python tools/generate_architecture.py
```

CI runs the same command with `--check` and fails when the committed map is
stale.

## LM Studio setup

1. Start LM Studio's local API server and load a model.
2. In **ComfyUI Settings â†’ MiniMax H3 Novel â†’ LM Studio**, enter the API key.
   Use `lm-studio` if authentication is disabled. The current nodes read this
   setting; an environment-variable API-key selector is not exposed.
3. Add **LM Studio Configuration**. Its default URL is
   `http://127.0.0.1:1234/v1`. Choose `model_family`: **Qwen** (default) or **Mistral**. Load the matching
   model in LM Studio first; the dropdown does not load or download weights.
4. Connect its `lmstudio_config` output to Extract, Consolidate and Generate.

The API key is kept out of workflows and node outputs. ComfyUI's browser
settings store the value locally in plain text and send it to the backend
before queuing; the backend holds it in memory.

To authorize another LM Studio endpoint, set **Trusted API URL** beside **API Key**
in **ComfyUI Settings â†’ MiniMax H3 Novel â†’ LM Studio**, then enter the same URL
in the configuration node. The default is `http://127.0.0.1:1234/v1`.
The former `MINIMAX_H3_LMSTUDIO_BASE_URL` environment variable is no longer read;
copy any custom endpoint into this setting after updating.

A trailing slash is accepted. Authenticated requests disable redirects and environment
proxies. The chapter picker and settings endpoints require direct local
browser access to ComfyUI, such as `http://localhost:8188`; remote,
cross-origin and forwarded proxy requests are rejected.

All stages select the first model exposed by LM Studio whose identifier contains
`qwen` or `mistral`, according to `model_family`. A missing match raises an error;
there is no fallback to another family. If several models of the same family are
exposed, keep only the intended one available for an unambiguous selection.

For **Mistral Small 3.2 24B Instruct Q4_K_M**, select **Mistral**. Requests use
standard system/user messages and LM Studio's model template, with `top_p=0.9`
and one compact retry. `thinking` and all `qwen35_*` controls are ignored.
Temperature and `max_tokens` remain controlled by each processing node.
Existing workflows default to **Qwen**; restart ComfyUI and refresh the browser
to see the new dropdown.

Family-specific request settings live in `src/lmstudio_model_qwen.py` and
`src/lmstudio_model_mistral.py`. To add a family, implement the same profile
functions and register the module in `src/lmstudio_models.py`; its name appears
in the dropdown. Streaming, schema constraints, parsing and cancellation remain
shared in `src/lmstudio_json.py`. Qwen3.5-specific template recovery remains
limited to Qwen3.5 model identifiers.

All stages require LM Studio structured JSON output. Keep `thinking=false` for
extraction without reasoning overhead. For Qwen3.5, requests include an assistant
prefill containing a closed `<think>` block, in addition to `enable_thinking=false`.
This asks LM Studio to continue directly with JSON even when it ignores the template
keyword. If sampler initialization rejects `<think>` with an empty grammar stack,
the request retries once through `/v1/completions` with an explicit Qwen ChatML
prompt and a closed thinking block, keeping the JSON schema enabled. This bypasses
the server chat template; unrelated API errors still propagate. `thinking=true`
omits the prefill and does not use this recovery. This applies to all three stages and
their compact retries; structured output remains enabled. Configuration also exposes
output-token caps, compact retries, safe extraction chunk size and sampler
controls. Requests stream responses and check ComfyUI cancellation between
chunks.

## Workflow

Set `max_tokens` separately on Extract, Consolidate and Generate. Each node's
value is the output-token limit for all of its requests, including retries and
merges. LM Studio Configuration has no shared output-token limit. Existing
configuration nodes migrate the removed fields when loaded in the browser.
Extraction uses its own `chunk_chars`, with no shared Qwen chunk cap or hidden
3,000-character minimum. Paragraph boundaries and overlap still affect actual
chunk sizes.

Each streamed request logs the requested `thinking` setting, `content_chars`, `reasoning_chars`, `finish_reason`
and `local_stop` to the ComfyUI console. Counts are characters, not tokens;
reasoning counts sum string values in `reasoning_content` and `reasoning`.
No generated text is logged. `finish_reason=not_received` with
`local_stop=json_complete` means the client closed the stream as soon as the
JSON object completed, before receiving the server's final event.

```text
LM Studio Configuration â”€â”€â–º Extract / Consolidate / Generate
Select Chapters â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Extract / Generate
Extract Chapter References â†’ Consolidate References â†’ Generate H3 Prompts
```

For spatial continuity, turn on `enable_spatial_continuity` in **Generate H3 Prompts**
(off by default). Existing **Spatial Continuity** node connections also enable
the pass, even when the toggle is off; disconnect the node to disable it.
The existing
`visual_designs.json` choices arrive automatically through
the consolidated references and are reused for scene staging; no extra design
file needs to be supplied. The model's review is included in the existing chapter
`manifest.json` under `spatial_continuity` and in each scene's `continuity_review`.
Define layout constraints in the relevant location's `added_details.layout`
in `visual_designs.json`, then import that file through Consolidate before generating.
Using the configured LM Studio model, Generate plans missing film details against
the previous state and all later scenes in the chapter. After prompt generation
and camera refinement, it checks the actual prompt against earlier final prompts
and future source requirements, including positions, eyelines and camera axes.
Added staging must remain consistent even when the novel does not describe it.
It corrects contradictions with up to `repair_attempts` retries and rechecks each
correction. Unresolved issues appear in the continuity report and prompt validation
warnings. Cached prompts are also reviewed. This adds model requests and chapter
context; continuity currently covers scenes within each chapter. Review the actual
generated video separately: this pass checks text, not rendered images.

In **Generate H3 Prompts**, `duration` is the length of **each scene** in seconds.
`max_shots` limits camera shots inside that scene (default 1); the generator may
choose fewer and requires at least 2.5 seconds per shot. For example, 5 seconds
permits at most 2 shots even if `max_shots` is set to 3. Camera travel without a
cut remains a single shot. Use the optional `camera_direction` field to specify
the camera axis, subject placement, starting composition, movement and ending
composition. It is passed to the continuity pass when enabled and to prompt
generation and repair.

Enable `refine_camera` to run a final LM Studio camera pass after a scene prompt
passes validation. It adds one request per scene and inserts one concise camera
instruction inside each existing `[Shot N]`, preserving the other H3 sections,
events, dialogue and reference labels. Invalid camera notes are discarded with a
`camera_warnings` entry in the scene output; the original validated prompt is kept.
When a camera note fails its checks, Qwen receives the failed notes and specific
validation errors and may correct them up to `repair_attempts` additional times.
The final warning includes the last reason if every attempt fails.

| Node | Inputs and result |
|---|---|
| LM Studio Configuration | URL and Qwen controls â†’ shared non-secret configuration |
| Select Chapters | Chapter files or folder â†’ shared chapter selection |
| Extract Chapter References | Shared chapter selection â†’ chapter catalog list and summary |
| Load Chapter Catalogs | Saved `*_references.json` files â†’ chapter catalog list |
| Consolidate References | Catalogs â†’ registry with entities, picture briefs and audio briefs, plus a text summary |
| Load Consolidated References | Saved registry JSON â†’ registry object |
| Generate H3 Prompts | Registry and shared chapter selection â†’ chapter/scene prompt payload and save-ready text |
| Spatial Continuity | Compatibility node; new workflows can use Generate's `enable_spatial_continuity` toggle |

Add **Select Chapters**, then connect its `chapter_selection` output to both
Cinematic Chapter Adapter and Generate. Connect the adapter to Extract. Use its picker or enter one file/folder per line in its
`chapter_paths` field.
Cinematic Chapter Adapter, Extract, Consolidate and Generate are output nodes:
they can terminate a queued workflow without a separate Preview Text node.
After execution, each displays a read-only, copyable text preview: adapted chapter
JSON, catalog summary, registry summary, or scene and image prompts respectively.
Their existing output sockets remain available for downstream nodes.
Chapter paths must stay inside ComfyUI's input directory. Relative paths start
there, for example `minimax_h3_novel/chapter_01.txt`; copy external chapters
into that directory or upload them through the picker.
Supported files are `.txt`, `.md`, `.markdown` and `.pdf`. Folder discovery
is non-recursive and naturally sorted.

The three stages return Python dictionaries/lists and also write results to
their required `out_dir`. Every queued execution reserves one shared local-time
`yyyyMMddHHMMSS` folder under `output/minimax_h3_novel/`, even when settings are
unchanged. Defaults within that run are `chapter_catalogs`, `references` and
`h3_prompts`. Timestamp collisions advance to the next free second without
overwriting an earlier run. LM Studio Configuration shows the run folder in its status.
Each stage also writes `extract_configuration.json`, `consolidate_configuration.json`
or `generate_configuration.json` beside its results. These readable JSON snapshots
record every LM Studio Configuration control, the resolved model, effective model
controls and fallback policy, node settings (including defaults), input identifiers,
run folder and UTC timestamps. Completed snapshots list result files with SHA-256
hashes, so you can match settings to the files and detect later edits. API keys are
never included. Qwen controls remain visible in the shared configuration when using
Mistral; `model_controls` describes which controls actually apply.
Snapshots are execution records, not configuration files to edit and reload.
`status: started` means execution did not reach successful completion (including
failure or cancellation); only `completed` snapshots contain the final output hashes.
A repeated stage in the same output subfolder replaces its snapshot, just like its
results; use different `out_dir` subfolders for multiple instances of the same stage.
Cached results are included without changing cache behavior. These records describe
the pipeline's configuration, not LM Studio's server-side model loading settings or
a per-request trace of retries/fallbacks.
Consolidation writes `consolidated_references.json`, `reference_asset_prompts.txt`,
`visual_designs.json` and an `image_prompts/` export. Generate writes the same image
export alongside its existing chapter/scene files. Loader nodes reuse saved inputs;
new outputs always belong to the new run. Fresh runs do not reuse another run's disk caches.
`out_dir`, `catalog_path` and `consolidated_path` must stay inside
`output/minimax_h3_novel`. Stage `out_dir` paths are relative to the current run:
use `chapter_catalogs`, `references` or `h3_prompts`. Loader paths start at the plugin
output root: use `20260911153042/references/consolidated_references.json`, for example.
Existing in-root absolute stage paths become run-relative subfolders (a leading
previous-run timestamp is removed). Absolute loader paths retain their original meaning.
Parent traversal (`..`), Windows special paths and symlinks/junctions
that escape the root are rejected. Existing workflows pointing elsewhere must
move their files and update their paths. Outside ComfyUI, node helpers use
`input/` and `output/minimax_h3_novel/` beneath the startup working directory.

Generate or load the media described by the registry's briefs, then use the
desired entry from Generate's `prompts` payload with your MiniMax H3 video
node. H3 labels such as `<Picture 1>` are local to each request; several
views may refer to the same subject. The novel pipeline produces no video
references. See [examples/README.md](examples/README.md) for wiring instructions.

## Qwen-Image reference prompts and editable designs

Consolidate References has an `image_style` dropdown: **realistic photographic**
(default), cinematic photographic, digital illustration, anime, watercolor and
3D render. `image_asset_scope` defaults to **all entities**, including optional
characters, places and objects. Select **existing priority threshold** to use
`picture_threshold` instead. Audio continues to use its own threshold.
The default asset batch size is 4; existing workflows retain their saved value.

Each entity gets a separate file in `image_prompts/characters/`, `places/` or
`objects/`, named with its stable entity ID and name. Each file contains the source
description, clearly labeled **Added design details**, and a complete copy-paste
prompt for each generated base-reference angle. `image_prompts.json` contains
the same records. Generate's appended `image_prompt_text` output provides these
texts in ComfyUI; existing `prompts` and `prompt_text` sockets keep their positions.
The `prompts` dictionary also includes `image_prompts` records.

Each scene also has an `*_prompt.txt` sheet listing named characters, objects,
places, image views and filenames in upload order, followed by `COPY-PASTE PROMPT:`.
`all_prompts.txt` and the node's `prompt_text` output collect these sheets.
The structured scene entry keeps its raw `prompt_text` and adds `asset_sheet_text`.
Copy only the prompt below that heading
into H3; attach the listed media in order. Numbered H3 labels remain unchanged,
and generation requests canonical names alongside them. `*_assets.json` retains
the binding table and adds `copy_paste_prompt` and validation status. The readable sheet is the only scene prompt text file;
no separate `*_assets.txt` is generated. Rerun Generate to produce the new exports.

Each scene now uses one continuous shot lasting the full `duration`. Planning
splits successive actions and dialogue into separate scenes instead of merging
shots to fit a clip. `scenes_per_chunk` and `max_scenes` remain selection limits;
raise them to allow more scenes. Dialogue instructions align the speaker's voice,
mouth movement and expression, while continuity instructions preserve physical
contact and the source of effects. Rerun Generate to replan and regenerate prompts;
older planning and prompt caches are invalidated automatically.

Before assembling views, LM Studio condenses overlapping source descriptions,
feature lists and approved designs into a shared English appearance paragraph
per entity. It is instructed to merge repeated facts and
translations, omit biography and scene actions from neutral references, and
replace conflicting base clothing with chapter clothing. Each view reuses that
paragraph with its own framing and composition, in natural language suitable for
Qwen-Image-2512. This adds one LM Studio request per entity/state (plus any retries).
Source facts and editable design details remain intact. Semantic condensation
depends on the LM Studio model; review the resulting prompts for faithful details.
To refresh older repetitive prompts, rerun Consolidate using Load Chapter Catalogs,
then pass the new registry to Generate. Loading an old registry keeps its old prompts.

Missing visual details are designed once per entity and stored separately from
novel facts in `references/visual_designs.json`. To change them, edit only that
entity's `added_details` object, for example `"hair": "Short copper hair."`.
Set Consolidate's `visual_designs_path` to the edited file, relative to the plugin
output root, and queue again. Use Load Chapter Catalogs to avoid repeating extraction.
Imported additions replace the previous additions for that entity; `{}` removes
them. Entities omitted from the file receive a new design. IDs, names and types
must match the current registry. The current novel facts take precedence over the
file's informational `source_facts` snapshot. LM Studio checks additions for
contradictions and reports conflicting traits for correction; this semantic check
still requires human review. Chapter appearance takes precedence over a base design.

Consolidation also writes `reference_links.json`, `reference_links.schema.json`
and `visual_designs.schema.json`. Both editable JSON files declare a relative
`$schema`; keep each schema beside its document for editor completion and validation.
Runtime validation is offline and runs before model requests. Install the updated
runtime dependencies (`python -m pip install -r requirements.txt`) and restart ComfyUI.

To review references before generating briefs, enable Consolidate's `links_only`.
This exports classifications and suggested links, then blocks its registry output
so downstream generation does not run. Open `reference_links.json`, check the
entity names and evidence, and edit the decisions. Set `reference_links_path` to
that file (inside `output/minimax_h3_novel`), disable `links_only`, and queue again.
File-content changes invalidate the node cache even when the path is unchanged.

- `entities[].classification`: `entity` for real people (including unnamed guards),
  places and objects; `manifestation` for non-independent sounds, actions, sensations
  or descriptive fragments mistakenly extracted as entities of any type.
  Manifestations remain in the saved narrative/events but receive no reference assets.
- `identity` / `same_as`: two local mentions represent one entity of the same type.
  Set `sequence` and `phase` to `null`; the target supplies the retained identity name.
- `attribution`: attach a manifestation at an exact sequence/phase to a person,
  place or object. Use `emitted_by` for a cry, `describes` for a descriptive aspect,
  or another precise relationship. Its observation is retained without turning
  the fragment into an alias or permanent visual/vocal trait.
- `relation`: a narrative relationship such as `assistant_of` or `located_in`.
  Supply sequence/phase for temporary facts, or two `null` values for persistent
  relationships. This does not merge entities or change their stable appearance.

Generated links are `proposed` or `unresolved`. Only `confirmed` links are applied.
Extraction receives earlier reference names, aliases and last observations to resolve
contextual mentions without seeing future phases. Review missed matches in `links_only`:
use identity links for the same man, crevasse or rope under different names, and
attribution links for descriptions of the crevasse's wall or abyss. These decisions
depend on the passage; similar words alone do not establish identity.
`rejected` identity links prevent automatic merging; other rejected links are not
applied. Ambiguous initial cries should remain unresolved when several speakers
are possible. Classification is applied independently of link status: restore
`entity` if the model incorrectly classified a real person as a manifestation.
Confirmed links and rejected identities protect their entities from automatic
matching/auditing; other entities still follow normal consolidation. Review any
remaining duplicates explicitly. Confirmed relations and attributions are passed
to prompt generation with their chapter/sequence scope.

Addresses use `chapter_id` + `local_id`, never mutable global IDs. Every source
entity must appear exactly once in the file. The source fingerprint rejects files
from changed catalogs; regenerate links after re-extraction. Conflicting decisions,
unknown addresses and invalid scopes are errors. Raw chapter snapshots remain
unchanged; applied decisions and manifestations are saved separately in the registry.
Identity edits can change global IDs: regenerate visual designs after changing
identity links, then edit/import the new `visual_designs.json`. Imported visual
traits are restricted by entity type (for example `hair` for a character and
`layout` for a location). JSON schemas validate structure; semantic checks still
compare designs against current novel facts.

Style is applied when Consolidate generates briefs. Loading an existing registry
and running Generate exports its saved prompts without restyling or additional
image-prompt LLM calls. Older registries must be regenerated with the v4 timeline schema. Copy a single view's prompt into your separate Qwen-Image-2512
workflow. Text prompts alone cannot guarantee identical identity across independently
generated images; visually review the resulting references before binding them to H3.

Extraction caches now include prompt text, schemas and generation settings. Completed
catalogs missing the new fingerprint regenerate when processed directly; loaders can
still read them. Existing JSON retry/fallback and streaming cancellation behavior remain.

## Repository layout

```text
__init__.py                         ComfyUI checkout entrypoint
pyproject.toml                      Package, dependency and tool configuration
requirements.txt                    ComfyUI runtime dependencies
MANIFEST.in                         Source distribution contents
src/       Node implementations and bundled pipeline
web/js/minimax_h3_novel.js           Chapter picker and API-key settings UI
examples/                           Workflow instructions
tests/                              Offline regression tests
```

The bundled `pipeline_step1_extract.py`, `pipeline_step2_consolidate.py` and
`pipeline_step3_generate.py` are loaded relative to the Python package.
Source checkouts serve
`web/js`; built wheels include the same extension inside the Python package.

## Development and checks

From the repository root, in a virtual environment:

```sh
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
python -m build
```

Tests cover ComfyUI-style registration, frontend paths, installed-package
imports, bundled step loading, real SDK transport construction, credential
destination checks and local route access. They require no live LM Studio
or ComfyUI server. CI runs tests and lint on Python 3.10/3.12 on Linux and
Windows and builds source/wheel distributions.

For a live smoke test, restart ComfyUI, confirm all ten nodes appear under
**MiniMax H3 Novel**, upload a short chapter, configure LM Studio, and run
Extract â†’ Consolidate â†’ Generate. Check the saved JSON and confirm Stop
interrupts a running request.

License: [GNU GPL v3](LICENSE).

Registry releases use `.comfyignore` to omit tests and development tools while
retaining runtime code, browser assets, documentation and examples.
The publish workflow checks the uploaded version for up to roughly ten minutes.
Only an active, non-deprecated version passes; flagged releases include the registry's
findings and a manual-review issue draft in the Actions summary. The workflow does
not submit issues or grant approval. If scanning is still pending, rerun it with
`check_only` enabled to check the current `pyproject.toml` version without uploading
again. Locally, use `python tools/check_registry_status.py` with Python 3.11 or newer.
The trusted endpoint environment setting is intentional security configuration;
if it is flagged, request review rather than removing that validation.


Compact retries retain the original entity capacities, six distinguishing features of up to 120 characters each, and justified reference views. They allow 500 characters for stable visual descriptions and 350 for chapter appearance/state, shortening summaries and evidence instead. Retry-policy changes invalidate cached outputs. Extraction still selects continuity-relevant entities within the passage schema's limits (6 characters, 4 locations, 6 objects); reduce `chunk_chars` for crowded passages. These limits are character counts, not token counts, and unknown source traits remain unknown.

Consolidation audits registries above `audit_max_entities` using likely-duplicate clusters instead of skipping the audit. `audit_similarity` (0.68) and `audit_cluster_size` (24) control matching and batch size; `no_audit` still disables auditing. Clustering is heuristic and may miss duplicates across groups.

Only current v4 chapter catalogs and registries are accepted; consolidation requires cinematic phase timelines. Regenerate older outputs and recreate configuration nodes: the legacy backend selector was removed. The package contains only the ComfyUI pipeline; standalone CLI and fallback implementations are removed.

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
