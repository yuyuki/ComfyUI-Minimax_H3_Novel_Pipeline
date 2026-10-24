# MiniMax H3 Novel Pipeline

Python code is organized into `src/nodes/` (the twelve ComfyUI nodes) and
`src/core/` (pipeline stages, LM Studio integration, shared helpers and JSON
schemas). `src/__init__.py` registers the nodes and local HTTP routes. The root
`__init__.py` remains the ComfyUI checkout entrypoint; browser assets stay in `web/`.

- [MiniMax H3 Novel Pipeline](#minimax-h3-novel-pipeline)
  - [Installation](#installation)
    - [LM Studio setup](#lm-studio-setup)
  - [Architecture analysis](#architecture-analysis)
  - [Development and checks](#development-and-checks)

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

### LM Studio setup

1. Start LM Studio's local API server and load the Qwen (default) or Mistral model you intend to use.
2. In **ComfyUI Settings → MiniMax H3 Novel → LM Studio**, enter the API key (`lm-studio` when authentication is disabled).
3. Add **LM Studio Configuration**, select the same model family, and use the default URL: `http://127.0.0.1:1234/v1`.
4. For a custom endpoint, set it as **Trusted API URL** in the same settings and enter that identical URL in the configuration node.
5. Connect `lmstudio_config` to Extract, Consolidate, and Generate; keep only the intended matching model available in LM Studio.


Set `thinking` on each LM Studio processing node: it defaults to `true` for
Cinematic Chapter Adapter, Extract Chapter References, Consolidate References
and Spatial Continuity, and `false` for Generate H3 Prompts. Mistral ignores
this Qwen control. LM Studio Configuration shares connection and sampling
settings only. Recreate existing nodes to refresh their widget layouts.

Optional `default_temporal_context` on **Extract Chapter References** accepts a date
or era, such as `1925` or `France, late 19th century`. Leave it empty when unknown.
Explicit narrative timing and flashbacks take precedence. This is user context,
not source evidence. It is saved in chapter catalogs and consolidated timelines,
preserved by the loaders, and used for missing visual design details and H3 prompts.
After changing it, rerun extraction, consolidation and generation; extraction caches
are invalidated automatically.

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

The registered custom nodes are:

| Custom node                      | Summary                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **LM Studio Configuration**      | Validates the LM Studio endpoint and shares non-secret model settings across processing nodes.               |
| **Select Chapters**              | Selects uploaded chapter files or folders and emits a reusable chapter selection.                            |
| **Cinematic Chapter Adapter**    | Converts selected chapters into saved cinematic chapter timelines for downstream extraction.                 |
| **Load Cinematic Chapters**      | Loads saved `.cinematic.json` chapter timelines from the output directory.                                   |
| **Extract Chapter References**   | Extracts chapter catalogs with entities, timelines, and source provenance using LM Studio.                   |
| **Load Chapter Catalogs**        | Loads saved `*_references.json` catalogs for consolidation.                                                  |
| **Consolidate References**       | Merges catalogs into a registry containing characters, locations, objects, picture briefs, and audio briefs. |
| **Load Consolidated References** | Loads a saved consolidated registry for prompt generation.                                                   |
| **Generate H3 Prompts**          | Generates chapter and scene prompt payloads plus save-ready image-prompt text from the registry.             |
| **Spatial Continuity**           | Plans all consolidated sequences with character, object, location and camera states; exports JSON and feeds Generate.        |
| **Load Spatial Continuity** | Loads a manually corrected spatial JSON, validates it against consolidation, and detects saved edits on the next run. |

## Development and checks

From the repository root, in a virtual environment:

```sh
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
python -m build
```

License: [GNU GPL v3](LICENSE).

Cinematic adaptation reviews source fidelity and state transitions with LM Studio.
Flagged sequences are corrected with the full passage source and neighboring
sequences as context, bounded by `correction_attempts`, then reviewed again.
If defects persist and `correction_attempts` is nonzero, one final reconstruction
adapts the entire passage as a single source-bound sequence and reviews it again.
Persistent narrative doubts never stop the run: the model supplies two or three
complete scene alternatives in `narrative_review.issues[].proposals`. Each issue
identifies its `sequence`, describes the doubts and quotes the source excerpt.
Each proposal has an `id`, `reason`, complete `adaptation` (initialState, event,
endingState), and `selected` boolean. The first proposal is selected automatically
and applied immediately. There is no `accepted` property. Multiple doubts in one
sequence share coherent scene alternatives rather than conflicting field edits.
Zero `correction_attempts` skips repairs, but still reviews and proposes alternatives.
Malformed semantic review entries are retained as doubts, including their original
content. An unusable sequence address applies the doubt to every sequence in the
passage. These advisory entries follow the same correction/proposal flow and never
count as approval or stop execution.

To choose another interpretation, edit `selected` in the saved `.cinematic.json`,
then run **Load Cinematic Chapters -> Extract -> Consolidate -> Spatial Continuity
(if used) -> Generate**. File edits invalidate the loader cache. Select exactly one
proposal per issue; no selection falls back to the first proposal, and multiple
selections retain the first selected one, with a console notice. Loaders normalize
in memory without overwriting your file. Edit a proposal's `adaptation` to customize
it; the selected proposal overrides the sequence's adaptation on loading.
Do not only edit downstream catalogs: re-extraction is needed to refresh entity
observations and reference assets after changing a scene.

Review metadata survives extraction and consolidation; only the selected scene
enters extraction, spatial planning and H3 narrative prompts. Extraction reviews
each phase with earlier identity context, allowing two corrections, then records
remaining doubts in `narrative_review.extraction_issues` without stopping. Results
and their warnings are cached. Connection, unreadable JSON, invalid adaptation/proposal schemas and filesystem
errors still block execution. These reviews add model calls. Regenerate adaptations and catalogs to apply the checks
to previous runs. Chapter-level `speaks` records any speech in the chapter;
phase observations retain their own speech flags.

Consolidation respects editable `link_to_local_id` decisions in extracted catalogs.
Load corrected files with **Load Chapter Catalogs**: links join same-type entries within
one chapter before cross-chapter reconciliation. All source IDs and `state_by_sequence`
observations are retained at their original chapter, sequence and phase; extracted
sequences are unchanged. Invalid targets, self-links, chains and cycles are rejected.
Reference image prompts cover every consolidated entity. See [examples](examples/README.md)
for catalog editing and visual design settings.

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

### Character Evidence Extractor

Connect **Select Chapters** (`chapter_selection`) and **LM Studio Configuration**
(`lmstudio_config`, model family **Qwen**) to **Character Evidence Extractor**.
Select every chapter file, or a folder containing them; each file is one chapter,
processed independently in natural filename order. TXT, Markdown and searchable
PDF inputs are supported. No character profiles are consolidated.

Outputs are a dictionary (`MINIMAX_CHARACTER_EVIDENCE`) and the saved JSON path
(`STRING`). `out_dir` defaults to `character_evidence` in the current run folder;
the complete editable output is `character_evidence.json`. Controls are
`chunk_chars`, `thinking`, `temperature`, `max_tokens`, `correction_attempts` and
`force`, plus `character_scope`:

- `proper_names` (default): scan every selected chapter for Unicode capitalized
  name candidates, index their occurrences throughout the novel, and batch nearby
  paragraphs with preceding/following context and consecutive pronoun paragraphs.
  Overlapping windows are merged. The model excludes places and other non-character
  entities; ambiguous aliases remain for the existing profile consolidation stage.
- `all_characters`: process the full text, including unnamed people and role labels.

Proper-name selection is a heuristic (with English/French sentence-start exclusions),
not language-independent entity recognition. It can miss unnamed descriptions before
a name is revealed, distant pronoun-only descriptions, or names without capitalization.
Use `all_characters` when that coverage matters. Both modes inspect all selected chapters;
neither samples only the beginning of the novel.

Completed batches are cached under `output/minimax_h3_novel/.cache/character_evidence`
and reused across run folders unless `force` is enabled. Changed batch text/context,
prompts, scope, model or settings invalidate the affected requests. Exact source offsets
are rebuilt from the current text, including when earlier text moves. Diagnostics in
`character_evidence_diagnostics.json` report elapsed/preprocessing time, selected text,
passages, batches, cache hits, extraction calls and transport requests including retries.
Input/output tokens are explicitly rough character-based estimates, not tokenizer usage.

Use **Load Character Evidence** with `evidence_path` to reopen this JSON after manual
edits, independently of the extractor and LM Studio. Paths must stay inside
`output/minimax_h3_novel`; relative paths start there. Its `character_evidence`
and `saved_file` outputs match the extractor. Saved edits reload on the next
workflow execution; loading preserves all data and never writes to the file.

Each observation records its character, attribute, value, explicit/inferred
classification, confidence, persistence, rationale, original-language quotation
and chapter/file provenance. Every quotation is checked as an exact substring.
Offsets are zero-based Unicode character indices with an exclusive end; lines
are one-based in the embedded `source_text`. UTF-8 whitespace and line endings
are retained (a UTF-8 BOM is excluded). PDF offsets refer to the embedded extracted
text, not PDF byte positions or visual page coordinates. Repeated quotation
matches retain all candidate locations; identical observations at the same location
are deduplicated across overlapping chunks. Conflicting observations remain separate.
Quotation verification establishes text fidelity, not the truth of a model's
interpretation: review inferences and confidence manually. Unsupported attributes
must be omitted. Invalid quotes/fields trigger bounded correction attempts;
persistent failures stop extraction with chapter/passage and retry guidance.


### Consolidate Character Profiles

Connect `character_evidence` from **Character Evidence Extractor** or **Load
Character Evidence**, and `lmstudio_config` from **LM Studio Configuration**
(model family **Qwen**). The node considers all supplied chapters together.
It returns `character_profiles` (`MINIMAX_CHARACTER_PROFILES`) and `saved_file`
(`STRING`), saving readable UTF-8 `character_profiles.json` under `out_dir`
(default `character_profiles`) in the current run folder.

Controls: `thinking`, `temperature`, `max_tokens`, `correction_attempts`, `force`.
The complete current input, model, prompt and settings determine the disk cache;
manual Load-node corrections take priority over any earlier extraction. Invalid
observations or quotation locations stop consolidation with correction guidance.
Correct them in the evidence JSON and reload. Existing warning metadata is retained.

Profiles retain every observation and original quotation/location, separate explicit
stable traits, temporary appearance, actions and inferred information, and list
unknown attribute categories. Identity aliases require explicit connecting source
quotations; ambiguous identities remain separate. IDs and ordering are deterministic
for the same evidence and grouping. The original input document is also embedded.
Differing values are conservatively flagged for review, even if they may coexist;
model-detected contradictions and explicitly supported temporal interpretations are
retained without deleting or overwriting observations. Chapter order alone is not
assumed to be story chronology. Quote validation proves fidelity, not semantic truth:
review model identity and chronology interpretations before using them as facts.

All evidence is sent together for cross-chapter reconciliation. For large novels,
configure enough LM Studio context and output tokens; failure never silently drops
chapters or truncates evidence. No profile loader, exported JSON Schema or cinematic
visual generation is provided by this node.

### Character Visual Profile Generator

Connect `character_profiles` from **Consolidate Character Profiles** and
`lmstudio_config` from **LM Studio Configuration** with Qwen. Outputs are
`character_visual_profiles` (`MINIMAX_CHARACTER_VISUAL_PROFILES`) and
`saved_file` (`STRING`). The node saves readable UTF-8
`character_visual_profiles.json` under `out_dir` (default
`character_visual_profiles`) in the current run.

Choose `strict_canon` for source-supported traits only, `balanced` (default) for
conservative contextual completion, or `cinematic` for restrained creative design.
Other controls are `thinking`, `temperature`, `max_tokens`,
`correction_attempts` and `force`. Each character is generated and reviewed with
all of their consolidated evidence. Shared structured JSON transport handles
streaming, cancellation and compact retries without truncating evidence.
Validated designs are cached per character, mode, input, model and settings;
`force` regenerates them.

Use **Load Character Visual Profiles** with `profiles_path` to reopen the saved
JSON, including renamed and manually edited copies. Paths must stay inside
`output/minimax_h3_novel`; relative paths start there. Its
`character_visual_profiles` and `saved_file` outputs match the generator.
Loading preserves descriptions, IDs, metadata and original evidence, detects
saved edits on the next execution, and never calls LM Studio or writes the file.

The seven fields `age_appearance`, `skin`, `hair`, `eyes`, `build`,
`default_outfit` and `accessories` contain lists of individual English traits
with `description`, `classification` (`canonical`, `inferred`, `creative`) and
`evidence` references. Empty lists mean unspecified. IDs, exact original-language
quotations, provenance, temporary observations, actions and conflicts are retained
unchanged, together with the complete input document. Temporary observations do
not establish habitual clothing or permanent physical traits.

Unresolved conflicts leave affected fields empty, listed in `blocked_fields`.
Consolidation uses broad physical categories, so a physical conflict conservatively
blocks all five physical fields; context conflicts can block all seven. No
chronological variant is silently selected. Structural validation enforces source
references and canonical coverage; a separate Qwen review checks English,
translation fidelity, contextual suitability and contradictions. This semantic
review remains model-based and should be checked by a human.

The saved JSON can be manually refined in any text editor. Keep trait provenance
accurate when editing. Rerunning writes the generated result again, so preserve
manual refinements to `character_visual_profiles.json` in a separate copy.
The separate `visual_designs.json` export retains edits as described below.

The Character Visual Profile Generator also saves a global `visual_designs.json`
containing every consolidated character, with stable IDs and aliases. Point
Consolidate References’ `visual_designs_path` at this file to reuse its editable
`added_details` for the selected chapter catalogs. Only matching chapter entities
receive designs; other novel characters are ignored. Matching uses exact normalized
names/aliases or stable character IDs, without LLM calls. Ambiguous matches fail
explicitly; unmatched chapter characters produce a warning and use normal design
generation. Temporary observations are excluded from the global export. Manual
`added_details` edits survive regeneration and are checked against current chapter
facts on import. The existing `saved_file` output still points to
`character_visual_profiles.json`.
