# MiniMax H3 Novel Pipeline

ComfyUI nodes that extract novel reference catalogs, consolidate characters,
locations and objects across chapters, and generate MiniMax H3 scene prompts.
All language-model work runs through LM Studio's OpenAI-compatible API.
The nodes produce text, JSON and media briefs; images and audio are generated
or loaded separately in ComfyUI.

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

## LM Studio setup

1. Start LM Studio's local API server and load a model.
2. In **ComfyUI Settings → MiniMax H3 Novel → LM Studio**, enter the API key.
   Use `lm-studio` if authentication is disabled. The current nodes read this
   setting; an environment-variable API-key selector is not exposed.
3. Add **LM Studio Configuration**. Its default URL is
   `http://127.0.0.1:1234/v1`. The loaded model is selected automatically.
4. Connect its `lmstudio_config` output to Extract, Consolidate and Generate.

The API key is kept out of workflows and node outputs. ComfyUI's browser
settings store the value locally in plain text and send it to the backend
before queuing; the backend holds it in memory.

To authorize another LM Studio endpoint, set the following before starting
ComfyUI, then enter exactly the same URL in the configuration node:

```powershell
$env:MINIMAX_H3_LMSTUDIO_BASE_URL = "http://127.0.0.1:1235/v1"
```

The default endpoint remains trusted unless this variable is set. A trailing
slash is accepted. Authenticated requests disable redirects and environment
proxies. The chapter picker and settings endpoints require direct local
browser access to ComfyUI, such as `http://localhost:8188`; remote,
cross-origin and forwarded proxy requests are rejected.

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
LM Studio Configuration ──► Extract / Consolidate / Generate
Select Chapters ─────────► Extract / Generate
Extract Chapter References → Consolidate References → Generate H3 Prompts
```

| Node | Inputs and result |
|---|---|
| LM Studio Configuration | URL and Qwen controls → shared non-secret configuration |
| Select Chapters | Chapter files or folder → shared chapter selection |
| Extract Chapter References | Shared chapter selection → chapter catalog list and summary |
| Load Chapter Catalogs | Saved `*_references.json` files → chapter catalog list |
| Consolidate References | Catalogs → registry with entities, picture briefs and audio briefs |
| Load Consolidated References | Saved registry JSON → registry object |
| Generate H3 Prompts | Registry and shared chapter selection → chapter/scene prompt payload and save-ready text |

Add **Select Chapters**, then connect its `chapter_selection` output to both
Extract and Generate. Use its picker or enter one file/folder per line in its
`chapter_paths` field.
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
prompt for each generated angle and chapter variant. `image_prompts.json` contains
the same records. Generate's appended `image_prompt_text` output provides these
texts in ComfyUI; existing `prompts` and `prompt_text` sockets keep their positions.
The `prompts` dictionary also includes `image_prompts` records.

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

Style is applied when Consolidate generates briefs. Loading an existing registry
and running Generate exports its saved prompts without restyling or additional
image-prompt LLM calls. Older v3 registries remain loadable and show no added design
details unless recorded. Copy a single view's prompt into your separate Qwen-Image-2512
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
external source/                    Historical reference bundles, not runtime code
```

The bundled `pipeline_step1_extract.py`, `pipeline_step2_consolidate.py` and
`pipeline_step3_generate.py` are loaded relative to the Python package.
They do not require the historical reference bundles. Source checkouts serve
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
Windows and builds source/wheel distributions. Lint excludes historical
`external source/` bundles.

For a live smoke test, restart ComfyUI, confirm all seven nodes appear under
**MiniMax H3 Novel**, upload a short chapter, configure LM Studio, and run
Extract → Consolidate → Generate. Check the saved JSON and confirm Stop
interrupts a running request.

License: [GNU GPL v3](LICENSE).

Extraction uses hierarchical merges (`merge_batch_size`, default 6) and caches each merge batch for resuming. This limits partial catalogs per call; dense catalogs can still require a larger context window. Enable `force` to regenerate cached results.

Consolidation audits registries above `audit_max_entities` using likely-duplicate clusters instead of skipping the audit. `audit_similarity` (0.68) and `audit_cluster_size` (24) control matching and batch size; `no_audit` still disables auditing. Clustering is heuristic and may miss duplicates across groups.

Only current v3 chapter catalogs and registries are accepted. Regenerate older outputs and recreate configuration nodes: the legacy backend selector was removed. The package contains only the ComfyUI pipeline; standalone CLI and fallback implementations are removed.
