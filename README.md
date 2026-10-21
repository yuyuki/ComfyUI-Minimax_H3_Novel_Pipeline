# MiniMax H3 Novel Pipeline

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
| **Spatial Continuity**           | Provides the compatibility continuity output; current workflows can use Generate's continuity toggle.        |

## Development and checks

From the repository root, in a virtual environment:

```sh
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
python -m build
```

License: [GNU GPL v3](LICENSE).
