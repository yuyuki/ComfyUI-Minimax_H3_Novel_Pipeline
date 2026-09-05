# AGENTS: MiniMax H3 Novel Pipeline

## Working rules

- Be concise; do not provide summaries unless explicitly requested.
- Follow user, repository and system instructions strictly. Do not silently
  weaken or bypass requirements.
- Ask for clarification only when necessary for safety or a material choice.
- Preserve existing user edits and the scaffold's `src/` layout.

## Current architecture

- Root `__init__.py` is the ComfyUI checkout entrypoint. It re-exports mappings
  from `src/minimax_h3_novel_pipeline` and serves `./web/js`.
- `src/minimax_h3_novel_pipeline/__init__.py` registers six nodes and four
  local HTTP routes. `nodes.py` exports the node classes.
- Each node has its own module: `lmstudio_config.py`,
  `extract_chapter_references.py`, `load_chapter_catalogs.py`,
  `consolidate_references.py`, `load_consolidated_references.py`,
  and `generate_h3_prompts.py`.
- `lmstudio_pipeline.py` dynamically loads bundled scripts relative to its
  own directory, configures Qwen behavior and wraps streaming cancellation.
- `util.py` contains file discovery, JSON persistence and shared helpers.
- `lmstudio_settings.py` stores the API key in memory and validates the
  trusted endpoint. `route_access.py` enforces direct local browser access.
- `web/js/minimax_h3_novel.js` implements chapter picking and settings.
- `external source/` contains historical reference bundles. Do not import
  them at runtime or edit them as a substitute for changing bundled code.

## Pipeline entry points

All paths below are relative to `src/minimax_h3_novel_pipeline/`:

| Stage | Node wrapper | Bundled implementation |
|---|---|---|
| Extract | `extract_chapter_references.py` | `pipeline_step1_extract.py` |
| Consolidate | `consolidate_references.py` | `pipeline_step2_consolidate.py` |
| Generate | `generate_h3_prompts.py` | `pipeline_step3_generate.py` |

Keep node wrappers small. Reuse bundled schema checks, JSON parsing,
streaming, compact retries and Qwen3.5 structured-JSON/ChatML fallbacks.
Preserve existing node IDs and socket types for workflow compatibility.

## Behavior and security constraints

- All language-model work uses LM Studio's OpenAI-compatible API. Never add
  CLIP inputs, model loading or CLIP-dependent execution paths.
- Share non-secret URL, model and backend controls through
  `LMStudioConfigurationNode` and `MINIMAX_LMSTUDIO_CONFIG`.
- Credentials may come only from ComfyUI settings or the runtime environment,
  never workflow inputs. The current node implementation uses ComfyUI settings;
  do not document an environment-key selector unless it is implemented.
- Keep API keys out of workflow JSON, saved outputs, logs and source code.
- Validate the endpoint before retrieving credentials. Only the operator's
  `MINIMAX_H3_LMSTUDIO_BASE_URL` (default `http://127.0.0.1:1234/v1`) is trusted.
  Keep redirects and ambient proxies disabled for authenticated requests.
- Preserve local-route checks before body parsing or filesystem mutations.
- Pass dictionaries/lists between nodes. The current three stages also require
  `out_dir` and save results; loaders support resuming from those files.
- Registry briefs describe image/audio assets; do not fabricate media or
  introduce video references. Preserve scene-local H3 binding order.
- Preserve streaming cancellation checks and existing cache/schema behavior.

## Packaging and validation

- Keep `requirements.txt` and `pyproject.toml` runtime dependencies aligned:
  `pypdf`, `openai>=1.0,<3`, `httpx>=0.27,<1`. PDF support uses `pypdf`.
- Setuptools maps the root `web/` directory into the wheel's
  `minimax_h3_novel_pipeline.web` package. Keep both source-checkout and
  installed-package frontend paths working.
- `MANIFEST.in` must include the root entrypoint, runtime requirements,
  documentation, examples and browser JavaScript in the source distribution.
- Update root `README.md` and `examples/README.md` when setup or wiring changes.
- Install development tools with `python -m pip install -e ".[dev]"`.
- Run `python -m pytest`, `ruff check .`, and `python -m build` for layout or
  packaging changes. Pytest configuration lives only in `pyproject.toml`.
- Add focused regression coverage for affected behavior; mock network work.
  Tests must run without ComfyUI or a live LM Studio server. Check root
  registration, installed imports and bundled assets after file moves.
- Lint excludes historical `external source/` bundles. Do not hide failures
  in the active package. Report live runtime validation separately from
  offline checks.
