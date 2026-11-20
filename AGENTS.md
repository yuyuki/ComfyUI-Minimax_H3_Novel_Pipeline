# AGENTS: MiniMax H3 Novel Pipeline

## Working rules

## Operational rules
- Focus on working environments and setup, not source-code development. Prefer small, validated setup changes over broad rewrites.
- Do not modify user data, model files, or outputs without explicit permission. Preserve existing edits and the scaffold’s src/ layout.
- If a workflow requires custom nodes, document the missing dependency and provide explicit installation steps.
- Follow user, repository, and system instructions. Do not weaken or bypass requirements. Ask for clarification when safety or a material choice requires it.
- Do not add or retain legacy code, obsolete formats, or compatibility paths for removed features. Remove related unused tests and documentation; support only current pipeline contracts.
- Keep responses concise. Do not provide summaries unless requested.
- Prefer warnings and explicit JSON diagnostics over aborting the pipeline for
  recoverable review, correction or advisory LLM failures. Record the affected
  scene, stage/check and failure reason, retain valid generated data, and continue
  independent work. Mark incomplete reviews as incomplete, never approved. Preserve
  cancellation, security checks, source/schema integrity and filesystem errors;
  stop when no valid result can be retained safely.
- For every node, save diagnostics in a separate, clearly associated JSON file
  beside its main output. Keep only data required by downstream nodes in the
  main output file and in-memory values passed between nodes. Do not include
  diagnostic reports, warnings, retry histories, validation details or execution
  metadata in those payloads. Preserve any functional data required downstream,
  including editable narrative choices. Record incomplete checks explicitly in
  the diagnostic file; never present them as approved.
  Write diagnostic explanations, suggested actions and technical messages in
  English; preserve quoted source text in its original language.
- Before researching or changing code in a repository with .codegraph/, use CodeGraph to locate relevant symbols and trace dependencies and callers. Check its source, freshness, and coverage before relying on it. Use text search for exact strings, logs, configuration keys, and prose. Skip CodeGraph if .codegraph/ is absent.
- Follow [README.md](.github/README.md) when updating the root README.
- running Python tests and validation commands outside the sandbox.

## Changelog maintenance

- Before every push to `master` (or `main`, the current default branch), update
  root `CHANGELOG.md` in the same push with concise summaries of all newly
  included commit messages. Read commit bodies as well as subjects.
- Keep dated entries newest first, group related changes and retain their
  short commit hashes for traceability. Include fixes, features, refactors,
  tests, documentation, workflow changes and version bumps; account for merge
  commits without duplicating the description of their changes.
- Record changes not yet committed under `Unreleased`. On the next update,
  move committed entries to their commit-date sections and add their hashes.
  Check Git history against the existing entries so no commits are omitted
  or summarized twice. A changelog-maintenance commit can be covered on the
  next update; do not amend repeatedly just to record its own hash.
- Do not infer registry publication dates or versions from commit dates.
  This is a required pre-push maintenance step, not an automatic Git hook.
- Preserve the publishing workflow's explicit `--changelog-file CHANGELOG.md`
  argument: a local changelog file alone does not populate registry metadata.
  Each new published version receives the complete changelog snapshot. Bump
  `project.version` in `pyproject.toml` for a new release; publishing does not
  backfill changelogs on existing registry versions.

## Current architecture

- Root `__init__.py` is the ComfyUI checkout entrypoint. It re-exports mappings
  from `src/` and serves `./web/js`.
- `src/__init__.py` registers seventeen nodes and four
  local HTTP routes. `src/nodes/__init__.py` exports the node classes.
- Nodes live in `src/nodes/`, each in its own module: `lmstudio_config.py`,
  `extract_chapter_references.py`, `load_chapter_catalogs.py`,
  `consolidate_references.py`, `load_consolidated_references.py`,
  and `generate_h3_prompts.py`.
- Shared logic and JSON schemas live in `src/core/`.
- `src/core/lmstudio_pipeline.py` dynamically loads bundled scripts relative to its
  own directory, configures Qwen behavior and wraps streaming cancellation.
- `util.py` contains file discovery, JSON persistence and shared helpers.
- `lmstudio_settings.py` stores the API key in memory and validates the
  trusted endpoint. `route_access.py` enforces direct local browser access.
- `web/js/minimax_h3_novel.js` implements chapter picking and settings.

## Pipeline entry points

All paths below are relative to `src/`:

| Stage | Node wrapper | Bundled implementation |
|---|---|---|
| Extract | `nodes/extract_chapter_references.py` | `core/cinematic_references.py` (shared helpers in `core/pipeline_step1_extract.py`) |
| Consolidate | `nodes/consolidate_references.py` | `core/pipeline_step2_consolidate.py` |
| Generate | `nodes/generate_h3_prompts.py` | `core/pipeline_step3_generate.py` |

Keep node wrappers small. Reuse bundled schema checks, JSON parsing,
streaming, compact retries and Qwen3.5 structured-JSON/ChatML fallbacks.
Preserve existing node IDs and socket types for workflow compatibility.

## Behavior and security constraints

### Cinematic continuity

- Treat the novel and the resolved cinematic adaptation as authoritative for
  narrative continuity. Follow the current sequence's opening state, event
  and ending state in order. An authored change of location, time or action
  determines how the next clip progresses.
- By default, Generate H3 Prompts prepares later sequences in each chapter
  for an opening image supplied directly in MiniMax. Use the previous clip's
  last frame when the action continues. For an explicit cut at the opening,
  use an image matching the new opening state. The first sequence of each
  chapter starts independently. Keep the opening image outside the registry
  and the numbered reference bindings. The operator can disable continuation
  for independent clips.
- Generate a prompt even when a sequence has no registry image or audio
  binding. Reuse explicitly linked references when useful. Keep Spatial
  Continuity optional for staging and camera guidance.
- Keep extraction and consolidation focused on source-grounded observations
  and reusable references. Avoid a second inferred timeline or automatic
  carry-forward of locations, characters, objects and local visual states.

### Neutral visual reference requirements

The reference contract and validation cases are documented in
[docs/neutral_visual_references.md](docs/neutral_visual_references.md).

- Keep reference prompts generic across books and genres. Never embed names,
  scene numbers, sample passages or prop examples from a particular book in
  system instructions. Story-specific details belong only in the input data.

- Reference images describe reusable visual appearance, not an opening frame or
  a staged action. Leave gestures, poses, expressions, object manipulation and
  narrative orientation to H3 animation. Choose views that expose visual details
  clearly, such as a frontal view of an inscribed tablet.
- Keep one identity per established physical entity and at most one visual
  association per entity per scene. Only visually present entities belong in
  that scene's visual references; offscreen speech remains a separate audio need.
- Use character visual profiles for timeless appearance. Keep scene-specific
  wardrobe and visible condition separate from global identity, and never import
  future actions, accessories or states into an earlier scene.
- Combine intrinsic object and location features into shared descriptions. Reuse
  unchanged material, construction, markings and proportions wherever the same
  entity is visible. A later intrinsic clarification may enrich earlier references
  only when the same entity and unchanged trait are supported; never backdate a
  transformation or infer presence from description reuse. Flag conflicting or
  uncertain traits in diagnostics instead of silently combining them.
  Address damage and other appearance changes chronologically; a later
  transformation must not contaminate an earlier reference. Distinct physical instances must
  remain distinct even when they share a generic name.
- Object and location references require useful visual detail, not just a name.
  Never invent unsupported traits to fill an empty description. A visible
  character may be identified without local traits when its visual profile can
  supply the appearance. Reuse unchanged references; create variants only for
  actual appearance changes, never merely for actions or new scene numbers.
- Chapter and consolidated reference contracts are v5. Require regeneration of
  incompatible saved data; do not convert it or retain compatibility readers.
  Keep the project version unchanged for this work.

### Current pipeline constraints

- Reference extraction reads `adaptation.initialState` followed by `adaptation.event`
  for each sequence, using their resolved names; exclude original `source` and
  `endingState` from extraction and chapter-local identity evidence. Preserve these
  neutral observations and quoted evidence under `state_by_sequence[sequence].source`.
  Validate scene-local trait quotations before accepting extraction. Keep actions
  in cinematic adaptation, not appearance. Keep adaptation and editable choices
  available to downstream nodes; move technical reviews to the diagnostic sidecar.
- Consolidate chapter-local identities after extraction using ordered initialState/event
  passages and validated, passage-quoted LLM merge groups. Keep occurrences separate
  until confirmed; identical generic names are not identity evidence. Preserve all
  original observations losslessly at their sequence addresses. Cache identity
  decisions independently from extraction and record incomplete decisions in diagnostics.
  Do not emit or accept obsolete `link_to_local_id`, `first_phase` or extracted
  adaptation-phase observations. Reject old catalogs and registries with a clear
  re-extraction/reconsolidation message; do not convert them or add fallback readers.
  Cross-chapter identity and alias reconciliation belong to consolidation; its audit
  handles remaining duplicates and must stay enabled when that is required.
- Compile `visual_reference.common_traits` and chapter/sequence-addressed presence
  and local traits from confirmed identities. Unchanged intrinsic traits can be
  shared; conflicting traits stay local and trigger incomplete diagnostics.
  Image assets use explicit `scene_bindings`; neither chapter membership nor a
  base variant alone establishes applicability to a scene. Reuse equal resolved
  appearances across scenes. Never carry local states through unobserved scenes.
- Locations are actual places/spaces, never positions or orientations of objects
  or people. Spatial properties belong to the relevant entity's observations.
  Keep distinct physical instances separate using source-supported qualified names.
- Generate `chunk_summary`, names and descriptive prose in the source narration's
  language. Review language fidelity as well as category correctness; preserve
  proper names and verbatim foreign-language quotations. Keep JSON keys/enums fixed
  and technical diagnostics in English.
- All language-model work uses LM Studio's OpenAI-compatible API. Never add
  CLIP inputs, model loading or CLIP-dependent execution paths.
- For every new LLM node, follow the existing stages: create the client through
  `lmstudio_pipeline.make_client_and_model` and request structured output through
  the shared `lmstudio_json.chat_json` transport with an explicit JSON schema
  (`name`, `strict`, `schema`). Use `response_format.type = "json_schema"`;
  never introduce `json_object` mode or pass `None` as the schema. Keep Qwen
  settings, streaming, cancellation and ChatML fallbacks in the shared transport.
  Add mocked request coverage confirming the schema is sent on initial requests
  and retries, and that compact retries preserve the intended entity/evidence
  coverage rather than truncating arrays of independent observations.
- Share non-secret URL, model and backend controls through
  `LMStudioConfigurationNode` and `MINIMAX_LMSTUDIO_CONFIG`.
- Credentials may come only from ComfyUI settings or the runtime environment,
  never workflow inputs. The current node implementation uses ComfyUI settings;
  do not document an environment-key selector unless it is implemented.
- Keep API keys out of workflow JSON, saved outputs, logs and source code.
- Validate the endpoint before retrieving credentials. Only the operator's
  Trusted API URL in ComfyUI settings (default `http://127.0.0.1:1234/v1`) is trusted.
  Receive it together with the API key through the protected local settings route.
  Keep redirects and ambient proxies disabled for authenticated requests.
- Preserve local-route checks before body parsing or filesystem mutations.
- Pass dictionaries/lists between nodes. The current three stages also require
  `out_dir` and save results; loaders support resuming from those files.
- Registry briefs describe image/audio assets; do not fabricate media or
  introduce video references. Preserve scene-local H3 binding order.
- Preserve streaming cancellation checks and existing cache/schema behavior.
- Resolve each node's output directory through `run_output.stage_output` inside
  the configured timestamped run folder. All nodes that use a cache must keep it
  directly under their own output directory: `<timestamp>/<out_dir>/.cache/`.
  Chapter-specific cache entries belong in `.cache/<chapter_id>/`, not beside
  chapter outputs. Do not repeat the node name inside `.cache`, create a cache
  at the run/shared output root, or add empty caches to nodes that do not use one.
  Character evidence resume reads the same `out_dir/.cache` in the selected
  previous run. Support only this layout; do not add legacy-path fallbacks.

## Packaging and validation

- Keep `requirements.txt` and `pyproject.toml` runtime dependencies aligned:
  `pypdf`, `openai>=1.0,<3`, `httpx>=0.27,<1`. PDF support uses `pypdf`.
- Setuptools maps the root `web/` directory into the wheel's
  `minimax_h3_novel_pipeline.web` package. Keep both source-checkout and
  installed-package frontend paths working.
- `MANIFEST.in` must include the root entrypoint, runtime requirements,
  documentation, examples and browser JavaScript in the source distribution.
- Install development tools with `python -m pip install -e ".[dev]"`.
- Run `python tools/generate_architecture.py` when incrementing `project.version`
  in `pyproject.toml`. Also update architecture descriptions and regenerate the map
  when removing an obsolete contract, without requiring a project version bump;
  obsolete documentation must not preserve removed behavior.
- Run `python -m pytest`, `ruff check .`, and `python -m build` for layout or
  packaging changes. Pytest configuration lives only in `pyproject.toml`.
- Add focused regression coverage for affected behavior; mock network work.
  Tests must run without ComfyUI or a live LM Studio server. Check root
  registration, installed imports and bundled assets after file moves.
- Do not hide lint failures in the active package. Report live runtime
  validation separately from offline checks.
