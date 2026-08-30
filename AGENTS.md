# AGENTS: ComfyUI plugin for MiniMax H3 novel pipeline

Purpose
- Provide concise instructions for AI coding agents to implement a ComfyUI plugin
  that reproduces the behavior of the three-step novel reference pipeline in this
  repository as ComfyUI nodes.

Token efficiency and instruction compliance
- Be concise: communicate and implement only what the task requires.
- Do not provide summaries unless the user explicitly requests one.
- Follow all user, repository, and system instructions strictly; do not weaken,
  reinterpret, or silently bypass them.
- Ask the user for clarification only when it is necessary to proceed safely or
  when a choice would materially affect the result.

Key repository entry points
- Step 1: [01_extract_chapter_references.py](01_extract_chapter_references.py) — extract
  per-chapter JSON reference catalogs using an LLM.
- Step 2: [02_consolidate_references.py](02_consolidate_references.py) — merge chapter
  catalogs to a consolidated registry and produce picture/audio asset briefs.
- Step 3: [03_generate_h3_prompts.py](03_generate_h3_prompts.py) — generate
  MiniMax H3 full-reference prompts per chapter/scene using the consolidated registry.

Why this file helps AI agents
- Links the canonical scripts (above) and summarizes responsibilities so an agent
  can implement ComfyUI node equivalents without re-reading the whole repo.
- Lists required runtime dependencies and key configuration knobs the plugin
  must expose to users.

Recommended plugin design (minimal, actionable)
- Plugin name: `minimax_h3_comfy` (folder: `comfyui_plugins/minimax_h3_comfy`).
- Primary node classes (one node per script):
  - `ExtractChapterReferencesNode` — inputs: generative `CLIP`, chapter files/folder;
    params: `chunk_chars`, `overlap_paragraphs`, `temperature`, `max_tokens`, `force`.
    outputs: per-chapter JSON payload(s) (Python dict objects) and optional saved files.
  - `ConsolidateReferencesNode` — inputs: list of chapter JSON payloads; params: `picture_threshold`,
    `audio_threshold`, `asset_batch_size`, `no_variants`, `audit_max_entities`, etc.; outputs: consolidated
    registry dict and picture/audio briefs list.
  - `GenerateH3PromptsNode` — inputs: consolidated registry and chapter text; params: `duration`,
    `scenes_per_chunk`, `scenes_per_chapter`, `max_tokens`, `temperature`; outputs: H3 prompt texts.
- Node behavior notes:
  - Keep the ComfyUI node outputs as Python objects (dicts / lists) so following nodes
    can bind, inspect, and optionally save to disk.
  - Preserve the scripts' schema checks and JSON parse logic; reuse functions where
    practical by importing from the original scripts or factoring shared utilities.
  - Expose `base_url` and `api_key` so users can point at LM Studio or other OpenAI-compatible endpoints.

Implementation details & constraints
- Dependencies: `pypdf` (optional for PDFs). Mirror the `requirements.txt` in
  the runtime environment used by ComfyUI.
- Streaming and Qwen3.5 special-cases: the scripts include robust streaming/ChatML
  helpers. A faithful node implementation should either reuse those helpers or
  implement equivalent streaming-aware calls (particularly for `qwen35-chatml`).
- Files vs in-memory: nodes should support both — return payloads in memory and
  optionally write the same JSON files the scripts produce.

Suggested plugin file layout
```
comfyui_plugins/minimax_h3_comfy/
  __init__.py            # node registration
  nodes.py               # ComfyUI node classes
  util.py                # shared helpers ported from scripts (parsers, schema, chat_json)
  requirements.txt       # subset of repo requirements for plugin runtime
  examples/              # small example flows demonstrating the 3-node chain
```

How an AI coding agent should proceed
1. Create `nodes.py` with three nodes matching the classes above. Keep nodes small —
   call out to `util.py` which ports parsing/LLM helpers from the scripts.
2. Add node registration in `__init__.py` per ComfyUI conventions (import and register nodes).
3. Provide example flows in `examples/` showing: (Extract → Consolidate → Generate).
4. Add minimal unit tests or an example notebook that runs the chain against a small
   sample chapter and a mocked LM client.

Security & safety
- Do not embed credentials in the plugin. The nodes run through ComfyUI's
  connected generative `CLIP` model.

Files added/modified by this change
| File | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Guidance for implementing a ComfyUI plugin mapping the three-step pipeline. |

Next suggestions
- I can scaffold the plugin (`comfyui_plugins/minimax_h3_comfy`) with `nodes.py` and
  `util.py` porting the core helpers from the scripts. Shall I scaffold those files now?
