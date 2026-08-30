# MiniMax H3 ComfyUI plugin (scaffold)

This folder contains a minimal scaffold for a ComfyUI plugin that reproduces
the three-step novel reference pipeline in this repository as nodes.

Files
- `__init__.py` — exposes `NODES` for registration.
- `nodes.py` — node exports and ComfyUI mappings.
- `load_chapter_catalogs.py` — reloads saved Step-1 `*_references.json` files
  without rerunning extraction.
- `util.py` — small helpers reused by nodes (file discovery, chapter reading).
- `requirements.txt` — plugin runtime dependencies.

Usage
- To reuse Step-1 results, add **Load Chapter Catalogs**, set `catalog_path` to
  the folder holding `*_references.json` (or to one such file), and connect its
  `chapter_catalogs` output to **Consolidate References**.

Next steps
- Implement full LLM calls by porting the `chat_json` and streaming helpers from
  the repository scripts into `util.py`.
- Add proper ComfyUI node metadata, UI parameters, and unit tests/examples.
