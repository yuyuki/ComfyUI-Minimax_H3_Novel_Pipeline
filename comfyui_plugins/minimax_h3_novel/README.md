# MiniMax H3 Novel ComfyUI nodes

These nodes turn local novel files into validated MiniMax H3 prompts while
using a generative ComfyUI `CLIP` model for all planning and JSON generation.

Files
- `Extract Chapter References` — builds chapter-local character, location, and
  object catalogs.
- `Consolidate References` — preserves continuity across chapters and creates
  image/audio reference briefs. The registry deliberately contains no video
  assets because the source is a novel.
- `Generate H3 Prompts` — plans scene prompts. `every_beat` is the default and
  asks for every distinct visualizable narrative beat; `key_scenes` is the
  faster selective mode.
- `Select H3 Scene` — outputs one scene prompt plus the exact image/audio asset
  ID order expected by `<Picture N>` and `<Audio N>`.
- `load_chapter_catalogs.py` — reloads saved Step-1 `*_references.json` files
  without rerunning extraction.
- `load_consolidated_references.py` — reloads a saved Step-2 registry.
- `requirements.txt` — plugin runtime dependencies.

Usage
- `Extract → Consolidate → Generate H3 Prompts → Select H3 Scene`.
- Generate or load media elsewhere in ComfyUI from the Step-2 briefs.
- In **MiniMax H3 Reference to Video**, attach the selected prompt and then
  attach images in `image_asset_ids_in_h3_order`; attach audio in
  `audio_asset_ids_in_h3_order`.

Installation note

The node package is standalone. It does not load or require the numbered CLI
pipeline scripts; copy only this `minimax_h3_novel` folder into ComfyUI's
`custom_nodes` directory, then restart ComfyUI.

`Select H3 Scene` does not fabricate media from text briefs. It makes the
required asset order explicit so the actual ComfyUI image/audio values can be
connected to the H3 node safely.
