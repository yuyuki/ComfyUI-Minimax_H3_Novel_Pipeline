# Example workflow

1. Add **LM Studio Configuration**, enter the server URL/model, and set the
   API key in ComfyUI Settings → MiniMax H3 Novel → LM Studio.
2. Add **Extract Chapter References**, **Consolidate References** and
   **Generate H3 Prompts**. Connect configuration to all three.
3. Choose a chapter in Extract and Generate, or enter the same chapter paths
   in both. Connect Extract's `chapter_catalogs` to Consolidate, then
   Consolidate's `consolidated_references` to Generate.
4. Connect Generate's `prompts` to **Select H3 Scene**. Start with
   `chapter_index=1` and `scene_index=1`.
5. Connect the selected prompt to your MiniMax H3 Reference to Video node.
   Generate/load the media from the registry's briefs and attach it in the
   emitted image/audio asset-ID order.

To resume, replace Extract with **Load Chapter Catalogs**, or replace
Extract and Consolidate with **Load Consolidated References**. Generation
still needs the original chapter text and LM Studio configuration.

All three stages save to their `out_dir`. The pipeline produces no video
references and does not generate images or audio itself.
