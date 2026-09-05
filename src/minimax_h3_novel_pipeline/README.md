# MiniMax H3 Novel ComfyUI nodes

These nodes turn local novel files into validated MiniMax H3 prompts while
using an LM Studio OpenAI-compatible server for all planning and JSON generation.

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

The chapter picker and API-key settings routes require direct local access at
`http://localhost:8188` or a loopback IP address (use your configured port).
Remote clients, forwarded proxy requests, and cross-origin browser requests
receive HTTP 403. This policy covers only this plugin's routes. A proxy that
strips forwarding headers and rewrites Host/Origin is indistinguishable from a
local client; do not expose these routes through such a proxy.

### LM Studio configuration

All non-secret connection settings are entered in the `LM Studio Configuration`
node: `api_url`, `model`, `chat_backend`, `thinking`,
`qwen35_max_output_tokens`, `qwen35_length_retries`, and the Qwen sampler /
safe-chunk controls. They are shared by all three LLM nodes through the
`lmstudio_config` output.

The workflow URL must exactly match the trusted server endpoint (an optional
trailing slash is accepted). The default is `http://127.0.0.1:1234/v1`. To use
another host, port, or base path, set this in the environment that starts
ComfyUI and use the same URL in the node:

```powershell
$env:MINIMAX_H3_LMSTUDIO_BASE_URL = "https://your-lmstudio-server.example/v1"
```

This trust setting cannot be changed by a workflow. All three LLM nodes validate
the endpoint before retrieving the API key; the configuration output contains
no key. Authenticated requests do not follow redirects or use environment proxy
settings, so configure the final, directly reachable LM Studio endpoint.

### Qwen3.5 extraction profile

Leave `chat_backend` on `auto` and `thinking` off.  For Qwen3.5, the Extract
node now asks LM Studio's schema-constrained chat endpoint first, with
`reasoning: off`, `top_k`, `min_p`, and `repeat_penalty` controls.  It falls
back to the manual ChatML stream only when that endpoint is unavailable.

`qwen35_safe_chunk_chars` defaults to 3600, so a larger Extract-node
`chunk_chars` value is safely split before an individual JSON catalog becomes
too large.  The retry schema is also intentionally smaller.  Use 2048 only for
short passages; 3072–4096 is the practical output budget for passages with
several named characters, locations, and props.  Reducing `max_tokens` cannot
repair a response that needs more tokens to close its JSON object.

Enter the API key in **ComfyUI Settings → MiniMax H3 Novel → LM Studio → LM
Studio API Key**. The browser sends it to the ComfyUI backend before a workflow
is queued; it is not embedded in the workflow JSON or written by this plugin to
a file.

Like ComfyUI's other global settings, this value is stored locally in its
browser profile in plain text. For a shared computer or a stronger security
boundary, select `Environment variable` in the configuration node instead and
define:

  ```powershell
  $env:MINIMAX_H3_LMSTUDIO_API_KEY = "lm-studio"
  ```

Use `lm-studio` when LM Studio authentication is disabled; otherwise use the
key configured in LM Studio. Start ComfyUI from that same PowerShell window.
The node lets advanced users change the environment-variable name, but it
defaults to `MINIMAX_H3_LMSTUDIO_API_KEY`.

To keep these settings for future Windows sessions, run the following once,
then completely close and reopen ComfyUI (and the terminal or launcher that
starts it):

  ```powershell
  setx MINIMAX_H3_LMSTUDIO_API_KEY "lm-studio"
  ```

- Connect `LM Studio Configuration` to each LLM node, then use
  `Extract → Consolidate → Generate H3 Prompts → Select H3 Scene`. All LM
  Studio requests use streaming: each response arrives as small token chunks
  rather than one final response. Pressing ComfyUI's **Stop** checks between
  chunks, closes the active LM Studio stream (which aborts its generation), and
  then lets ComfyUI cancel the workflow. Qwen3.5 in `auto` mode additionally
  stops once complete JSON is received.
- Generate or load media elsewhere in ComfyUI from the Step-2 briefs.
- In **MiniMax H3 Reference to Video**, attach the selected prompt and then
  attach images in `image_asset_ids_in_h3_order`; attach audio in
  `audio_asset_ids_in_h3_order`.

Installation note

The node package uses its bundled `pipeline_step1_extract.py`,
`pipeline_step2_consolidate.py`, and `pipeline_step3_generate.py` files. No
external `origin` folder is required.

`Select H3 Scene` does not fabricate media from text briefs. It makes the
required asset order explicit so the actual ComfyUI image/audio values can be
connected to the H3 node safely.
