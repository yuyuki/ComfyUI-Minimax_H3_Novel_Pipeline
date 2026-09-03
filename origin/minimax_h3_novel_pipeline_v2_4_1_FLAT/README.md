
## Version check (v2.3.1)

Step 1 should expose:

```bash
python 01_extract_chapter_references.py --version
# 01_extract_chapter_references.py 2.3.1

python 01_extract_chapter_references.py --help
# includes --qwen35-max-output-tokens
```

The release ZIP is provided in a **flat layout** so extracting it directly into an existing project directory can overwrite the previous scripts.

# MiniMax H3 Novel Pipeline v2 — Multi-view References

Three standalone scripts, designed for a local LM Studio OpenAI-compatible server.
Default LM Studio base URL: `http://127.0.0.1:1234/v1`.

## What changed in v2

A single important entity can have several reusable image references.

Example character set:

- `PIC_CHAR_001_FACE_FRONT`
- `PIC_CHAR_001_FULL_BODY_FRONT`
- `PIC_CHAR_001_THREE_QUARTER`
- `PIC_CHAR_001_BACK_VIEW`

Example location set:

- `PIC_LOC_001_WIDE_ESTABLISHING`
- `PIC_LOC_001_SECONDARY_ANGLE`
- `PIC_LOC_001_KEY_DETAIL`

Step 3 selects only the useful views for a scene and maps them to request-local
MiniMax H3 labels. Several `<Picture N>` labels can define the same `<Subject N>`.

Example scene binding:

```text
<Picture 1> -> Hotel wide view -> <Subject 1>
<Picture 2> -> Elena face      -> <Subject 2>
<Picture 3> -> Elena full body -> <Subject 2>
<Picture 4> -> Elena back view -> <Subject 2>
```

The generated H3 `subject_definitions` should therefore define Elena once and cite
all three pictures inside that one subject definition.

## Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Start LM Studio's local server and load your Qwen model.

---

## Qwen3.5 / LM Studio thinking workaround (v2.2)

Qwen3.5 GGUF models can expose an `enable_thinking` Jinja variable, but some LM Studio versions have ignored `enable_thinking=false` / `chat_template_kwargs` on `/v1/chat/completions`. Structured `json_schema` output has also interacted poorly with Qwen3.5 reasoning streams.

For that reason the scripts default to:

```text
--thinking = off
--chat-backend = auto
```

When the selected model ID looks like Qwen3.5 (`qwen3.5` or `qwen35`), `auto` uses `/v1/completions` and constructs the model's ChatML generation prefix explicitly. In non-thinking mode it ends the prompt with an empty thinking block:

```text
<|im_start|>assistant
<think>

</think>

```

This bypasses the LM Studio chat-template toggle and makes the requested non-thinking state deterministic for Qwen3.5-style ChatML models. JSON is then validated in Python instead of using constrained `json_schema` decoding.

You can override the transport:

```bash
--chat-backend auto
--chat-backend qwen35-chatml
--chat-backend openai-chat
```

Examples:

```bash
python 01_extract_chapter_references.py output_0.txt --no-thinking --chat-backend auto --force
```

```bash
python 01_extract_chapter_references.py output_0.txt --thinking --chat-backend qwen35-chatml --force
```

Each LLM call prints its selected backend, thinking state, elapsed time and, when available, generated-token count.


## Step 1 — per-chapter extraction

```bash
python 01_extract_chapter_references.py chapters \
    --out-dir chapter_references \
    --model "YOUR-LM-STUDIO-MODEL-ID"
```

Outputs one JSON per chapter:

```text
chapter_references/
  chapter_01_references.json
  chapter_02_references.json
  ...
```

The v2 chapter schema adds `reference_view_hints`, e.g.:

```json
{
  "local_id": "CHAR_001",
  "canonical_name": "Elena",
  "reference_view_hints": [
    "face_front",
    "full_body_front",
    "three_quarter",
    "back_view"
  ]
}
```

These are hints, not final MiniMax labels.

If you already generated Step-1 JSONs with the previous v1 pipeline, Step 2 can read
them directly. Missing `reference_view_hints` are treated as empty and the deterministic
multi-view policy supplies the normal character/location views. Rerunning Step 1 with
v2 is recommended when you want the LLM to contribute chapter-specific view hints.

---

## Step 2 — consolidate + build multi-view assets

```bash
python 02_consolidate_references.py chapter_references \
    --out consolidated_references.json \
    --asset-prompts-out reference_asset_prompts.txt \
    --model "YOUR-LM-STUDIO-MODEL-ID"
```

Default multi-view policy:

### Character

- major / required: face front, full body front, 3/4, back
- recurring / recommended: face front, full body front, 3/4
- minor / optional: face front

### Location

- major / required: wide establishing, secondary angle, key detail
- recurring / recommended: wide establishing, secondary angle
- minor / optional: wide establishing

### Object

- major / required: hero 3/4 + detail close-up
- otherwise: hero 3/4

The chapter extraction hints are merged into this policy when they fit the configured
maximum number of views.

Useful controls:

```bash
--max-character-base-views 4
--max-location-base-views 3
--max-object-base-views 2
--picture-threshold recommended
--audio-threshold recommended
--asset-batch-size 16
```

To generate references even for optional entities:

```bash
--picture-threshold optional --audio-threshold optional
```

Step 2 writes:

- `consolidated_references.json`
- `reference_asset_prompts.txt`

The JSON contains stable book-level entity IDs and stable asset IDs. It also includes
`entity_asset_index` for fast lookup.

Chapter-specific material variants can be created for major visual changes such as a
disguise, substantial injury, transformation, time jump, or heavily altered location.
Disable them with `--no-variants`.

---

## Step 3 — scene-specific H3 prompts

```bash
python 03_generate_h3_prompts.py chapters \
    --references consolidated_references.json \
    --out-dir h3_prompts \
    --duration 8 \
    --model "YOUR-LM-STUDIO-MODEL-ID"
```

Step 3 asks the LLM which available views are useful for each scene. Examples:

- close dialogue -> face + 3/4
- full-body movement -> full body + 3/4
- walking away -> back + an identity view
- establishing shot -> location wide
- alternate angle -> location secondary/reverse

Then a deterministic allocator applies the reference budget.

Useful controls:

```bash
--max-pictures 8
--max-pictures-per-subject 4
--max-audio 4
--repair-attempts 2
```

These are workflow budgets, not claims about MiniMax's service-side limits.

Each scene produces:

```text
scene_001_..._prompt.txt
scene_001_..._assets.json
scene_001_..._source.txt
```

The `*_assets.json` file contains:

- `subjects`: one H3 Subject per referenced entity
- each subject's list of one or more H3 Pictures
- `picture_input_order`: exact image attachment order
- `audio_input_order`: exact audio attachment order

Example:

```json
{
  "subjects": [
    {
      "h3_subject_label": "<Subject 2>",
      "global_id": "CHAR_001",
      "canonical_name": "Elena",
      "pictures": [
        {
          "h3_picture_label": "<Picture 2>",
          "asset_id": "PIC_CHAR_001_FACE_FRONT",
          "view_type": "face_front"
        },
        {
          "h3_picture_label": "<Picture 3>",
          "asset_id": "PIC_CHAR_001_FULL_BODY_FRONT",
          "view_type": "full_body_front"
        },
        {
          "h3_picture_label": "<Picture 4>",
          "asset_id": "PIC_CHAR_001_BACK_VIEW",
          "view_type": "back_view"
        }
      ]
    }
  ]
}
```

Step 3 validates and, if necessary, asks the LLM to repair:

- all six H3 full-reference sections and their order
- request-local label consistency
- multiple Pictures correctly remaining under one Subject
- no unwanted standalone Picture retention lines for identity references
- shot numbering and timestamps
- target duration
- dialogue `<d>[Language] ...</d>` syntax
- retention markers
- normal 350–500 word `detailed_description` target

## Recommended workflow

```text
chapters/
   ↓
01_extract_chapter_references.py
   ↓
chapter_references/*.json
   ↓
02_consolidate_references.py
   ↓
consolidated_references.json
reference_asset_prompts.txt
   ↓
create the actual PNG/WAV references
   ↓
03_generate_h3_prompts.py
   ↓
h3_prompts/<chapter>/scene_*_prompt.txt
h3_prompts/<chapter>/scene_*_assets.json
```

## Important continuity principle

Book IDs are permanent:

```text
CHAR_001
LOC_003
PIC_CHAR_001_FACE_FRONT
```

MiniMax H3 labels are per-generation-request:

```text
<Subject 1>
<Picture 1>
<Audio 1>
```

That separation is intentional. A scene may use only 5 of the hundreds of assets in
the global reference library, so Step 3 remaps the selected subset to compact local
H3 numbering each time.


## Qwen thinking / reasoning mode

The three scripts run Qwen in **non-thinking mode by default**. Each LM Studio request prepends:

```text
/no_think
```

This is recommended for the structured extraction/consolidation/prompt-generation workflow because it normally reduces latency and avoids spending a large token budget on hidden reasoning.

Explicitly keep the default with:

```bash
python 01_extract_chapter_references.py chapters --no-thinking
```

Enable reasoning for an experiment with:

```bash
python 01_extract_chapter_references.py chapters --thinking
```

The same `--thinking` / `--no-thinking` options are available in **all three scripts**. The selected mode is also included in cache keys, so switching modes does not silently reuse results generated under the other setting.
\n\n## v2.3 — Qwen3.5 long-generation protection\n\nThe manual Qwen3.5 ChatML backend now uses **streaming JSON completion detection**.\nIt no longer waits indefinitely for `<|im_end|>` after the model has already finished\na JSON object. As soon as the root JSON object closes syntactically, the HTTP stream\nis closed and the JSON is parsed.\n\nStep 1 also defaults to smaller `8000`-character source chunks. This is intentional:\nfor a local 9B Q8 model, several moderate extraction calls are normally faster and\nmore recoverable than one very long request.\n\nA safety cap can be changed with:\n\n```bash\n--qwen35-max-output-tokens 3500\n```\n\nFor Step 1, a useful conservative command is:\n\n```bash\npython 01_extract_chapter_references.py chapters \\\n  --no-thinking \\\n  --chat-backend qwen35-chatml \\\n  --chunk-chars 8000 \\\n  --max-tokens 3000 \\\n  --qwen35-max-output-tokens 3000\n```\n\nLM Studio `Max Concurrent Predictions = 1` is recommended while using the current\nsequential pipeline.\n
---

## v2.4 — Compact JSON / Qwen3.5 reliability update

This release addresses local Qwen3.5 models that can generate very long extraction JSON and reach the output-token limit before closing the root object.

### Step 1 changes

- Default chapter chunk size reduced to `5500` characters.
- Default `--max-tokens` reduced to `2200` for chapter-reference extraction.
- Strict compactness rules are embedded directly in the extraction schema/prompt:
  - max 3 evidence anchors per entity;
  - max 120 characters per evidence anchor;
  - max 6 aliases;
  - max 6 distinguishing features;
  - concise persistent/temporary descriptions;
  - long dialogue quotations are explicitly forbidden in evidence.
- `stable_visual_description` is explicitly restricted to persistent identity traits; temporary clothing, wounds, wetness, dirt, restraint state and carried equipment belong in the chapter-specific state fields.
- When a Qwen3.5 JSON generation ends because of the output-token limit or parses as incomplete, the script automatically retries with a much more aggressive compact-output instruction.

### New / changed CLI options

Both names below are accepted and mean the same thing:

```bash
--qwen35-max-output-tokens 2200
--max-output-tokens 2200
```

Configure compact retries with:

```bash
--qwen35-length-retries 2
```

Check that you are running the expected script with:

```bash
python 01_extract_chapter_references.py --version
```

Expected output for this release:

```text
01_extract_chapter_references.py 2.4.1
```

### Recommended command for Qwen3.5 9B via LM Studio

```bash
python 01_extract_chapter_references.py output_1.txt \
  --no-thinking \
  --chat-backend qwen35-chatml \
  --chunk-chars 5500 \
  --max-tokens 2200 \
  --max-output-tokens 2200 \
  --qwen35-length-retries 2 \
  --force
```

On Windows `cmd.exe`, either put the command on one line or use `^` for line continuation.

The same Qwen3.5 truncated-JSON retry mechanism is also present in Steps 2 and 3.


## Automatic next-step command

Starting with **v2.4.1**, every successful stage prints a copy/paste-ready recommended command for the next stage. Step 1 proposes the Step 2 consolidation command using the actual output directory and active LM Studio/model settings. Step 2 reconstructs the original chapter source location from the chapter JSON metadata and proposes the Step 3 prompt-generation command. Step 3 reports that the pipeline is complete and prints a platform-appropriate command to open the generated output directory.

If a non-default LM Studio API key is in use, the command prints `YOUR_LM_STUDIO_API_KEY` rather than echoing the secret value.
