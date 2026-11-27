# Example workflow

Node implementations live in `src/nodes/`; shared pipeline code and schemas live
in `src/core/`. This internal organization preserves node IDs and socket types,
so the example workflow uses the same wiring.

To try the standalone **Cinematic Chapter Adapter**, connect **Select Chapters**
to its `chapter_selection` input and **Provider Configuration** to its
`lmstudio_config` input, then queue. The adapter is an output node, so it needs
no downstream connection. It saves one `NNN_<chapter>.cinematic.json` object per
chapter under the run's `cinematic_chapters/` directory, with top-level
`chapter_name` (the source filename without its extension) and `sequences`.
Each sequence has `sequence`, `source`,
and an `adaptation` object containing `initialState`,
`event`, and `endingState` strings. `event` preserves all ordered events, numbered
and separated by newlines. Its `saved_files` output lists the files;
`cinematic_chapters` exposes the chapter records and sequence arrays in memory.
Execution diagnostics are saved separately in `NNN_<chapter>.cinematic.diagnostics.json`,
beside the chapter. This sidecar is updated atomically at each checkpoint, so failed
attempts remain inspectable after interruption. It is not included in `saved_files`
or passed to downstream nodes. The main chapter retains the source, adaptation,
chapter metadata and editable narrative proposals; review execution metadata lives
only in the diagnostics sidecar and internal resume cache.
Review the generated adaptations, then connect `cinematic_chapters` to
**Extract Chapter References** (replacing its former `chapter_selection` input).

Adapter defaults are `duration=8.0` (maximum seconds per scene), `chunk_chars=4000`,
`temperature=0.15`, `max_tokens=32768`, `correction_attempts=2` and `thinking=True`.
Each generation, repair or review request has its own token budget. Reasoning
and JSON share it, so even 32768 tokens can be exhausted. Consecutive gestures
may share a scene when they fit naturally. Existing workflows retain their saved
widget values; update `max_tokens` explicitly when using an older node instance.

Before reviewing each scene, the adapter estimates natural action/dialogue timing,
including sequential or simultaneous beats and pauses. A compound event can be
partitioned into verbatim action fragments, each referencing its original 1-based
`event_index`. Distinct successive operations within one event can use separate
beats with the same event index. Concatenated fragments must restore
every original event in order; only boundary whitespace is restored automatically.
Missing, duplicated, reordered or rewritten fragments are rejected. The planning
schema uses `{event_index, text}` references. For a whole event, the model sets
`text: null` and refers only to its identifier: it does not copy the event text.
Python resolves that identifier to the original adaptation text for semantic reviews,
repairs and readable diagnostics. Cached plans retain the identifier and null marker;
resolution does not rewrite the source or adaptation. A whole-event reference cannot
coexist with any other reference to the same event. Non-null text is reserved for
verbatim portions when a compound event needs separate action beats. Integer-only
references are not accepted.
Planning instructions are independent of the book and contain no fixed story excerpts.
They keep a simple physical action whole, including its tool, destination and spatial
qualifiers; these complements do not by themselves introduce another operation.
For a text mismatch or incomplete event,
the bounded correction request first offers a single whole-event identifier without
recopying text, or identifies the affected field and original event index for a split,
and supplies the original event, validated prefix and exact remaining text. This
correction context comes only from the supplied events; rejected model text is not
echoed into it. Citation retries identify the source and matching source anchor;
duplicate entity/aspect retries identify both conflicting update indices and require
one final source-supported fact for that pair. Earlier correction constraints remain
in subsequent retries. When the same defect persists, the next retry explicitly
rebuilds the affected section from original data. Retry requests therefore change
instead of repeating an identical deterministic request. The retry budget is unchanged;
exhaustion retains valid scene data without approval.
This validates textual coverage, not the model's semantic choice of action boundaries
or simultaneity, which independent review still checks. Oversized scenes trigger
bounded repairs that can split a single source sentence into multiple scenes.
For two or more successive beats, a 25% uncertainty allowance is applied to the
total estimate: at an 8-second maximum, such a scene must be estimated at 6.4 seconds
or less. This editorial allowance does not apply to a single ongoing/simultaneous
beat. Timing repairs must produce at least two replacement scenes, each planned and
reviewed separately; shortening the wording alone is rejected.
These are estimates, not measured video durations. Persistent overruns or unavailable
planning/reviews retain valid scene data with explicit incomplete-review diagnostics.
An unavailable plan does not skip semantic checks: source fidelity, pacing and local
continuity are still reviewed against the source. Partial reviews retain findings,
never enter the approval cache, and do not prevent independent successor scenes.
Known timing overruns can still trigger splitting when semantic review is unavailable;
each replacement must undergo its own planning and review.
Planning validation diagnostics include the scene, stage, check, a bounded reason code
(for example `event_coverage`, `event_reference`, `event_order`, `event_text_mismatch`,
`event_text_incomplete`, `timing_total_mismatch`, `source_quote_mismatch` or
`schema_type`) and a safe JSON field path. Arbitrary exception messages and rejected
backend values are not saved. `output_limit` is reserved for an explicit output-limit
failure; `request_failed` alone does not identify a token limit or a validation cause.
Each logical planning attempt is checkpointed at `planning_attempt` and retained in
`narrative_review.planning_diagnostics` in internal `progress.json` and as
`planning_attempts` in the diagnostics sidecar, never in the completed chapter.
Records include the scene (`sequence`), stage, `assessment_id`, 1-based
`attempt`, and `status` (`failed` or `completed`). A later completed attempt in the
same assessment resolves the request failure; a failed attempt remains in the history
even when the final scene is reviewed successfully. These statuses describe planning
requests, not narrative review approval. Failed requests also carry a safe reason;
validation failures include a field path. Sidecar records include English explanations
and suggested actions. They identify repeated failures and identical rejected responses
using fingerprints, record the retry strategy, and link recovered failures to their
successful attempt with `resolved_by_attempt`. A changed first validation error alone
does not prove that all earlier defects were fixed.
For text mismatches, `event_index` is 1-based; `beat_index` and `reference_index` are
0-based. `validated_prefix_length` counts original-event characters before the failing
fragment (including restored boundary whitespace). `matched_chars` counts matching
characters within that fragment; `mismatch_offset` is the 0-based position of the first
difference in the original event. `expected_remaining_length` and `received_length`
describe the remaining original text and stripped received fragment. Incomplete tails
have a zero received length. Counts use Unicode characters, not bytes. The first
mismatching expected/received characters are recorded as Unicode code points (null
means end of text), with a readable difference in the sidecar. Raw rejected model
responses and arbitrary exception messages are never saved. The sidecar also contains
the source and events of committed scenes, available timing beats/pauses, the estimate
with any sequential allowance, and incomplete checks. Generation completion and
review completion are reported separately; unavailable timing is null, never zero.
Where an event reference cannot be validated, an original event index is not guessed.
Each record includes SHA-256 fingerprints of the effective system prompt (including
the configured duration), strict schema and complete logical request prompt. These
identify planning inputs before model-specific shared-transport transformations;
internal transport retries are not counted as separate planning attempts. Resume
preserves existing records and assigns a new assessment ID to new requests. Cache hits
do not fabricate attempt history or fingerprints for requests from earlier runs.
Source-cited physical facts help track earlier objects and support changes even when
recent state descriptions omit them; generated paraphrases never outrank the book.
Identity matching ignores case, whitespace and one definite article only; qualifiers,
numbers and distinct prop names remain distinct. A new source-cited observation
conservatively replaces that entity's whole earlier context snapshot, across aspects,
to prevent stale contradictions when generated categories are wrong. Other entities
persist, within the 24-fact bound. This can forget an unchanged older facet; it does
not assert that the facet ceased to exist. Full source quotations and historical plans
remain in checkpoints. Category instructions guide the model; structural validation
cannot certify the meaning of a physical claim or its category.
Internal timing plans and the bounded physical context are saved in
`cinematic_chapters/.cache/<chapter_id>/progress.json`, alongside resumable drafts.
Only v2 checkpoints with review policy `timed-source-context-v10` can be resumed; start a fresh adaptation for older
checkpoints. Completed `.cinematic.json` chapters remain loadable and are not rewritten.

To resume from those saved adaptations, replace the adapter with **Load Cinematic
Chapters**. Set `cinematic_path=20260911153042/cinematic_chapters` (using your
previous run's timestamp), or select a single `*.cinematic.json` file in that
folder. Connect the loader's `cinematic_chapters` output to Extract's
`cinematic_chapters` input and keep Provider Configuration connected to Extract.
The loader validates saved chapters and loads folders in natural filename order.
Its paths start at `output/minimax_h3_novel` and must stay within that root.

1. In ComfyUI Settings → MiniMax H3 Novel → Providers, enable a provider, fetch and select a model, then save its URL, key and model. Add **Provider Configuration** and choose that provider. The saved model appears automatically; Qwen controls appear only for a Qwen model in LM Studio.
2. Add **Select Chapters**, **Extract Chapter References**, **Consolidate
   References** and **Generate H3 Prompts**. Connect configuration to all three.
   Connect **Spatial Continuity** or **Load Spatial Continuity** to the optional
   `spatial_continuity` input to use its staging plan and review final prompts.
   Without a connection, Generate does not infer spatial staging internally.
   Reference bindings use the chapter catalogs with or without a spatial plan.
   Generate checks final prompts after camera refinement, corrects contradictions
   within `repair_attempts`, and records unresolved issues as validation warnings.
   Existing visual designs are carried by `consolidated_references`.
   Review the chapter manifest for spatial plans and prompt checks.
   The example disables generation thinking and camera refinement. Independent
   `camera_thinking` and `review_thinking` controls also default to false.
   Completed structured requests are checkpointed under the node's chapter cache,
   including individual continuity-review batches. Use `resume_run` and the same
   `out_dir` to recover matching work after a failure, with `force=false`.
   Request counts, cache hits and elapsed times are recorded in the chapter cache's
   `request_metrics.json`. Existing runs cannot supply checkpoints they never saved.
   For a reusable chapter-wide plan, connect **Consolidate References** and
   **Provider Configuration** to **Spatial Continuity**, then its
   `spatial_continuity` output to **Generate H3 Prompts**. This exports
   `spatial_continuity.json` and provides a `spatial_json` text output.
   The node also copies `spatial_continuity.schema.json` beside it and adds a
   relative `$schema` link for editor validation. Keep both files together when
   editing a copy. Load Spatial Continuity validates against the bundled schema.
   Each sequence is planned with the entire chapter visible, including distant
   future constraints, initial/final entity positions, camera, evidence and warnings.
   Generate uses this plan during scene planning and prompt generation, and reviews
   final prompts even with the toggle off. Rerun Spatial Continuity after consolidation changes.
   To edit the plan manually, save a corrected JSON file inside
   `output/minimax_h3_novel` (renamed copies are supported). Add **Load Spatial
   Continuity**, set `spatial_path` to that file or its folder, and connect the
   same `consolidated_references` used by Generate. Connect the loader's
   `spatial_continuity` output to Generate instead of the planning node.
   The loader preserves edits, validates the v2 structure and source digest,
   and reloads changed file contents on the next execution without LM Studio.
   Keep chapter/sequence IDs and `source_digest` intact; edit positions, camera,
   states and evidence. A folder path looks for `spatial_continuity.json`.
   Set `duration` in Cinematic Chapter Adapter to the length of each scene. Set
   `max_shots` to your desired limit (1 by default; a shot needs at least 2.5s).
   Optionally describe the camera axis and movement in `camera_direction`.
   Enable `refine_camera` for a final LM Studio camera-only pass after prompt
   validation. Check `camera_warnings` if the extra description is rejected.
   `repair_attempts` also limits camera-only correction retries after a rejection.
3. Choose the chapters once in Select Chapters, then connect its
   `chapter_selection` output to Cinematic Chapter Adapter. Connect the adapter's
   `cinematic_chapters` output to Generate's `cinematic_chapter` input. Connect the
   adapter’s `cinematic_chapters` output to Extract. Share configuration with the adapter. Connect Extract's
   `chapter_catalogs` to Consolidate, then
   Consolidate's `consolidated_references` to Generate. Each of these four processing
   nodes can terminate the workflow and displays its own text preview after execution;
   no separate Preview Text node is required.
4. In Consolidate, choose `image_style` (default **cinematic photographic**).
   Image prompts cover visually present entities with usable appearance details;
   characters can be enriched from their independent visual profiles.
   Use `asset_batch_size=4` as the initial setting for the Qwen3.5 9B model.
5. Queue the workflow. Open the new timestamp folder shown in configuration status.
   Copy each view's prompt from `references/image_prompts/` into your Qwen-Image-2512
   workflow, or use Generate's `image_prompt_text` output. The same export is saved
   under `h3_prompts/image_prompts/`. Generate and review one image per view.
   Consolidate merges repeated facts into shared appearance prose in the source
   language, with separate variants for actual visual changes. For current-format
   catalogs, use Load Chapter Catalogs to rerun Consolidate and Generate.
   Older reference formats require extraction again.
6. Use the desired chapter and scene entry from Generate's `prompts` payload
   with your MiniMax H3 Reference to Video node. Generate/load the media from
   the registry's briefs and attach it in the entry's image/audio asset-ID order.
   Alternatively, open the scene's `*_prompt.txt` (or chapter `all_prompts.txt`),
   attach the named references in the listed order, and copy the text below
   `COPY-PASTE PROMPT:` into H3. Check any validation warnings before use.

To resume, replace Extract with **Load Chapter Catalogs**, or replace
Extract and Consolidate with **Load Consolidated References**. Generation
still needs the original chapter text and LM Studio configuration.

All three stages save to their `out_dir`. The pipeline produces no video
references and does not generate images or audio itself.

Set each stage's output budget with its own `max_tokens`; configuration no
longer applies a second cap. Old configuration widgets migrate on workflow
load. Set extraction passage size with Extract's `chunk_chars`; the shared
`qwen35_safe_chunk_chars` field is removed. Paragraph order is preserved within each phase; phases never overlap.

Optional `default_temporal_context` on **Extract Chapter References** accepts a date
or era, such as `1925` or `France, late 19th century`. Leave it empty when unknown.
Explicit narrative timing and flashbacks take precedence. This is user context,
not source evidence. It is saved in chapter catalogs and consolidated timelines,
preserved by the loaders, and used for missing visual design details and H3 prompts.
After changing it, rerun extraction, consolidation and generation; extraction caches
are invalidated automatically.
Restart ComfyUI and refresh the browser after updating the node package.
For JSON failures, check the ComfyUI console's `LLM stream` lines for character
counts, `finish_reason` and `local_stop`. These diagnostics omit generated text.

Set `thinking` on each LM Studio processing node: it defaults to `true` for
Cinematic Chapter Adapter, Extract Chapter References, Consolidate References
and Spatial Continuity, and `false` for Generate H3 Prompts. Mistral and Gemma
ignore this Qwen control; configure GPT-OSS reasoning in LM Studio.
Provider Configuration shares connection and sampling settings only.
Recreate existing nodes to refresh their widget layouts.

Chapter paths are relative to ComfyUI's input directory, for example
`minimax_h3_novel/chapter_01.txt`. Upload or copy external chapters there.
Every queue execution creates a shared `yyyyMMddHHMMSS` folder, using local time,
under `output/minimax_h3_novel`. Output subfolders are relative to that run: use
`chapter_catalogs` for extraction, `references` for consolidation and
`h3_prompts` for generation. Loader paths include the previous timestamp: for example,
`catalog_path=20260911153042/chapter_catalogs` or
`consolidated_path=20260911153042/references/consolidated_references.json`. Absolute paths
must stay within the corresponding root; `..` and links escaping it are rejected.

To compare settings with results, open `extract_configuration.json` in
`chapter_catalogs`, `consolidate_configuration.json` in `references`, or
`generate_configuration.json` in `h3_prompts`. Each snapshot includes the shared
LM Studio controls, resolved model, node settings, inputs and run timestamps.
When `status` is `completed`, `outputs` lists result paths relative to the snapshot
and SHA-256 hashes. `started` indicates an incomplete execution. API keys are omitted.
Change settings in the nodes; these JSON files are records, not editable presets.


Compact retries preserve entity capacities and visual features, allowing 500 characters for stable descriptions and 350 for chapter appearance/state. Cached results from the older retry policy are regenerated automatically.

To edit invented appearance details, open the previous run's
`references/visual_designs.json`. Each entity has its ID, name, type, a source-facts
snapshot and an editable `added_details` object:

```json
"added_details": {
  "hair": "Short copper hair.",
  "default_outfit": "A plain charcoal linen tunic."
}
```

Change or remove traits in that object; leave IDs, names and types intact. Use
Load Chapter Catalogs for the same novel, set Consolidate's `visual_designs_path`
to that edited file, and queue again. Outputs go into a fresh run. Imported designs
are checked against the current source facts; contradictions must be corrected.
An empty object removes all additions for an entity. Omitted entities are designed
anew. The selected style affects all new reference prompts. Loading a consolidated
registry preserves its already generated style and prompts.

Edit names, aliases and source-supported descriptions in the chapter catalog using
`chapter_references.schema.json`, then reload it with Load Chapter Catalogs.
Consolidation reconciles identities and aliases; keep its duplicate audit enabled
for variants found within one chapter. Source observations retain their sequence
addresses. Neutral image prompts combine shared intrinsic traits with the applicable
scene state. Equal resolved appearances share assets with explicit `scene_bindings`;
a new scene or action alone does not require another reference. Regenerate visual
designs when identity edits change global IDs.

For each entity, the text export labels source description and added design details
separately, followed by individual copy-paste prompts for each neutral reference view and applicable appearance variant.
`image_prompts.json` contains asset IDs, descriptions and prompts for automation.
Style/design choices do not modify extracted novel facts. Review generated images
for consistency before using them as H3 references.

Consolidation audits registries above `audit_max_entities` using likely-duplicate clusters instead of skipping the audit. `audit_similarity` (0.68) and `audit_cluster_size` (24) control matching and batch size; `no_audit` still disables auditing. Clustering is heuristic and may miss duplicates across groups.

Only current v5 chapter catalogs and registries are accepted; consolidation requires neutral visual observations and scene-addressed image applicability. Regenerate older outputs and recreate configuration nodes: the legacy backend selector was removed. The package contains only the ComfyUI pipeline; standalone CLI and fallback implementations are removed.

The adapter uses `duration` to split the chapter into naturally paced scenes.
Generate preserves exactly one scene per supplied sequence. Generate reads duration
from each chapter; `max_scenes=0` generates the complete chapter.

## Cinematic preprocessing

Use **Cinematic Chapter Adapter** or **Load Cinematic Chapters** for the
cinematic extraction workflow. Generate uses those cinematic chapters and the
matching consolidated registry, with optional spatial continuity.
Recreate older configuration nodes and use relative `out_dir` subfolders;
old widget layouts, narrative bundles and absolute output paths are unsupported.

### Extraction timeline contract

Each chapter is independent. Sequence IDs must be positive and strictly increasing.
Extract Chapter References reads resolved `adaptation.initialState` followed by
`adaptation.event`, splitting long passages by `chunk_chars`. Original `source` and
`endingState` are not extraction evidence. Full cinematic
sequences and editable narrative choices remain available to downstream nodes.

Catalog references contain `first_sequence` and
`state_by_sequence[sequence].source`, a list of source observations. Extraction
then consolidates identities in ordered source windows (up to eight chunks,
12,000 source characters and 32 incoming occurrences per batch; a single larger
chunk remains indivisible). The model receives these adaptation passages, the preceding
passage and the local identity registry. It proposes source-quoted merge groups;
code validates IDs, entity types, names and verbatim quotations before merging.
Identical generic names alone never force a merge. Distinct physical instances
stay separate; uncertain or failed decisions retain unmerged occurrences and
incomplete diagnostics. Requests exceeding the safe context budget are not sent.
Identity decisions have their own cache under `.cache/<chapter_id>/identity_*.json`;
source extraction caches remain reusable. All original observations retain their
sequence addresses. Cross-chapter consolidation and its duplicate audit still run.
Locations are places/spaces. Narrative positions, poses, handling and orientations
remain in adaptation, not neutral appearance. `visual_traits` carry source-quoted
intrinsic or scene-local facts; `presence` distinguishes visible entities, offscreen
voices and mentions. Summaries and descriptions use the source narration's language.

Consolidation retains `timeline[chapter_id][sequence].source` observations losslessly,
including conflicts and repeated evidence. Identity reconciliation and duplicate
audits use stable facts; image briefs never promote temporary state to identity.
`first_occurrence_by_chapter` contains each chapter's `first_sequence`; global
`first_chapter_id` and `first_sequence` follow supplied chapter order and numeric
sequence order. `reference_timeline.source_observations_at(entity, chapter_id,
sequence)` returns only observations at that sequence; missing means unknown.

Catalogs reject obsolete local identity links, phase-timing fields and phase
observations. Re-extract and reconsolidate older outputs; no conversion or fallback
reader is provided. H3 receives source observations separately from cinematic
adaptation; it never invents phase timing for a source observation.

Each uncached source chunk gets extraction and semantic review, with at most two
corrections. Persistent defects and incomplete reviews retain valid data and are
recorded in `<chapter_id>_references.diagnostics.json`, outside the catalog.
Technical diagnostics are English. `force` bypasses caches. Changes to the resolved
adaptation invalidate dependent extraction results and identity decisions.

Adapter reviews also check source detail
coverage and transitions; `correction_attempts` bounds targeted sequence repairs.
After bounded repairs/reconstruction, persistent doubts produce two or three full
scene alternatives in `narrative_review.issues[].proposals`, with the first
`selected: true`. No narrative disagreement blocks execution and there is no
`accepted` flag. Each proposal includes a reason and complete `adaptation` with
`initialState`, `event`, and `endingState`; the selected one governs the scene.
Set another proposal's `selected` to true and the others to false in the original
`.cinematic.json`, then reload with **Load Cinematic Chapters** and rerun extraction,
consolidation, spatial continuity (if used), and generation. This refreshes reference
observations, assets and prompts together. Missing selections default to the first;
multiple selections use the first selected proposal, with a console notice. File
changes are detected on the next run. Alternatives remain in saved review metadata
but only the resolved adaptation is sent to downstream narrative model calls.
Malformed data or technical failures still stop the workflow.
Reviews use additional LM Studio requests and remain model judgments, not a proof
of narrative fidelity. Regenerate adaptations and catalogs from previous runs.
Generate still receives the matching cinematic adaptation. Neutral reference
appearance does not replace its narrative actions or continuity input.
See [the neutral reference contract](../docs/neutral_visual_references.md) for
shared traits, conflict diagnostics and exact scene/image applicability.

### Generation from cinematic chapters

Connect Cinematic Chapter Adapter or Load Cinematic Chapters to Generate's
`cinematic_chapter` input. The connected sequences must match the consolidated
registry's source provenance, IDs and adaptation phases. Generate preserves one
scene per sequence, without replanning or deduplication. Generate reads the adapter's
`duration` (4–15 seconds) from each chapter. Missing duration requires rerunning
the adapter; changed scenes require regenerating downstream references and plans.
`max_scenes` limits outputs while retaining all later scenes as continuity evidence.

Each saved scene records its sequence ID. Reference state is scoped to that exact
sequence and phase, with no automatic carry-forward. Stable visual designs and
image styles remain part of the consolidated assets and image-prompt export.


H3 reference generation allows at most 9 images and 3 audio references per scene.
The defaults are 8 images and 3 audio references; the per-subject image budget
cannot exceed 9 and remains subject to the total image budget. Audio references
are selected only when an image reference is available.
Scene duration is restricted to 4–15 seconds. The pipeline's 2.5-second minimum
shot length permits at most 6 shots at 15 seconds; shorter scenes automatically
use a lower shot cap. This pacing rule is internal, not an H3 shot-count limit.
max_scenes defaults to 0 (all scenes per chapter).
Temperature, token budget and repair attempts control LM Studio, not H3.
Actual reference audio files must each be 2–15 seconds and total no more than
15 seconds. This prompt-only pipeline cannot validate media duration or file size.
Limits: https://www.minimax.io/news/minimax-h3-open-source

### Extract original character evidence only

Connect **Select Chapters** and **Provider Configuration** (Qwen) to
**Character Evidence Extractor**. Select all novel chapter files (one chapter per
file) or their containing folder. The node returns the complete evidence dictionary
and the path to `character_evidence/character_evidence.json` in the current run.
`character_scope=all_characters` (default) reads the full text, including unnamed
people and descriptions that precede identification.
`character_scope=proper_names` indexes named-character candidates across
all chapters and batches each occurrence with one preceding/following paragraph,
bounded by `chunk_chars`, without extending through chains of pronouns. Single
newlines delimit paragraphs; oversized lines use sentence boundaries.
Extraction keeps enduring physical appearance, including age and established
scars, and explicit identity statements. It retains naming evidence for alias
resolution and reviews descriptions to remove temporary states and staging.
The default batch size is 12,000 characters. Both reasoning settings default to
false. They can be enabled independently for extraction and attribution audit.
Diagnostics are saved after each batch and on interruption, with selected text
fraction, extraction/audit call counts and per-batch durations.
Proper-name selection uses capitalization heuristics. It can miss distant
pronoun context and names without capitalization.
Completed batch caches are reused within the current run in
`output/minimax_h3_novel/<timestamp>/character_evidence/.cache/<chapter_id>/`. Diagnostics alongside the evidence
report timing, passage selection, cache hits, requests/retries and estimated tokens.
To continue a failed extraction from a previous run, set `resume_run` to just its
folder name, for example `20261025154413`, and keep `force=False` and the original
source paths/model/settings. Matching batches from that run's `character_evidence/.cache`
are copied into the current run's cache; only missing or unreadable batches require
extraction. New outputs belong to the current run; the previous run is unchanged.
Leave `resume_run` empty to disable reuse from a previous run.
Edit that file manually. Use `force` to regenerate compatible cached batches;
adjust `chunk_chars`, `max_tokens` or `correction_attempts` if validation fails.
`correction_attempts` bounds extraction, semantic audit and neutral appearance
review retries. Passage IDs
resolve into exact quotations and positions in the normalized chapter. The audit
returns a verdict and issue code per observation. An unavailable or unusable
audit keeps observations that pass deterministic checks and marks the review
incomplete in `character_evidence_diagnostics.json`. Definite rejections are recorded
there. Consolidation records its own exclusions in `character_profiles_validation.json`.

Keep the original chapters under ComfyUI's input directory. The evidence file
contains their references and normalized fingerprints. Consolidation reopens the
files to verify their fingerprints, exact quotations and positions. PDF positions
refer to normalized extracted text. Regenerate earlier character formats in node order.
To resume after editing, use **Load Character Evidence** with `evidence_path` set
to the evidence output folder, such as `20261024163701/character_evidence`
(relative to `output/minimax_h3_novel`, or an absolute path inside it).
The node automatically finds `character_evidence.json` in that folder. A saved
JSON file path is also accepted, including renamed edited copies.
Its `character_evidence` and `saved_file` outputs replace the extractor
outputs directly. No LM Studio configuration or source chapters are needed.
Saved edits reload on the next workflow execution without rewriting the file.


To consolidate characters, connect either evidence node's `character_evidence`
output to **Consolidate Character Profiles**, together with **LM Studio
Configuration** using Qwen. Include all novel chapters. `persistence_filter` defaults
to `persistent`; select `everything` to include temporary and unknown observations.
The filter applies to appearance evidence. The saved source document retains
selected appearance observations and explicit identity evidence. `thinking` defaults to `false` for this node.
Set it manually on existing nodes that retain `true`. Keep `out_dir` at
`character_profiles` to save `character_profiles/character_profiles.json` and its
JSON Schema in the current run. Set `context_tokens` at or below LM Studio's
loaded context (default 65,536); it includes the response and correction reserves.
Identity requests include verbatim context around each source location to resolve
short quotations and aliases. Physical classification uses a fixed trait vocabulary
and shared batches across characters, without synonym-comparison requests.
Value comparisons attach both source IDs from their inputs. Independent valid
decisions survive a malformed neighbor; unresolved decisions remain flagged.
Requests preserve complete evidence coverage and cache completed validated batches. Lower the control for smaller
requests. `character_profiles_requests.json` records sizes and request counts.
Set `correction_attempts=0` (the new default) to skip invalid decisions without
semantic retries or recursive batch subdivision. Existing nodes may retain `2`.
Sources are preserved, but rejected decisions remain unverified. Shared JSON
transport recovery remains active for malformed responses.

Edit a copy of `character_profiles.json`, then connect **Load Character Profiles**
to it through `profiles_path` (file, renamed copy or containing directory under
`output/minimax_h3_novel`). It returns `character_profiles` and `saved_file` without
network calls. Change conflict `choices[].selected` and direct `link_to_local_id`
links as needed. No selection and multiple selections are both accepted; links
must target a principal entry, without chains or cycles. Saved edits reload on
execution and are never replaced by an old model response. Correct invalid
source evidence through **Load Character Evidence** and reconsolidate instead.

The generator projects validated permanent physical traits from the entire profile
document, including other chapters. It makes no model calls, invents no missing
features and creates no cache. The first selected conflict choice applies; an
unselected conflict is ignored. Explicit identity links combine evidence, while
incompatible linked traits remain unresolved. Chronological variants, temporary
clothing, accessories, injuries, expressions, poses and actions are excluded.

The output folder contains:

- `character_visual_profiles.json` contains selected profiles and embedded
  visual designs. The `saved_file` socket points to this file.
- `visual_designs.json` and schemas: stable `source_facts`, separate editable physical
  `added_details`, and validated `character_profiles` with identity and provenance.

Connect `character_visual_profiles` to **Consolidate References**, or set its
`visual_designs_path` to the exported file. Both paths carry the same permanent facts
and identity evidence. Profile IDs and registry IDs are independent: consolidation
checks contextual identity evidence and leaves ambiguous mappings unassigned with a
warning. Matching names alone never establish a profile-backed identity.

Consolidation joins permanent physical traits by a unique canonical name or alias.
Presence and reference-relevant appearance come from the chapter catalogs. H3
uses the resolved cinematic adaptation to stage actions and momentary visual
details at their authored point in the sequence.

Ambiguous profile mappings remain unresolved and are reported in
`consolidated_references.diagnostics.json`.
Missing, changed or ambiguous source files stop source verification with an
actionable error.

For another chapter, pass edited `visual_designs.json` to the generator's optional
input to reuse compatible physical defaults. Only selected identities are exported.
Defaults in ignored, chronologically changing or source-established physical fields are omitted; clothing and
accessory defaults are excluded from this physical projection. To correct facts,
identity links or conflict choices, edit the embedded `character_profiles` document
(or the upstream profile file), reload it and rerun the generator. `source_facts`
is a deterministic projection of these decisions. Files remain inside
`output/minimax_h3_novel`.

**Load Character Visual Profiles** reloads saved current generator output without
model calls or writes. Consolidation recomputes local states when cinematic sequences
change; the physical generator does not depend on their revision.


All node caches live directly under their configured output directory:
`<timestamp>/<out_dir>/.cache/`. Chapter-specific caches use `.cache/<chapter_id>/`.
Nodes without caching do not create this directory. Character evidence `resume_run`
uses the same `out_dir` in the selected previous run, including custom output paths.
Only this cache layout is read; existing caches at older locations are not moved
or reused automatically.

## H3 cinematic input (2.6.0)

Connect the same Cinematic Chapter Adapter or Load Cinematic Chapters output to Extract Chapter References and Generate H3 Prompts.cinematic_chapter. Set duration in Adapter before adaptation; Generate reads it automatically. One cinematic sequence becomes one H3 scene; Generate does not replan it. Recreate the two nodes when updating existing workflows. Regenerate saved chapters without duration and rebuild references if the adaptation changes.
