# Generate H3 Prompts

## Input contract

Connect `Cinematic Chapter Adapter.cinematic_chapters` or `Load Cinematic Chapters.cinematic_chapters` to `Generate H3 Prompts.cinematic_chapter` (`MINIMAX_CINEMATIC_CHAPTERS`). The input is an ordered list of cinematic chapters, not source-file paths.

Each chapter carries `chapter_name`, `duration` and ordered `sequences`. Each sequence contains its ID, exact source excerpt and `initialState`, `event`, `endingState` adaptation phases. Selected narrative proposals are resolved by the shared cinematic parser.

One supplied sequence produces one H3 scene. Generation never splits, merges, deduplicates or replans sequences. Repeated events remain distinct. The former chapter-selection input, scene-count-per-chunk control and scene-plan cache are removed.

## Duration and provenance

Set `duration` (4–15 seconds) in Cinematic Chapter Adapter before adaptation. The model receives this target during initial adaptation, source recovery, semantic review and correction. It must split long passages while preserving action, dialogue, pacing and exact source coverage. This is a planning constraint, not a measured guarantee of rendered video length; inspect dense dialogue and action scenes.

The adapter reads source windows of at most `chunk_chars`. For a window before the chapter's end, the model returns complete scenes and an exact `unprocessed_source` suffix for the last incomplete scene. Only validated, reviewed scenes advance the source cursor; the next window starts at that suffix and adds new text. Python verifies verbatim coverage, without omissions or duplicated committed text. If no complete scene fits, increase `chunk_chars`; the adapter stops rather than inventing a boundary or looping.

Instructions, generated descriptions and review context add to the source window's size. Repairs send the target source once with its immediate neighbors. Reviews are batched by input size and split further after an output-budget failure; an exhausted single-scene review still fails explicitly. Output-limited initial adaptations retry a smaller window without advancing the cursor. `Cinematic reading` logs show committed source characters and deferred text; `Cinematic request` logs show prompt and schema characters. `reasoning_chars` remains a count of model-generated reasoning for one transport attempt, separate from input size. Thinking stays under the node's control.

Generate H3 Prompts reads duration directly from each connected chapter; it has no duration widget. Multiple chapters may carry different durations. Chapters without a duration are rejected. The explicit cinematic sequences must match those used to extract and consolidate references, including IDs, source and selected adaptation phases. Changed adaptation requires updated references and spatial continuity.

## Execution

1. Validate cinematic input, matching registry provenance and duration before opening the model client.
2. Compile exact sequence, reference, character and optional spatial evidence.
3. Create one immutable scene from each supplied sequence.
4. Request only scene reference metadata: visible/speaking entities, location, dialogue presence and picture views. Strict schemas restrict IDs to the catalog. Local validation rejects speakers with absent dialogue, unavailable views and contradictions with approved character presence. Failed metadata receives bounded correction feedback.
5. Build scene-local bindings, generate the six H3 sections and validate the prompt.
6. Repair affected sections without changing the supplied source or adaptation. Optionally refine camera wording.
7. Review continuity against earlier valid prompts and all future scenes. Recheck every correction before approval.
8. Save scene files, bindings, validation status and the chapter manifest.

`max_scenes` limits generated outputs, not future evidence: reviewers still receive all later cinematic scenes. Metadata is prepared for the whole chapter.

## Continuity citations

Each review request receives a strict JSON schema whose `evidence_paths` enum contains only addresses present in that exact packed batch. Examples:

- `/current_scene/adaptation_notes`: cinematic phases and adaptation context.
- `/current_scene/dialogue_present`: scene metadata.
- `/current_prompt_sections/detailed_description`: generated prompt wording.
- `/future_requirements/0`: the complete first future-scene record in this batch.

Container citations preserve all nested evidence. Paths are JSON-pointer escaped and resolved locally before a review is accepted. No alias or legacy path correction is performed.

On an unresolved citation, one bounded retry receives the rejected review, invalid paths, allowed paths and explicit field-location guidance. It retains the original evidence and strict schema. Continued failure stops the review; it never approves a scene with missing evidence.

Review requests contain prompt sections once. Raw prompt text is retained only when parsing would lose content. Evidence is packed losslessly under a 48,000-byte budget, including the dynamic schema. Repairs and later reviews may require several requests.

## Caches and migration

Restart ComfyUI and recreate Generate H3 Prompts in existing workflows, then reconnect the cinematic chapter, consolidated references, configuration and optional spatial plan. Recreate Cinematic Chapter Adapter to expose its duration widget. No automatic migration of positional widgets or removed sockets is provided.

Use a new run with `force=false`. Matching exact requests are reusable through `resume_run`; changed inputs, schemas or model settings are recomputed. The prompt checkpoint contract is version 3: previous scene-plan caches and prompt contracts are not read. Preserve previous run files; cache cleanup is unnecessary.

Caches remain under `<run>/<out_dir>/.cache/<chapter_id>/`. The chapter manifest records the cinematic chapter digest. A result with `valid=false` is not an approved prompt and is not included in continuity history.

## Controls

- `thinking`: reference metadata, generation and format repairs.
- `review_thinking`: continuity reviews and semantic repairs.
- `camera_thinking`: optional camera refinement.
- `repair_attempts`: bounded metadata/format/semantic corrections; distinct from transport retries and the evidence-path retry.
- `max_tokens`: output allowance including reasoning.

No configuration guarantees artistic quality or successful validation. Review saved validation results before using prompts.
