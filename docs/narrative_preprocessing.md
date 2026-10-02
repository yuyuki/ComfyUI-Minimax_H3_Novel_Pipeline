# Cinematic normalization and temporal continuity

The optional path uses the existing trusted LM Studio client, structured JSON/ChatML
fallbacks, streaming cancellation and scene-local image/audio bindings. It does not
add fields or sections to the MiniMax H3 prompt format.

## Wiring

Connect **Select Chapters** and **LM Studio Configuration** to **Novel Cinematic
Simplifier**. Connect its `cinematic_narrative` output to the optional input of the
same name on **Generate H3 Prompts**. Continue to feed the original chapter selection
and consolidated registry to Generate. Reference extraction still reads original prose.
Spatial Continuity and camera refinement remain independently optional.

The simplifier runs separate normalization, state extraction and semantic review
requests per non-overlapping passage. It preserves actions, dialogue, visual atmosphere,
props, identity and causality; it removes nonvisual metaphors and makes physical
relationships explicit. Normalization is not summarization. The state pass sees original
prose, normalized text and the preceding passage's ending state.

Generate plans scenes using those contracts, reviews event coverage, then projects
them into per-scene contracts. It propagates ending states and verifies passage-ending
states. The existing H3 generator and repairer receive these contracts. Final wording
is reviewed after camera processing and on cache hits. Conflicts in opening descriptions,
subject definitions or summary are errors, even when the action timeline is correct.

`chunk_chars` on the simplifier controls passage size (default 6000, paragraph boundaries
preserved; a single long paragraph can exceed this). Generation's overlap/chunk-size
settings do not split these passages again. `scenes_per_chunk` still limits planning:
increase it or reduce preprocessing passage size if coverage validation reports omitted
events. `max_scenes` deliberately limits generation; it does not require reaching the
chapter's final state when truncating a passage.

## Nodes and debugging

| Node | Inputs | Outputs |
|---|---|---|
| Novel Cinematic Simplifier | chapter_selection, lmstudio_config, out_dir, chunk_chars, correction_attempts | cinematic_narrative, cinematic_text, validation_report, state_before_json, events_json, state_after_json |
| Narrative State Tracker | original_scene, cinematic_text, lmstudio_config, correction_attempts; optional current_state_json | narrative_state, state_before_json, events_json, state_after_json, continuity_report |

All outputs except the first on each node are strings suitable for Preview Text.
The standalone tracker lets you inspect a passage or connect its `state_after_json`
to the next tracker's `current_state_json`. Empty input means infer the first scene's
opening state. Explicit `{"entities": []}` means an established empty state: new
entities must be introduced by events. Null attributes mean unknown.

The simplifier saves `cinematic_narrative.json` beneath its timestamped run `out_dir`.
It contains original and normalized passages, contracts and review history. Generate
saves per-scene `narrative_state` and `continuity_review` in the existing chapter
`manifest.json`. Original chapter digests reject stale preprocessing. Prompt cache keys
include the contract and previous final prompts. Existing result sockets are unchanged.

## Internal schema

`minimax-cinematic-narrative.v1` contains chapters keyed by resolved source path;
each chapter has a source digest, cinematic text, reports and ordered segments.
`minimax-narrative-state.v1` is the standalone tracker socket wrapper.
The strict LM Studio schemas and their runtime validators live in `src/narrative_state.py`.
They are internal pipeline contracts, not H3 API schemas.

Each contract has `state_before`, `initial_frame`, `events`, `state_after`.
States contain an `entities` array with unique stable IDs (including distinct prop
instances such as `torch_1` and `torch_2`). Every entity has:

```json
{
  "id": "torch_2", "kind": "object",
  "location": "indy", "position": null, "posture": null,
  "owner": "indy", "relationship": "held", "status": "lit",
  "visible": true, "environment": null
}
```

Locations and characters use the same fields with kind `location` or `character`.
Owned objects have exactly one owner and location equals that owner's ID; an object
stored in a holder uses the holder ID. Unowned props use their physical location.
Events carry a unique ID, description, verbatim source evidence, kind (`action`,
`introduction`, `repair`) and atomic changes:

```json
{
  "id": "mouth_1", "description": "Indy places the torch between his teeth.",
  "source_evidence": "Tenant la torche entre les dents, il tendit la main...",
  "kind": "action",
  "changes": [{"entity": "torch_2", "field": "relationship",
               "before": "held", "after": "in_mouth"}]
}
```

The illustrative evidence must be replaced with an exact excerpt from the actual
scene. Evidence alone cannot prove the interpretation; the separate semantic review
must verify it supports the change.

Deterministic checks enforce schema/types, unique identities, chronological evidence,
event preconditions, unchanged carried state, introduction before visibility/ownership,
one location/owner, opening-frame consistency, explicit movement/posture changes and
event-replayed ending state. `intact → damaged → broken` is allowed. Restoring damaged
or broken objects requires an explicit repair event supported by the source.

Errors identify entity, expected state, conflicting state, introducing event and
suggested correction. Corrections receive original text, cinematic text, current state,
the previous candidate and errors. Every candidate is revalidated. `correction_attempts`
(0–10, default 2) bounds preprocessing/scene-contract corrections; existing Generate
`repair_attempts` bounds final H3 corrections. Invalid contracts stop before H3
generation. Unresolved final-prompt errors retain the existing `valid=false` and
validation-warning behavior; inspect warnings before submitting prompts to H3.

## Limits and validation

Deterministic replay validates the declared facts, not arbitrary natural language.
Literary interpretation, completeness and final-prompt leakage detection still rely
on the configured model's semantic review. Multiple structured calls cost additional
time and context; reduce passage size for small models. State grows with the chapter.
State resets for each chapter; the standalone tracker can carry state manually across
chapter boundaries. There is no dedicated loader for saved preprocessing bundles yet.

Offline tests mock LM Studio. `tests/fixtures/00_PROLOGUE.md` contains the supplied
French source, copied from the operator's ComfyUI input directory. Source-backed tests
check exact evidence and chronology: the holder is revealed from the backpack;
intending to store the torch precedes actual storage; rope damage precedes retrieval;
the torch moves between the teeth only afterward; and the rope breaks before Indy
hangs by one hand, loses his grip and finally falls. These tests complement the
smaller synthetic fixtures; mocked responses do not establish model interpretation quality.
Live ComfyUI/LM Studio model quality and generated videos require separate validation.
