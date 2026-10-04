# Legacy cinematic narrative contracts

Novel Cinematic Simplifier and Narrative Continuity have been removed.
This document describes the retained internal helpers and compatible bundles
accepted by Generate's optional `cinematic_narrative` input. For current node
wiring, see [the workflow guide](../examples/README.md).

The optional path uses the existing trusted LM Studio client, structured JSON/ChatML
fallbacks, streaming cancellation and scene-local image/audio bindings. It does not
add fields or sections to the MiniMax H3 prompt format.

## Internal processing

Semantic complaints about both prose and state contracts are verified before they
trigger corrections. Reviewers select indexed source IDs; Python supplies their exact
source text, avoiding failures caused by the model retyping quotations. Confirmed
complaints require existing source IDs and exact candidate evidence.
Contract evidence can quote exact decoded string content or a complete existing JSON
object/array regardless of JSON whitespace, key order or Unicode escaping; prose is
never normalized and separate values are not joined to manufacture a quote.
Malformed verification responses receive up to three attempts without spending the
candidate correction budget. Structural continuity errors still block immediately or
enter the bounded correction loop. Lost props remain tracked in ending states and stay
out of the visible opening frame; references to them do not imply reacquisition.

Generate plans scenes using those contracts, reviews event coverage, then projects
them into per-scene contracts. It propagates ending states and verifies passage-ending
states. The existing H3 generator and repairer receive these contracts. Final wording
is reviewed after camera processing and on cache hits. Conflicts in opening descriptions,
subject definitions or summary are errors, even when the action timeline is correct.

## Debugging

With Qwen thinking enabled, reasoning and JSON share the request's token budget.
If a response includes reasoning and reaches the length limit before returning valid
JSON, structured-JSON retries increase the budget to at least 16,384 tokens, then
double it up to 32,768 (an already larger requested budget is preserved). Thinking
stays enabled and the existing retry count still applies. LM Studio needs enough
context space for the input plus this output budget. If it still runs out, reduce
`chunk_chars`, increase LM Studio's output/context limit, or disable thinking in
LM Studio Configuration. Increasing `correction_attempts` alone does not give
the reviewer more tokens.

If continuity event extraction still ends at the output length limit after those
retries, it automatically switches to one event per JSON response. Each request
retains the full passage and accepted events; Python carries the running state,
checks each page and assembles the existing v1 contract. Opening entities appear
only on the first page, and later pages declare only newly encountered entities.
The assembled contract still receives full validation and semantic review;
corrections restart event extraction and remain in per-event mode. The validation
report records `extraction_mode: per_event`. Both narrative nodes and Generate's
scene-contract projection share this fallback; node sockets and saved bundle
formats remain unchanged. This costs more requests and reduces event output size,
but opening entities, input context, prose simplification and review responses
can still exceed model limits. An unfinished page sequence fails rather than
silently dropping remaining events.

Generate saves per-scene `narrative_state` and `continuity_review` in the existing chapter
`manifest.json`. Original chapter digests reject stale preprocessing. Prompt cache keys
include the contract and previous final prompts. Existing result sockets are unchanged.

## Internal schema

`minimax-cinematic-narrative.v1` contains chapters keyed by resolved source path;
each chapter has a source digest, cinematic text, reports and ordered segments.
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
The model returns an event-only extraction schema (`narrative_events_v2`): opening
entities for the first passage, declarations of later entities, and ordered events
with `source_ids` and field `after` values. It does not generate quotations,
`change.before`, `initial_frame` or `state_after`. Python slices evidence directly
from indexed original sentences/lines and derives all redundant state by replay.
Unknown source IDs, reversed source order and undeclared entities remain errors.
Events within the same source unit retain their supplied order; semantic review
checks within-sentence chronology, introductions and whether evidence supports changes.
For subsequent passages the opening comes exclusively from the previous ending.
New entities start with null physical attributes, `not_introduced` and invisible.
Duplicate IDs, ownership cycles and unsupported transitions are rejected.
The public persisted contract remains v1 for downstream and existing bundle compatibility.
State carries across passages/scenes within each chapter; each selected chapter starts
independently because chapter boundaries can contain time jumps or viewpoint changes.

Deterministic checks enforce schema/types, unique identities, chronological evidence,
event preconditions, unchanged carried state, introduction before visibility/ownership,
one location/owner, opening-frame consistency, explicit movement/posture changes and
event-replayed ending state. `intact → damaged → broken` is allowed. Restoring damaged
or broken objects requires an explicit repair event supported by the source.

Errors identify entity, expected state, conflicting state, introducing event and
suggested correction. When normalization review reports errors, a separate verification
request checks each complaint against the full original and candidate before correction.
Equivalent attribution, combined actions and actions already present are not errors.
Confirmed complaints require valid source IDs and exact candidate excerpts. Incomplete or malformed
verification responses receive up to two verification retries with specific feedback,
keeping the candidate and proposed findings unchanged. These retries do not consume
`correction_attempts` or trigger prose rewrites. If verification still fails, execution
stops with a separate `Review verification failed` error; unverified text is not accepted.
This adds one to three verification calls per flagged normalization candidate and still
relies on model interpretation. Confirmed errors retain the normal correction budget
and can stop execution.
Corrections receive original text, cinematic text, current state,
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
There is no dedicated loader for saved preprocessing bundles.

Offline tests mock LM Studio. `tests/fixtures/00_PROLOGUE.md` contains the supplied
French source, copied from the operator's ComfyUI input directory. Source-backed tests
check exact evidence and chronology: the holder is revealed from the backpack;
intending to store the torch precedes actual storage; rope damage precedes retrieval;
the torch moves between the teeth only afterward; and the rope breaks before Indy
hangs by one hand, loses his grip and finally falls. These tests complement the
smaller synthetic fixtures; mocked responses do not establish model interpretation quality.
Live ComfyUI/LM Studio model quality and generated videos require separate validation.
