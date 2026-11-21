# Neutral visual reference contract

Status: implemented with offline regression coverage. Live model quality must be
verified separately; existing generated outputs have not been rewritten.

## Purpose

Reference images provide reusable visual appearance for entities relevant to each
cinematic scene. They are not staged opening frames. H3 determines gestures,
expressions, poses, object manipulation and narrative orientation during animation.
System prompts remain generic across books and genres. Particular names, objects,
scene numbers and passages belong in input data, never story-specific instructions.

## Extraction and identity

Extraction reads each sequence's resolved `adaptation.initialState` and
`adaptation.event`. `reference_state` supplies initial-condition context, without
prescribing image composition. An entity first depicted during an event can have
its established wardrobe or appearance described there. Results of transformations
to animate must not be treated as already completed appearance.

Each source observation records `presence` (`visible`, `offscreen`, `mentioned`)
and `visual_traits`. Every trait has a consistent attribute key, a source-language
description, `intrinsic` or `scene` scope, and local source evidence. Extraction
and correction requests select passage IDs. Python resolves them to exact quotes
with offsets in the combined initialState/event source. The existing quote field
contains the first selected quote, and `evidence_passages` retains each selected
passage individually. The catalog loader checks addressed citations against their
sequence. Before extraction is accepted,
semantic review checks relevance, neutrality, language and chronology. An individual
trait rejected by the review is dropped if correction cannot resolve the issue.
Chapter context disambiguates referents and categories only; it cannot establish
local presence or import changing traits. Actions remain in adaptation.

- Objects record intrinsic design such as shape, construction, color, texture,
  markings and explicitly stated material. Damage, tension and handling stay in
  the cinematic adaptation.
- Characters record only neutral wardrobe and footwear at the sequence address.
  Character visual profiles supply timeless physical appearance. Injuries, marks
  from interactions, dirt and torn or perforated garments stay in the adaptation.
- Locations record durable setting details and the visible conditions established
  for each sequence, including light, weather, fog and already visible changes to
  the backdrop. A fixed opening belongs to the setting geometry. Transient light
  cast by a moving source and ongoing transformations remain in the adaptation.

Chapter identity decisions group confirmed physical identities and retain their
observations at `state_by_sequence[sequence].source`. Generic labels do not prove
identity. Incomplete identity decisions remain explicit in diagnostics; uncertain
physical instances stay separate rather than being silently merged.
An uninterrupted action can establish that a contextual alias or a visible wall
belongs to the same space. A visible location can be recorded with empty local
traits and use supported shared geometry once its identity is confirmed.

## Shared traits and local states

Each identity has `visual_reference.common_traits` and
`visual_reference.scenes[chapter_id][sequence]`, containing presence, local traits and ambiguous attributes to exclude.
Only unambiguous intrinsic traits are shared. A later explicit clarification may
enrich an earlier appearance of the same confirmed entity when it describes an
unchanged intrinsic property. Description reuse never establishes scene presence.

For example, a material established for one rope is retained wherever that same
rope is visible even if later scenes do not repeat it. Another rope retains its
own identity and material. A misty backdrop and a clear view of the same location
can use different images with the same durable setting details.

Resolved appearance combines shared traits with applicable local overrides by
attribute. Unchanged attributes remain available. Local states apply only to their
attested scenes; the code does not guess persistence through omitted observations.
Alternative intrinsic wording is reviewed through a cached structured request when
one attribute has multiple descriptions. Only confirmed equivalent supplied strings
can share a representative; the model cannot invent replacement prose. Actual
conflicts or failed reviews remain source-local with incomplete diagnostics.
Ambiguous local values and their conflicting baseline fallback are omitted from image
appearance through functional `excluded_attributes`. Attribute keys and semantic classification are model decisions, so
source-grounded review remains necessary.

Objects and places require useful visual description. Name-only entries do not
produce empty image references. Visible characters may rely on their independent
Character Visual Profile Generator output, including `visual_designs.json`, for
physical identity. Offscreen speech is retained independently for audio needs.

## Consolidation and image binding

Consolidation recompiles shared and scene-local traits after global identity
reconciliation and duplicate audit, then integrates validated character profiles.
Profile-derived presence can further restrict character image applicability.

Image specs group identical resolved appearances across scenes. A distinct
appearance receives a content-based state variant; a new scene number or action
alone does not create a new image. Multiple useful views are not duplicate identities.
Objects default to a clear straight-on view of their defining surface; supported
view hints can select a more informative geometry view. Additional views expose
supported geometry or details. Source narrative orientation does not
prescribe the reference camera. Appearance normalization preserves distinct useful
facts, with a 12,000-character response ceiling instead of a fixed word target.

Every image has explicit `scene_bindings`: chapter IDs mapped to sequence numbers.
H3 binding selects only images valid for the exact visible entity and sequence.
Chapter membership or a base variant alone is insufficient. An absent entity cannot
become visible because its shared description exists.
Consolidation plans the fewest location views needed for the approved sequences,
up to the configured view limit. The briefs describe camera placement relative
to stable landmarks. If two views serve a scene, H3 can group them under one
location subject. The opening image remains a separate MiniMax input. Missing
location coverage is recorded in the scene diagnostic JSON.

## Persistence and validation

Chapter and consolidated contracts are v5 with the current
`cinematic-neutral-references.v2` timeline revision. Old catalogs and registries
must be regenerated. The project version is unchanged.
The former standalone source-only extraction and hierarchical catalog merge path
have been removed. Extraction uses the cinematic sequence path and source-grounded
chapter identity consolidation; their cache, cancellation and progress tests cover
the current implementation.
Loaders validate current visual observations and reference applicability. Technical
conflicts and incomplete checks stay in associated diagnostic sidecars, including
`consolidated_references.diagnostics.json`.

Offline coverage includes unchanged material reuse, supported later clarification,
local backdrop conditions, conflicting traits, distinct physical instances, name-only objects,
neutral frontal views, state-specific image selection, local evidence validation,
structured retries and generic prompts. The full pipeline tests also cover saved
outputs, character-profile integration, speech, cancellation, caches and access rules.

The implementation is in `reference_visuals.py`, the extraction and chapter-identity
modules, the consolidation engine, timeline helpers, schemas, loaders and H3 bindings.
The generated [architecture map](architecture.md) records the active dependencies.
