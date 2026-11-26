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

Extraction reads the numbered sentences of each sequence's resolved
`adaptation.initialState` and `adaptation.event`. Opening passage IDs mark the
starting state. An entity first depicted during an event can have its established
wardrobe or appearance described there. Transformations during the clip belong in
the animation.

Each source observation records `presence` (`visible`, `offscreen`, `mentioned`)
and `visual_traits`. Every trait has a consistent attribute key, a source-language
description quoted from a selected local sentence and an `intrinsic` or `scene`
scope. Python
resolves selected passage IDs to exact sentence-level quotes with offsets in
source order. `evidence_passages` retains each selected sentence. The loader
checks addressed citations against their sequence. A bounded retry addresses
invalid IDs or response structure. Incomplete results retain valid observations
and record rejected fields in the diagnostic sidecar.

- New objects and filming spaces require visible appearance details. A described
  reference can be reused when the current passage identifies the same visible
  entity. A distinct fixture can be a prop. A constituent detail stays with its
  parent even when revealed by use. Local aliases require source-supported coreference.
- Objects record intrinsic design such as shape, construction, color, texture,
  markings and explicitly stated material. Established visible physical conditions
  and fixed installed geometry are local traits. Depicted light sources retain their
  physical design. Scene lighting, handling, incidental placement and
  transformations taking place during the clip remain in the cinematic adaptation.
- Depicted characters are recorded even without local appearance, and identifiable
  offscreen speakers are recorded for audio. Character observations describe only
  neutral wardrobe and footwear. Consolidation can match them to the whole-book
  character profiles supplied through `visual_designs.json` or the profile socket.
  Injuries, marks from interactions, dirt and torn garments stay in the adaptation.
- Locations record durable setting details and the visible conditions established
  for each sequence, including weather, fog and already visible physical changes to
  the backdrop. A fixed opening belongs to the setting geometry.

The chapter dictionary grows in sequence order. Extraction receives a bounded list
of known IDs, names, aliases and defining traits for identification. Local
sentences establish current presence and appearance. A shared generic label alone
does not confirm identity. A reused object cites its local name. A new alias also
needs an identifying appearance detail from the passage. The chapter audit checks
provisional identities and possible duplicates with cited merge groups. Unresolved identities remain separate
and are reported in diagnostics. Confirmed identities retain all observations at
`state_by_sequence[sequence].source`.

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
appearance through functional `excluded_attributes`. A source-grounded review can
retain multiple values under a scene attribute when the quotations establish
compatible spatial scopes. Those values stay local to the scene.

New objects and places require useful visual description. A known reference may have
visible observations without new traits. Visible characters may rely on their independent
Character Visual Profile Generator output, including `visual_designs.json`, for
physical identity. Offscreen speech is retained independently for audio needs.

To resume extraction in a new run, set `resume_run` on Extract Chapter References
to the previous timestamp and keep `out_dir`, source data and extraction settings
the same, with `force` disabled. The previous run's
`<out_dir>/.cache/<chapter_id>/` is read-only. Compatible completed extractions and
confirmed identity decisions are reused. Replaying the cached answers in sequence
order reconstructs the dictionary. Missing or invalid answers are requested again.
New files go to the current run.

## Consolidation and image binding

Each character, location and object in `visual_designs.json` has an
`exclude_image` flag. The default is `false`, including when the field is absent.
Set it to `true` and import the edited file through `visual_designs_path` in
Consolidate References to omit the entity's picture prompts and H3 picture bindings.
The registry keeps the entity, and eligible voices still receive audio prompts.
`image_prompts/image_prompts.json` lists the prompts for the current run.

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
