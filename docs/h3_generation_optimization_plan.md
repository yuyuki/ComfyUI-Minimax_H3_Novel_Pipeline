# H3 generation optimisation plan

Date: 2026-11-10. Status: proposed; implementation has not started.

## Objective and boundaries

Reduce unnecessary LM Studio requests and repeated context processing while
preserving scene coverage, source fidelity, dialogue, bindings, camera quality,
reasoning settings, and final format and continuity checks. Keep existing node
IDs, sockets, output layout, cancellation and structured-JSON transport.

Do not change or interrupt the active run, overwrite upstream outputs, or modify
the user's existing character-profile work. Implement in small, separately
validated changes. No smaller model, lower token budget, fewer scenes, shortened
evidence, or disabled camera/review pass is part of this plan.

## Measured baseline

Snapshot from `20261110202656/h3_prompts` at approximately 21:10 Brussels time:

| Stage                                    | Completed-call time | Underlying model requests |
| ---------------------------------------- | ------------------: | ------------------------: |
| Planning                                 |               528 s |                        15 |
| Generation                               |                76 s |                         2 |
| Format repair                            |               166 s |                         4 |
| Continuity review, including its repairs |             1,802 s |                        58 |

Fifteen sequences produced 41 planned scenes. Scene 2 was being reviewed; scene
1 was saved as invalid (summary prefix and description length). The metrics are
cumulative completed-request measurements, not a complete run benchmark.

Observed review feedback asks scene 2 to include actions explicitly assigned to
scenes 3 and 4. Other feedback asks for metadata or binding changes that the
prompt-only repair cannot make. These are correctness problems as well as speed
problems. Existing exact-request checkpoints already avoid identical calls;
adding another cache without fixing dependencies and request construction is
not sufficient.

## 1. Establish measurement and replay fixtures

- Add explicit scene, stage, review round, batch, attempt and in-flight timing
  to progress metrics. Separate continuity repairs from continuity review calls.
- Record cache hits/misses and invalidation reasons, request bytes, estimated
  tokens (labelled as estimates), retry counts and final validation state.
  Record backend-reported usage separately when available; never record keys.
- Build small synthetic regression fixtures reproducing the observed sequence
  split, malformed prefix, invalid binding feedback and distant contradictions.
- Add an offline batch-count comparison using read-only copies of representative
  input structures. Do not run a second inference workload against the busy
  server or treat cached timings as fresh inference performance.

Acceptance: metrics distinguish useful generation, formatting, semantic repair,
review and retries; tests reproduce the problematic control flow before fixes.

## 2. Compile upstream information once, without an LLM

Create a deterministic, immutable evidence bundle for the selected chapter from
the connected, validated node outputs. Reuse information rather than asking a
model to rediscover identities, phase observations or approved spatial staging.

| Existing information                                         | Reuse in H3 generation                                               |
| ------------------------------------------------------------ | -------------------------------------------------------------------- |
| Consolidated chapter timeline and selected narrative choices | Ordered sequence phases, exact events and source provenance          |
| Entity registry and chapter mappings                         | Stable global IDs, names and identity resolution                     |
| Phase-specific observations and character enrichment         | Visibility, speech presence and applicable character state           |
| Approved visual designs and reference assets                 | Appearance constraints, available views and deterministic bindings   |
| Spatial Continuity output                                    | Explicit/inferred layout, anchors, state transitions and uncertainty |
| Existing compatible H3 scene plans                           | Beat selection, dialogue assignment and ordering                     |

- Validate schemas, source digests, selected choices and referenced IDs before
  reuse. Unknown or absent facts stay unknown; upstream inference is not promoted
  to source evidence. Preserve phase boundaries and distinct prop instances.
- Give evidence records stable IDs with chapter, sequence, phase, entity and
  provenance. Include the full original evidence, not an LLM summary.
- Do not automatically skip H3 beat planning: an upstream sequence is not
  necessarily an eight-second scene. Reuse a compatible saved beat plan; a
  deterministic no-model planning path is allowed only where an explicit input
  contract already supplies all required scene decisions.
- Keep the bundle in the generation node's own chapter cache. Consume upstream
  payloads; do not browse unrelated run folders or write into upstream caches.

Acceptance: changing a selected upstream fact invalidates affected artifacts;
unchanged inputs produce the same bundle; compilation makes zero model calls.

## 3. Make scene scope explicit and route repairs correctly

- Represent each planned scene's assigned beat, exact dialogue/speaker,
  starting/ending state and sibling order separately from sequence-wide context.
  Retain full contextual evidence and identify neighbouring beats explicitly.
- Tell generation and review to check only the assigned beat for completeness.
  Neighbouring events constrain ordering; their absence from the current clip
  is not an omission. Do not add later actions to satisfy a mistaken review.
- Use a strict structured issue schema with target (prompt, scene plan, binding
  or upstream conflict), evidence references and requested correction.
- Validate issue references locally. Route prompt issues to prompt repair.
  Derive metadata from explicit beat evidence where unambiguous; otherwise use
  a bounded targeted planning repair. Rebuild bindings deterministically from
  approved registry assets after an authorised scene-plan correction.
- Never silently change an approved upstream choice or invent assets. Report an
  irreconcilable upstream conflict as unresolved; preserve diagnostics.
- Invalidate downstream results affected by plan/binding changes. Bound repair
  cycles and detect repeated unchanged issues; never convert exhaustion into
  success. Invalid scenes must remain visibly invalid.

Acceptance: the call-for-help scene is not required to gag or pull upward;
genuine early actions, wrong dialogue, incorrect bindings and distant spatial
conflicts still fail. No metadata issue is sent to a text-only repair loop.

## 4. Run cheap validation before expensive semantic review

Use this processing order:

1. Reuse or generate the scene draft.
2. Apply narrowly defined, idempotent formatting normalisation.
3. Run local format, reference-label and timing validation.
4. Perform bounded format repair if necessary, then validate again.
5. Refine camera when enabled, followed by local validation/repair.
6. Review the actual final wording against all required continuity evidence.
7. Route semantic issues; after any correction, repeat local validation before
   semantic review. Accept only when all required checks pass.

Safe deterministic repair includes adding brackets around an exact recognised
task prefix when it agrees with the actual audio bindings. Do not guess task
types, fabricate words to meet a length target, or mechanically alter meaning.

If local errors remain after the allowed repair budget, checkpoint the invalid
draft and report it rather than spending full-chapter review calls on it. A
changed candidate must pass final semantic checks; an old approval cannot carry
over to new wording. Avoid replaying camera refinement when its exact inputs
and validated output are unchanged.

Acceptance: known-invalid drafts trigger no continuity calls; valid repaired
drafts still receive full review; camera changes cannot bypass validation.

## 5. Reduce context size losslessly

- Extend existing evidence interning to exact repeated source excerpts,
  reference descriptions and repeated instructions. Keep occurrence, phase and
  entity relationships even when their text is shared.
- Use compact canonical JSON and stable IDs/order. Each request must include
  every referenced definition; references cannot depend on model memory across
  independent calls.
- Build stable evidence batches and cache their deterministic construction.
  Preserve all previous valid prompts and future requirements, including distant
  relationships. Do not substitute a nearest-scenes window or semantic summary.
- Keep conservative size limits initially. Consider a larger budget only after
  verifying the loaded model's context and reserving completion/reasoning space.
  Do not infer available context from the model name.

Acceptance: reconstruction recovers the original evidence exactly, no records
are dropped, all references resolve, and representative requests use fewer
bytes/batches. No later-batch error or failure can be reported as overall success.

## 6. Cache stages by their actual dependencies

Retain the existing exact-request checkpoint layer. Add explicit artifact
dependencies for evidence, planning, bindings, generation, camera and reviews.

- Replace broad H3 artifact keys with stage-specific settings and dependencies.
  A camera/review setting must not invalidate beat planning when it cannot affect
  that plan. Include relevant prompts, schemas, validators, helpers and model
  transport/profile semantics; avoid hashing unrelated whole-module code into
  every stage. Test invalidation rather than relying on manual version bumps alone.
- A review approval is valid only for the exact final prompt, bindings, current
  scene, all context in that batch, constraints, review contract and effective
  model settings. Cache both results and the complete coverage manifest.
- Reuse unchanged approved batches on resume. If prompt wording changes, all
  its semantic review batches become stale. If only one evidence batch changes,
  reuse others only when their full dependencies and overall coverage still match.
- Reuse upstream-derived bundles across a selected resume run only when digests
  match. Copy reused artifacts into the new run so it can itself be resumed.
- Keep caches under `<timestamp>/<out_dir>/.cache/<chapter_id>/`; use atomic
  writes, integrity checks and cancellation-safe checkpoints. Preserve `force`.
- Reject incompatible cache records and recompute; do not add legacy formats or
  fallback paths. Changes to review semantics necessarily rerun those reviews.
  Existing exact-request records remain reusable only if still contract-compatible.

Acceptance: unchanged resume makes zero calls for completed compatible work;
changing camera settings reuses planning; changed dialogue, upstream choices,
bindings, backend settings or review rules invalidate all affected results;
corrupt/partial/invalid results cannot be accepted as approved artifacts.

## 7. Verification and implementation order

Implement in this order: measurement/fixtures; scene-scoped review and repair
routing; local-validation gate; upstream bundle and lossless packing; granular
artifact caching. Cache correctness follows the corrected contracts, not vice versa.

Primary implementation areas: `pipeline_step3_generate.py`,
`continuity_context.py`, `generation_requests.py`, and small focused core helpers.
Keep `nodes/generate_h3_prompts.py` thin. Avoid changing the shared prompt-cache
behaviour for unrelated nodes merely to optimise H3.

Extend focused coverage in `test_h3_prompt_validation.py`,
`test_generation_timeline.py`, `test_generation_spatial_regressions.py`,
`test_continuity_context.py`, `test_generation_requests.py`,
`test_generation_resume.py`, `test_camera_refinement.py` and
`test_spatial_continuity.py`. New LLM repair requests must use the shared client
and structured transport with explicit strict schemas on initial calls/retries.

Run focused tests and then the full offline suite and lint outside the sandbox,
as required by repository instructions. Regenerate the architecture map if
modules/imports change; run packaging checks if package layout changes.

After offline checks, perform an isolated live comparison when the active server
is available, without overwriting the current run. Keep model, reasoning, token
budget, scene/shot counts and quality checks identical. Compare request counts,
bytes, retries, cache reuse, wall time, valid-scene rate, exact dialogue and story
order, asset bindings and distant continuity. Include both the problematic first
sequence and a later scene with distant constraints. Follow with a full-chapter
run to expose growth in review cost.

No fixed speedup is promised before this comparison. Completion requires fewer
unnecessary calls/context bytes with all coverage preserved, no newly missed
regression conflicts, and live output quality assessed separately from tests.

## Deferred options

Parallel independent review batches may help only if measured LM Studio
throughput improves. Do not parallelise dependent scenes or assume one GPU
benefits from more requests. Defer concurrency until serial waste is removed;
it requires correct context propagation, cancellation and atomic metrics.

Disabling thinking/reviews/camera, reducing scene count, changing model or
summarising away evidence is outside this quality-preserving fix.
