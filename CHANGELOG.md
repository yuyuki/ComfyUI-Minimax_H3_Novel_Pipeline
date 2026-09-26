# Changelog

Summaries of all commit messages in the local Git history, newest first.
Related commits are grouped by commit date, with short hashes for traceability.
Dates and version bumps below describe repository history, not confirmed
Comfy Registry publication dates. Historical features may have been replaced
by later entries.

## Unreleased

- Add optional camera-only post-processing after H3 prompt validation, preserving
  existing shot structure and reporting rejected camera notes.
- Add scene-duration guidance, a configurable maximum shot count with pacing
  checks, and optional camera direction in H3 generation.
- Add operator-defined spatial anchors and per-scene LM Studio continuity
  resolution, using future scenes and prior final states when generating H3
  prompts; include offline regression tests.
- Add this changelog and require updates before each push to `master` or `main`.
- Pass the complete changelog to Comfy CLI when publishing so new registry
  versions include it in their `changelog` field.

## 2026-09-23

- Use the trusted LM Studio API URL and key in settings and improve connection
  handling (`451d8db`).

## 2026-09-22

- Add a check-only publishing option and a registry status checking tool
  (`c47a22f`).

## 2026-09-18

- Add generated architecture analysis and merge its pull request
  (`f8696f4`, `adf2c1b`).

## 2026-09-17

- Preserve off-screen dialogue replies and source order during scene selection
  (`78c1c8d`).
- Require non-empty audio brief descriptions and generation prompts (`28894c5`).
- Enforce continuous shots and refine pacing instructions and shot validation
  (`2f71443`, `cdebf1f`).
- Add asset sheets and copy-paste scene prompts (`81353ac`).
- Test subject definitions containing whitespace and colons (`2325fed`).
- Bump versions to 1.10.3, 1.10.4 and 1.10.5, including author metadata updates
  (`ab7849c`, `44b937b`, `14cda86`).

## 2026-09-16

- Improve view-specific appearance rules, facial appearance and identity
  projection, with regression coverage (`57beaf6`, `8ea0587`).
- Snapshot execution settings and normalize prompt appearances (`3d1c36a`).
- Handle execution results for ComfyUI progress and test workflow completion
  (`1f7bf7c`).
- Correct project repository URLs (`fb7db59`).
- Bump versions to 1.10.0, 1.10.1 and 1.10.2, including author metadata updates
  (`bc36ba3`, `da2365b`, `b2354c3`).

## 2026-09-13

- Add pipeline progress tracking, completion times and improved context handling
  (`d5325b6`, `b8ef17f`).
- Improve H3 prompt normalization and formatting validation (`5f63786`).
- Improve extraction logging and Qwen model handling (`0718396`).
- Validate canonical names and test blank names against user message content
  (`dbc0db6`, `30b1dca`).
- Refine extraction parameters and compact retry schemas to preserve entity
  details and visual information (`431ad19`).
- Add Qwen/Mistral model-family selection, model profiles, request settings,
  cache fingerprints, documentation and migration tests (`43b2cf6`).
- Bump versions to 1.5.0, 1.8.0 and 1.9.0
  (`8e2b5f8`, `15d47f9`, `995ae8f`).

## 2026-09-12

- Remember successful ChatML preferences and improve LM Studio JSON error
  handling and retries, with tests (`84f73c3`, `177bf87`).
- Add consolidation registry summaries and improve visual design handling
  (`8c3b376`).
- Bump the version to 1.4.0 (`791c2a1`).

## 2026-09-11

- Add deterministic image prompt exports, editable visual designs, prompt/cache
  fingerprints, bounded semantic retries and execution output directories;
  refactor the pipeline and add regression coverage (`c358700`).

## 2026-09-06

- Introduce the Select Chapters node and replace `saved_chapter` inputs with
  `chapter_selection` in extraction and generation; update mappings,
  documentation and tests (`f9fe34b`, `f29ab32`, `54f7823`, `7619806`).
- Clarify the thinking control and remove manual output-token and Qwen chunk
  controls, updating token/chunk handling and documentation
  (`6219e5c`, `31cd260`, `2f8f25b`).
- Bump the version to 1.0.0 (`2cf5115`).

## 2026-09-05

- Add pipeline nodes, shared bundled execution, local route restrictions,
  scene selection with H3 reference ordering, file/JSON helpers, chapter-picker
  and API-key frontend controls, examples and credential/access/pipeline tests
  (`4b1d80b`, `a70c316`, `943ed71`).
- Confine filesystem paths, validate Windows filenames, improve path errors
  and output-directory handling, and test upload preservation (`1922178`).
- Upgrade catalog/reference schemas to v3, strengthen schema and API error
  handling, return prompt text from generation, remove the deprecated scene
  selection node and add validation tests (`44febfd`).
- Remove the model parameter from configuration and related nodes (`390cc0b`).
- Clarify workflow documentation and streamline agent instructions (`3283571`).

## 2026-09-03

- Refine plugin structure, LM Studio configuration, chapter extraction and
  Qwen3.5 parameters (`cb6cbc9`).
- Fix chapter selection validation with an empty enum for single-file fallback
  (`a780836`).
- Refactor node return types and output handling; update project metadata
  (`d54d5cb`).
- Add Comfy Registry publishing, project configuration/dependencies and the
  root ComfyUI entrypoint; revert the package version to 0.9.0
  (`f4f6060`, `abed170`, `106bf80`, `73252f2`).
- Add v2.4.1 installation/workflow documentation and requirements, then remove
  the README and requirements from the historical flat bundle
  (`fd697d7`, `10a9314`).
- Remove legacy migration functions (`9ca6843`).
- Handle ComfyUI Stop requests in JSON chat across pipeline stages (`3f018b9`).

## 2026-08-31

- Improve the saved chapter picker, upload validation and legacy migration;
  use an on-completion callback for chapter deletion and dialog handling
  (`dfc0414`, `dd21723`).

## 2026-08-30

- Introduce the MiniMax H3 Novel plugin: chapter processing, H3 prompt
  generation, saved catalog/reference loaders, JavaScript chapter selection,
  file/JSON utilities, compatibility mappings and dependencies (`1602fc0`).
- Refactor extraction, generation and shared pipeline execution; add ordered
  scene selection, concise catalog/reference summaries and workflow
  documentation (`a288297`).
- Add chapter deletion and improve input handling (`1ebe76b`).
- Immediately select uploaded chapters and synchronize the chapter paths
  widget (`d95231f`).
