---
name: update-minimax-architecture
description: Refresh and verify docs/architecture.md in ComfyUI-Minimax_H3_Novel_Pipeline using its architecture generator. Use when asked to update the architecture map or synchronize it after Python module, import, or pipeline-flow changes in this repository.
---

# Update MiniMax architecture

Keep `docs/architecture.md` reproducible from `tools/generate_architecture.py` and consistent with the active package.

## Locate and inspect

Use the repository containing this skill: its root is three directories above this skill folder. Confirm that `tools/generate_architecture.py` and `docs/architecture.md` exist there. Run commands from that root.

Read applicable `AGENTS.md` instructions, inspect `git status --short`, and review existing diffs for the document and generator before writing. Preserve user edits. Read the generator, `tests/test_architecture_map.py`, and the current map.

Verify actual paths against the checkout and `pyproject.toml`: the inspected repository maps the installed package `minimax_h3_novel_pipeline` to flat `src/`, while older architecture notes refer to `src/minimax_h3_novel_pipeline/`. Do not move code to reconcile stale descriptive paths.

## Refresh the map

For an ordinary refresh, run:

```powershell
python tools/generate_architecture.py
```

The entire document is generated. Change its generator for persistent prose or diagram corrections, then regenerate; do not hand-edit the output. If the document already contains uncommitted manual edits, account for their intent in the generator before overwriting them; clarify only if that intent is materially ambiguous.

The current scanner parses top-level `src/*.py` with AST, records internal relative imports and public top-level classes/functions, and sorts its output. It does not infer all runtime dependencies. The pipeline-flow diagram is maintained explicitly in `render_architecture`; check the relevant node wrappers, registrations and bundled stages when the requested change affects that flow. Do not invent dynamic-import edges or claim the diagram proves runtime behavior. Historical `external source/` bundles are excluded.

If an actual layout change makes scanning incomplete, update the generator and add focused regression coverage for the affected behavior before regenerating. Do not change application behavior or packaging merely to refresh documentation.

## Verify

Run:

```powershell
python tools/generate_architecture.py --check
python -m pytest tests/test_architecture_map.py
```

Review `git diff -- docs/architecture.md tools/generate_architecture.py tests/test_architecture_map.py` for accurate module paths, dependency arrows, symbols, counts and pipeline flow. An unchanged map is a valid result; do not force a diff.

For generator edits, run relevant regression tests and `ruff check tools/generate_architecture.py tests/test_architecture_map.py`. If the authorized task includes layout or packaging changes, also run the repository-required `python -m pytest`, `ruff check .`, and `python -m build`. Keep offline checks distinct from live ComfyUI/LM Studio validation.

Report completion and check results concisely, including any blockers. Commit or push only when requested; if pushing, follow the repository's changelog maintenance instructions.
