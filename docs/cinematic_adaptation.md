# Cinematic adaptation source fidelity

The cinematic adapter normalizes chapter text before windowing or sending it to
LM Studio. It applies the same rules to generated text, including source excerpts,
deferred source suffixes, descriptions and narrative proposals:

- Unicode NFC (canonically equivalent accented characters).
- Curly apostrophes `‘` and `’` become straight apostrophes `'`.
- Nonbreaking spaces U+00A0 and U+202F become ordinary spaces.
- CRLF and CR line endings become LF.

Paragraphs, repeated ordinary spaces, accents, case and other punctuation remain
significant. Source files are unchanged. Saved cinematic text uses normalized
typography; source progress and error offsets refer to the normalized chapter.

Sequence excerpts must still cover the complete normalized passage in order,
without omissions, repetitions or rewriting. Boundary whitespace is restored from
the reference passage. Saved `source` fields are slices of that reference, rather
than copies retained from model output. Deferred source must be an exact normalized
suffix of its reading window.

Copy failures report the first mismatching character. Retry prompts additionally
include bounded expected/received context to help correct the exact error.

Targeted semantic repairs replace existing sequences only after structural and
source validation. After bounded repair
failure, the adapter retains its last source-validated work and uses the existing
narrative proposal mechanism for remaining review issues. These issues and their
proposals remain in `narrative_review`; the first proposal is selected as usual.
Client/configuration failures and cancellation still propagate. Output-budget
failures while performing semantic review retain their explicit error behavior.

## Progressive checkpoints

Each chapter writes `<out_dir>/.cache/<index>_<chapter_name>/progress.json`.
The checkpoint is replaced atomically after source validation, semantic review,
each completed review batch, successful repair, proposal selection and scene commitment.
It includes committed public sequences, pending internal sequences, review issues,
source progress, model and node settings. Pending sequences are drafts, not final
or semantically accepted scenes. A review with remaining issues is also a draft.
Review still applies to a complete reading window and can revise its scene boundaries.

After window review and proposal resolution, each committed scene is saved as
`scene_001.json`, etc. These files survive later window failures or cancellation.
Only a fully completed chapter is written to the public `*.cinematic.json` output;
loaders therefore cannot mistake a partial chapter for a finished one. Checkpoints
can be reused with the optional `resume_run` field: enter the previous timestamped
parent folder name, for example `20261112124902`, and keep the same `out_dir`,
selected chapters, model and generation settings. A blank value starts fresh.
The previous run remains unchanged; restored checkpoints and scene files are
written under the current run. Completed scenes retain their numbering and final
states. Pending source-validated drafts resume semantic review without regeneration.
Changed source hashes, model controls or node settings are rejected, as are invalid
scene numbering and source coverage. Chapters not yet cached start normally.
Runs created before progressive caching cannot supply a resume checkpoint.
Existing runs must finish or be stopped and ComfyUI restarted to load this behavior.

## Avoiding repeated work

Positive semantic validations are cached per scene, using its source, descriptions,
both immediate neighbors and preceding final state. The key also includes model,
duration rules, review instructions/schema and token budget. A changed or split
scene invalidates its own validation and those of its neighbors automatically.
Unchanged distant scenes keep their approvals. Reviews of remaining contiguous
regions still receive the neighboring scenes as context, retain local target
numbering and split on excessive input/output size. Malformed reviews never
produce approvals. Positive validations are checkpointed between review batches
and restored by `resume_run` under the existing model/settings/input checks.

Semantic corrections regenerate only rejected source spans, with neighboring
source and states supplied as context. When corrections are enabled, one final
targeted repair round replaces the former whole-passage reconstruction. Changed
scenes and their neighbors are reviewed again; persistent doubts still produce
editable proposals without replacing unrelated scenes.

Output exhaustion or exhausted structural retries halves the reading window,
down to the existing 1,000-character recovery floor. Two source-copy failures on a
large window trigger this reduction early, even with a larger retry allowance.
The reduced window size is retained for the remaining chapter and checkpointed
for resumption. The source cursor, exact coverage validation and deferred suffix
remain authoritative; a window with no complete scene still requests a larger
configured context instead of silently losing source. No thinking controls,
schemas, semantic criteria or duration requirements are disabled.
