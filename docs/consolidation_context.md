# Consolidation context budget

Consolidate References exposes `context_tokens` for character evidence enrichment.
The default is 65,536. Set it no higher than the context loaded for the model in
LM Studio. This control does not resize the model's context or change the other
reference-generation requests. The configuration snapshot records its value.

The input budget subtracts `max_tokens`, instructions, the structured-output
schema and an 8,192-token retry reserve. Input sizing conservatively counts UTF-8
bytes, not tokenizer tokens. Raising `max_tokens` therefore reduces input capacity.

Oversized registry/profile identity comparisons split the profile's supporting
evidence into complete observations. Each batch retains the complete chapter
source passages, character catalog and registry identity context. Retries retain
the batch's evidence, and validated decisions are cached under the consolidation
output's `.cache/<chapter_id>/` directory.

Every batch must confirm identity with a validated current-source quotation before
the profile is imported. An uncertain or conflicting batch leaves the mapping
unresolved and triggers the existing mapping warnings; it is not overridden by a
positive batch. All batches are reviewed even after a negative decision. This can
leave identities unresolved when their proof requires observations across batches.

If the complete chapter/catalog context plus a single observation cannot fit,
execution reports the required input bytes, available budget and token settings.
Increase `context_tokens` only if LM Studio has sufficient loaded context, or reduce
`max_tokens`. Evidence is never truncated to force a request to fit.

After updating the plugin, restart ComfyUI and reload the workflow to expose the new
widget. The bundled ExtractChapter workflow explicitly sets it to 65,536.

## Resume a previous consolidation

Set `resume_run` to the previous run's 14-digit folder name, for example
`20261031202452`, and keep the same `out_dir`. Consolidation revalidates matching
cached requests from `<resume_run>/<out_dir>/.cache/` and copies valid hits into
the current run. The previous run stays unchanged. Reconciliation and character
enrichment caches use `.cache/<chapter_id>/`; shared audit, design and asset
requests use `.cache/` directly.

Matching depends on the request, schema, model and effective settings. Missing,
invalid or changed requests are recomputed. `force` ignores both caches. Leave
`resume_run` empty for a fresh run. Runs made before this feature can only reuse
the character enrichment caches they already contain. Restart ComfyUI and reload
the workflow after updating to expose the new controls.
