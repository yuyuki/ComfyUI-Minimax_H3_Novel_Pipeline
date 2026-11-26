# Normalized source text

Every chapter reader uses `core.source_text.read_source`. Text and extracted PDF
content are normalized before passage selection, model requests, hashing or
provenance calculation. The same shared normalization applies to model text.

Normalization uses Unicode NFC, straight apostrophes and quotation marks,
ordinary spaces for nonbreaking spaces, and LF line endings. Paragraphs, case,
accents and wording are retained. Whitespace-tolerant quotation matching anchors
accepted quotations back into this normalized source.

Character evidence format 3 stores the chapter file reference, normalized text
SHA-256, location basis, exact quotations, and their character and line
positions. The location basis is `normalized_utf8_text` or
`normalized_pdf_extracted_text`. Consolidation and sequence enrichment reopen
the original confined input file and verify its normalized fingerprint.
A missing, changed, or ambiguous source requires attention before its locations
can be used.

Regenerate older outputs in pipeline order with Character Evidence Extractor,
Consolidate Character Profiles, Character Visual Profile Generator, and
Consolidate References. Keep the original chapters accessible under ComfyUI's
input directory. Previous character response caches use a different fingerprint.

Character evidence includes short, explicit naming statements. The persistence
filter applies to appearance evidence and preserves identity evidence.
Consolidate Character Profiles verifies citations and locations and resolves
source-supported aliases. An unavailable semantic audit leaves deterministically
valid observations usable and records an incomplete audit in
`character_evidence_diagnostics.json`. Invalid evidence and failed decisions
are reported in the corresponding diagnostic files. Sequence-level findings
are recorded in `character_chapter_enrichment.diagnostics.json`.
