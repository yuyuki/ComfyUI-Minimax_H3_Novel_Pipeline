# Normalized source text

Chapter readers share `core.source_text.read_source`. Character evidence is
normalized before passage selection, model requests, hashing and provenance
calculation. The same normalization applies to generated descriptions. Cinematic
Chapter Adapter preserves the decoded text and line endings for exact `source`
spans and review quotations. For PDFs, its spans refer to the text extracted by
pypdf.

Normalization uses Unicode NFC, straight apostrophes and quotation marks,
ordinary spaces for nonbreaking spaces, and LF line endings. Paragraphs, case,
accents and wording are retained. Whitespace-tolerant quotation matching anchors
accepted quotations back into this normalized source.

Character evidence format 3 stores the chapter file reference, normalized text
SHA-256, location basis, exact quotations, and their character and line
positions. The location basis is `normalized_utf8_text` or
`normalized_pdf_extracted_text`. Consolidation reopens
the original confined input file and verify its normalized fingerprint.
A missing, changed, or ambiguous source requires attention before its locations
can be used.

Regenerate older outputs in pipeline order with Character Evidence Extractor,
Consolidate Character Profiles, Character Visual Profile Generator, and
Consolidate References. Keep the original chapters accessible under ComfyUI's
input directory. Previous character response caches use a different fingerprint.

Character Evidence Extractor reads all selected chapter text by default and
keeps enduring visual traits and explicit naming statements. It reviews the
passages around uncovered appearance cues and checks the extracted descriptions
in compact batches. Excluded or trimmed wording is recorded in the diagnostic
file. Identity evidence is retained for alias resolution.
Consolidate Character Profiles verifies citations and locations and resolves
source-supported aliases. A selected chapter can use an enduring trait attested
elsewhere when a unique confirmed name or alias identifies the profile there.
An unavailable semantic audit leaves deterministically
valid observations usable and records an incomplete audit in
`character_evidence_diagnostics.json`. Invalid evidence and failed decisions
are reported in the corresponding diagnostic files. Ambiguous registry
mappings are recorded in `consolidated_references.diagnostics.json`.
