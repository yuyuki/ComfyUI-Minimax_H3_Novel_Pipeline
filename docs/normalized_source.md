# Normalized source text

Every chapter reader uses `core.source_text.read_source`. Text and extracted PDF
content are normalized before passage selection, model requests, hashing or
provenance calculation. The same shared normalization applies to model text.

Normalization uses Unicode NFC, straight apostrophes and quotation marks,
ordinary spaces for nonbreaking spaces, and LF line endings. Paragraphs, case,
accents and wording are retained. Whitespace-tolerant quotation matching anchors
accepted quotations back into this normalized source.

Character evidence format version 2 stores that exact working text in
`source_text`. Its SHA-256, character offsets and line numbers all refer to this
normalized text, not the input file's raw characters. `location_basis` is
`normalized_utf8_text` or `normalized_pdf_extracted_text`. There is no mapping
back to raw-file character positions. Input files are never rewritten.

Regenerate older character evidence, then its character profiles and visual
profiles. Old raw-source evidence is rejected rather than converted or rebased.
Regenerate cinematic chapters and downstream catalogs/references with the same
reader for a consistent source basis. Cache fingerprints include the source
normalization contract so previous cached model decisions are not reused.
