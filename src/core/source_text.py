"""One normalized source basis for reading, model text and persisted locations."""
import hashlib
import json
import re
import unicodedata

from .path_access import input_path

VERSION = "normalized-source.v1"
PASSAGE_VERSION = "source-passages.v2"
PASSAGE_IDS = {"type": "array", "minItems": 1, "uniqueItems": True,
               "items": {"type": "string", "minLength": 1}}


_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00ab": '"', "\u00bb": '"', "\u00a0": " ", "\u202f": " ",
    "\u2028": "\n", "\u2029": "\n\n",
})


def normalize(text, *, collapse_whitespace=False):
    """Canonical typography and NFC, without changing case, accents or words."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n").translate(_TYPOGRAPHY))
    return " ".join(text.split()) if collapse_whitespace else text


def comparison_text(text):
    """Normalize both sides of a source comparison, including whitespace runs."""
    return normalize(text, collapse_whitespace=True)


def normalize_data(value):
    """Normalize model-owned text values without changing JSON keys or structure."""
    if isinstance(value, str):
        return normalize(value)
    if isinstance(value, list):
        return [normalize_data(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_data(item) for key, item in value.items()}
    return value


def read_source(path):
    """Decode a chapter once; all subsequent offsets refer to this normalized text."""
    path = input_path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        basis = "normalized_pdf_extracted_text"
    elif path.suffix.lower() in {".txt", ".md", ".markdown"}:
        text = path.read_bytes().decode("utf-8-sig")
        basis = "normalized_utf8_text"
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    text = normalize(text)
    if not text.strip():
        raise ValueError(f"Chapter {path.name} has no text; supply UTF-8 text or a searchable PDF.")
    return text, basis


def _mapped_text(text):
    """Map whitespace-tolerant matches to the normalized source, never the raw file."""
    chars, starts, ends = [], [], []
    for token in re.finditer(r"\s+|\S", text):
        chars.append(" " if token.group().isspace() else token.group())
        starts.append(token.start())
        ends.append(token.end())
    return "".join(chars), starts, ends


def source_spans(text, passage):
    """Find overlapping passages at offsets in the normalized source."""
    text = normalize(text)
    needle = comparison_text(passage)
    if not needle:
        return []
    haystack, starts, ends = _mapped_text(text)
    spans = []
    position = haystack.find(needle)
    while position >= 0:
        end = position + len(needle)
        spans.append((starts[position], ends[end - 1]))
        position = haystack.find(needle, position + 1)
    return spans


def source_span(text, passage):
    """Return a normalized-source range only when the passage is unique."""
    spans = source_spans(text, passage)
    return spans[0] if len(spans) == 1 else None


def contains(text, passage):
    """Test nonempty source membership with the shared comparison policy."""
    needle = comparison_text(passage)
    return bool(needle) and needle in comparison_text(text)


# East Asian text may contain compound actions without whitespace boundaries.
_UNSPACED = r"\u3040-\u30ff\u3400-\u9fff"
_SOURCE_UNIT = re.compile(rf"[{_UNSPACED}]|[^\s{_UNSPACED}]+")


def source_units(text):
    """Return token matches with positions in the exact input string."""
    return list(_SOURCE_UNIT.finditer(text))


class PassageIndex:
    """Select exact slices by occurrence within a particular authoritative text."""

    def __init__(self, text, scope, *, offset=0, whole_sentences=False):
        self.text = text
        self.offset = offset
        key = hashlib.sha256(f"{PASSAGE_VERSION}\0{scope}\0{offset}\0{text}".encode()).hexdigest()[:16]
        self.ranges = {}
        if whole_sentences:
            sentences = re.finditer(r"\S.*?(?:[。！？]|[.!?…](?=\s|$)|(?=\n|$))", text)
            self.ranges = {f"{key}:{number}": sentence.span()
                           for number, sentence in enumerate(sentences, 1)}
            return
        clauses = re.finditer(r"\S.*?(?:[。！？；，、]|[.!?;,:…](?=\s|$)|(?=\n|$))", text)
        for clause in clauses:
            words = source_units(clause.group())
            for start in range(0, len(words), 18):
                end = min(len(words), start + 18) - 1
                left, right = clause.start() + words[start].start(), clause.start() + words[end].end()
                self.ranges[f"{key}:{len(self.ranges) + 1}"] = (left, right)

    def passages(self):
        return [{"id": identifier, "text": self.text[start:end]}
                for identifier, (start, end) in self.ranges.items()]

    def resolve(self, ids, *, allow_empty=False):
        records = {key: {"start": self.offset + start, "end": self.offset + end, "text": self.text[start:end]}
                   for key, (start, end) in self.ranges.items()}
        return select_passages(records, ids, allow_empty=allow_empty)


def passage_catalog(sources, scope):
    """Address passages across distinct source texts and retain each one's provenance."""
    catalog = {}
    for provenance, text in sources:
        label = json.dumps(provenance, ensure_ascii=False, sort_keys=True)
        offset = provenance.get("offset", 0)
        index = PassageIndex(text, scope + ":" + label, offset=offset)
        for passage in index.passages():
            start, end = index.ranges[passage["id"]]
            catalog[passage["id"]] = {**{k: v for k, v in provenance.items() if k != "offset"},
                                      "start": offset + start, "end": offset + end, "text": passage["text"]}
    return catalog


def select_passages(catalog, ids, *, allow_empty=False):
    if (not isinstance(ids, list) or (not ids and not allow_empty)
            or any(not isinstance(value, str) or value not in catalog for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Select distinct passage IDs from the authorized source.")
    return [catalog[value].copy() for value in ids]


def offered_passages(catalog):
    return [{"id": key, **value} for key, value in catalog.items()]


def contiguous_passage(text, passages):
    """Resolve adjacent selections into one exact citation for single-span consumers."""
    if not passages or any(left["end"] > right["start"] or text[left["end"]:right["start"]].strip()
                           for left, right in zip(passages, passages[1:])):
        raise ValueError("Select consecutive passages in source order for this citation.")
    start, end = passages[0]["start"], passages[-1]["end"]
    return {"start": start, "end": end, "text": text[start:end]}


def paragraph_chunks(text, size):
    """Split at paragraph boundaries and keep exact offsets in the original text."""
    start = end = 0
    for boundary in [*(match.end() for match in re.finditer(r"\n[^\S\n]*\n+", text)), len(text)]:
        if end > start and boundary - start > size:
            yield start, text[start:end]
            start = end
        end = boundary
    if end > start:
        yield start, text[start:end]
