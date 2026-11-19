"""One normalized source basis for reading, model text and persisted locations."""
import re
import unicodedata

from .path_access import input_path

VERSION = "normalized-source.v1"


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


def prefix_span(text, passage):
    """Match the next ordered excerpt, ignoring leading source whitespace only."""
    text = normalize(text)
    for start, end in source_spans(text, passage):
        if not text[:start].strip():
            return start, end
        break
    return None


def suffix_span(text, passage):
    """Match a deferred suffix on the normalized source, including its whitespace."""
    text = normalize(text)
    for start, end in reversed(source_spans(text, passage)):
        if not text[end:].strip():
            while start > 0 and text[start - 1].isspace():
                start -= 1
            return start, len(text)
        break
    return None


def contains(text, passage):
    """Test nonempty source membership with the shared comparison policy."""
    needle = comparison_text(passage)
    return bool(needle) and needle in comparison_text(text)


def restore_quote(text, passage):
    """Return a citation from the normalized source when occurrences agree."""
    text = normalize(text)
    quotes = {text[start:end] for start, end in source_spans(text, passage)}
    return quotes.pop() if len(quotes) == 1 else None
