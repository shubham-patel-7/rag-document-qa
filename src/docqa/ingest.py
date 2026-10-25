"""PDF -> pages -> overlapping chunks that remember (document, page)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config as C


@dataclass
class Chunk:
    id: int
    doc: str     # file name, e.g. "j1939_primer.pdf"
    page: int    # 1-based page number
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    """Normalise PDF text: re-join hyphenated line breaks, collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list:
    return [s for s in _SENT_SPLIT.split(text) if s]


def chunk_page(text: str, max_words: int = C.CHUNK_WORDS, overlap_words: int = C.OVERLAP_WORDS) -> list:
    """Greedy sentence packing up to `max_words`, repeating ~`overlap_words` of context between chunks.

    Chunks never cross a page boundary, so every chunk maps to exactly one citable page.
    A single sentence longer than `max_words` becomes its own chunk.
    """
    sents = split_sentences(clean_text(text))
    chunks, cur, cur_words = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and cur_words + w > max_words:
            chunks.append(" ".join(cur))
            # carry trailing sentences as overlap
            keep, kw = [], 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if kw + pw > overlap_words:
                    break
                keep.insert(0, prev)
                kw += pw
            cur, cur_words = keep, kw
        cur.append(s)
        cur_words += w
    if cur:
        tail = " ".join(cur)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


def read_pdf_pages(path: Path) -> list:
    """Return [(page_number, text)] for a PDF. Pages without extractable text are skipped."""
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            out.append((i, text))
    return out


def load_chunks(sources, max_words: int = C.CHUNK_WORDS, overlap_words: int = C.OVERLAP_WORDS) -> list:
    """Build chunks from a directory, a PDF path, or an iterable of (name, file-like/bytes/path)."""
    import io

    items = []
    if isinstance(sources, (str, Path)):
        p = Path(sources)
        files = sorted(p.glob("*.pdf")) if p.is_dir() else [p]
        items = [(f.name, f) for f in files]
    else:
        items = list(sources)
    chunks: list = []
    for name, src in items:
        if isinstance(src, (bytes, bytearray)):
            src = io.BytesIO(src)
        elif isinstance(src, Path):
            src = str(src)
        reader = PdfReader(src)
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if not text:
                continue
            for c in chunk_page(text, max_words, overlap_words):
                chunks.append(Chunk(len(chunks), name, i, c))
    if not chunks:
        raise ValueError("No extractable text found. Scanned PDFs need OCR first (e.g. ocrmypdf).")
    return chunks
