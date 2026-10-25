"""Question -> retrieve -> generate -> cited answer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config as C
from .index import HybridIndex
from .ingest import load_chunks
from .llm import cited_numbers, get_llm


@dataclass
class Source:
    n: int            # the [n] used in the answer text
    doc: str
    page: int
    text: str
    score: float
    cited: bool       # did the answer actually cite this passage?


@dataclass
class Answer:
    question: str
    text: str
    sources: list
    mode: str
    llm: str


def _bm25_idf(index: HybridIndex) -> dict:
    """Token -> IDF weight, reused by the offline extractive answerer (floored so no weight is <= 0)."""
    return {t: max(float(v), 0.1) for t, v in index.bm25.idf.items()}


class DocQA:
    def __init__(self, index: HybridIndex, llm: str = "auto", mode: str = "hybrid", k: int = C.TOP_K):
        self.index, self.mode, self.k = index, mode, k
        self.llm = get_llm(llm, idf=_bm25_idf(index), embedder=index.embedder)

    @classmethod
    def from_pdfs(cls, sources, embedder: str = "tfidf", chunk_words: int = C.CHUNK_WORDS,
                  overlap_words: int = C.OVERLAP_WORDS, **kw) -> "DocQA":
        chunks = load_chunks(sources, chunk_words, overlap_words)
        return cls(HybridIndex.build(chunks, embedder), **kw)

    @classmethod
    def load(cls, index_dir: Path, **kw) -> "DocQA":
        return cls(HybridIndex.load(index_dir), **kw)

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None) -> list:
        return self.index.search(question, k or self.k, mode or self.mode)

    def ask(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        hits = self.retrieve(question, k, mode)
        text = self.llm.generate(question, hits) if hits else "I could not find that in the documents."
        cited = set(cited_numbers(text, len(hits)))
        sources = [Source(i, h.chunk.doc, h.chunk.page, h.chunk.text, h.score, i in cited)
                   for i, h in enumerate(hits, start=1)]
        return Answer(question, text, sources, mode or self.mode, self.llm.name)
