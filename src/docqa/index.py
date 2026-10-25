"""Hybrid retrieval index: FAISS (dense) + BM25 (lexical), fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config as C
from .embedders import load_embedder, make_embedder
from .ingest import Chunk

_STOP = set("a an the of to in on for and or is are was were be been it its this that with as by at from "
            "what which who whom how why when where does do did can could should would about into than then".split())
_TOKEN = re.compile(r"[A-Za-z0-9_\-\.]+")


def tokenize(text: str) -> list:
    toks = [t.strip(".-").lower() for t in _TOKEN.findall(text)]
    return [t for t in toks if t and t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class HybridIndex:
    MODES = ("hybrid", "dense", "bm25")

    def __init__(self, chunks: list, embedder, embeddings: np.ndarray | None = None):
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.text for c in chunks]
        self.embeddings = embedder.encode(texts) if embeddings is None else embeddings
        self.faiss = faiss.IndexFlatIP(self.embeddings.shape[1])  # exact search; fine up to ~1M chunks
        self.faiss.add(self.embeddings)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, chunks: list, embedder_name: str = "tfidf") -> "HybridIndex":
        emb = make_embedder(embedder_name)
        emb.fit([c.text for c in chunks])
        return cls(chunks, emb)

    # ----------------------------------------------------------------- search
    def _dense(self, query: str, k: int) -> list:
        q = self.embedder.encode([query])
        scores, ids = self.faiss.search(q, min(k, len(self.chunks)))
        # Drop non-positive scores, like _bm25 does: a zero cosine means "no similarity at all", and the order
        # among tied zeros is arbitrary (it even changes with k), so reporting a rank for them is meaningless.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0 and s > 0]

    def _bm25(self, query: str, k: int) -> list:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(self, query: str, k: int = C.TOP_K, mode: str = "hybrid", pool: int = 20) -> list:
        """Return the top-k chunks. `pool` is how many candidates each retriever contributes before fusion."""
        if mode == "dense":
            ranked = self._dense(query, k)
        elif mode == "bm25":
            ranked = self._bm25(query, k)
        elif mode == "hybrid":
            fused: dict = {}
            for lst in (self._dense(query, pool), self._bm25(query, pool)):
                for rank, (i, _) in enumerate(lst, start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (C.RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda t: -t[1])[:k]
        else:
            raise ValueError(f"mode must be one of {self.MODES}")
        return [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(ranked, start=1)]

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(directory / "vectors.faiss"))
        (directory / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]))
        (directory / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "n_chunks": len(self.chunks)}))
        self.embedder.save(directory)

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        chunks = [Chunk(**d) for d in json.loads((directory / "chunks.json").read_text())]
        emb = load_embedder(meta["embedder"], directory)
        vectors = faiss.read_index(str(directory / "vectors.faiss"))
        obj = cls.__new__(cls)
        obj.chunks, obj.embedder, obj.faiss = chunks, emb, vectors
        obj.embeddings = vectors.reconstruct_n(0, vectors.ntotal)
        obj.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        return obj

    @property
    def n_docs(self) -> int:
        return len({c.doc for c in self.chunks})
