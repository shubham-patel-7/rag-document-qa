"""Text embedders. All return L2-normalised float32 arrays, so inner product == cosine similarity.

* `minilm`  - sentence-transformers all-MiniLM-L6-v2 (neural, 384-d). Default when installed.
* `tfidf`   - dense TF-IDF (word + bigram). No download, no GPU: baseline and offline fallback.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from . import config as C


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(n, 1e-12, None)).astype(np.float32)


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english",
                                   max_features=30000, strip_accents="unicode")
        self.fitted = False

    def fit(self, texts) -> "TfidfEmbedder":
        self.vec.fit(texts)
        self.fitted = True
        return self

    def encode(self, texts) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TfidfEmbedder must be fit on the corpus first")
        return _normalize(self.vec.transform(list(texts)).toarray())

    def save(self, directory: Path) -> None:
        joblib.dump(self.vec, Path(directory) / "tfidf.joblib")

    @classmethod
    def load(cls, directory: Path) -> "TfidfEmbedder":
        e = cls()
        e.vec = joblib.load(Path(directory) / "tfidf.joblib")
        e.fitted = True
        return e


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = C.MINILM_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install sentence-transformers to use the MiniLM embedder") from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts) -> "MiniLMEmbedder":  # nothing to fit; kept for a uniform interface
        return self

    def encode(self, texts) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def save(self, directory: Path) -> None:
        (Path(directory) / "minilm.txt").write_text(self.model_name)

    @classmethod
    def load(cls, directory: Path) -> "MiniLMEmbedder":
        return cls((Path(directory) / "minilm.txt").read_text().strip())


EMBEDDERS = {"tfidf": TfidfEmbedder, "minilm": MiniLMEmbedder}


def make_embedder(name: str):
    if name not in EMBEDDERS:
        raise ValueError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}")
    return EMBEDDERS[name]()


def load_embedder(name: str, directory: Path):
    return EMBEDDERS[name].load(directory)


def minilm_available() -> bool:
    """True if sentence-transformers is installed AND the model can actually be loaded."""
    try:
        MiniLMEmbedder()
        return True
    except Exception:
        return False
