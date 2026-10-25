"""Paths, defaults and a tiny .env loader (no extra dependency)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCS = ROOT / "data" / "sample_docs"
EVAL_QUESTIONS = ROOT / "data" / "eval" / "questions.json"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

CHUNK_WORDS = 120       # target chunk size (words)
OVERLAP_WORDS = 30      # words repeated between neighbouring chunks
TOP_K = 5
RRF_K = 60              # reciprocal-rank-fusion constant
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


load_dotenv()
