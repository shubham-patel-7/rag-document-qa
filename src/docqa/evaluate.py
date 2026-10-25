"""Retrieval + answer evaluation against a hand-labelled question set.

Each question has a gold (document, page) and a few keywords that must appear in a correct answer.

Metrics
  Hit@k        a gold page appears among the top-k retrieved chunks
  MRR          mean reciprocal rank of the first gold-page chunk
  Context recall@k   all answer keywords occur somewhere in the text of the top-k chunks
                     (an answerability check that does not require the exact page)
  Answer keyword accuracy   generated answer contains all keywords (works with any LLM backend)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from .index import HybridIndex
from .ingest import load_chunks
from .pipeline import DocQA

KS = (1, 3, 5)


def load_questions(path: Path = C.EVAL_QUESTIONS) -> list:
    return json.loads(Path(path).read_text())


def _first_gold_rank(hits, gold) -> int | None:
    gold_set = {(g["doc"], g["page"]) for g in gold}
    for h in hits:
        if (h.chunk.doc, h.chunk.page) in gold_set:
            return h.rank
    return None


def _has_all(text: str, kws) -> bool:
    t = text.lower()
    return all(k.lower() in t for k in kws)


def evaluate_retrieval(index: HybridIndex, questions: list, mode: str) -> dict:
    kmax = max(KS)
    rows = []
    for q in questions:
        hits = index.search(q["question"], kmax, mode)
        rank = _first_gold_rank(hits, q["gold"])
        rows.append({
            "id": q["id"], "type": q["type"], "rank": rank,
            **{f"ctx@{k}": _has_all(" ".join(h.chunk.text for h in hits[:k]), q["answer_keywords"]) for k in KS},
        })

    def agg(sub):
        out = {f"hit@{k}": float(np.mean([r["rank"] is not None and r["rank"] <= k for r in sub])) for k in KS}
        out["mrr"] = float(np.mean([1.0 / r["rank"] if r["rank"] else 0.0 for r in sub]))
        out.update({f"context_recall@{k}": float(np.mean([r[f"ctx@{k}"] for r in sub])) for k in KS})
        out["n"] = len(sub)
        return out

    res = {"overall": agg(rows)}
    for t in sorted({r["type"] for r in rows}):
        res[t] = agg([r for r in rows if r["type"] == t])
    res["per_question"] = rows
    return res


def evaluate_answers(index: HybridIndex, questions: list, mode: str = "hybrid", llm: str = "extractive", k: int = 5) -> dict:
    qa = DocQA(index, llm=llm, mode=mode, k=k)
    ok, rows = [], []
    for q in questions:
        a = qa.ask(q["question"])
        correct = _has_all(a.text, q["answer_keywords"])
        cited_gold = any((s.doc, s.page) in {(g["doc"], g["page"]) for g in q["gold"]} for s in a.sources if s.cited)
        ok.append(correct)
        rows.append({"id": q["id"], "correct": correct, "cites_gold_page": cited_gold, "answer": a.text})
    return {"keyword_accuracy": float(np.mean(ok)),
            "cites_gold_page": float(np.mean([r["cites_gold_page"] for r in rows])), "rows": rows}


def run_eval(docs, questions: list, embedders=("tfidf",), chunk_sizes=(60, 120, 200), answer_llms=("extractive",)) -> dict:
    """Run every experiment and return a JSON-serialisable result dict."""
    out = {"n_questions": len(questions), "retrievers": {}, "chunk_size_sweep": {}, "answers": {}, "skipped": []}
    base_chunks = load_chunks(docs, C.CHUNK_WORDS, C.OVERLAP_WORDS)
    out["n_chunks"] = len(base_chunks)
    out["n_docs"] = len({c.doc for c in base_chunks})
    indexes = {}
    for emb in embedders:
        try:
            indexes[emb] = HybridIndex.build(base_chunks, emb)
        except Exception as e:  # e.g. MiniLM weights cannot be downloaded
            out["skipped"].append({"embedder": emb, "reason": f"{type(e).__name__}: {str(e)[:120]}"})
    first = next(iter(indexes.values()))
    out["retrievers"]["BM25 only"] = evaluate_retrieval(first, questions, "bm25")
    labels = {"tfidf": "TF-IDF", "minilm": "MiniLM"}
    for emb, idx in indexes.items():
        out["retrievers"][f"{labels[emb]} dense (FAISS)"] = evaluate_retrieval(idx, questions, "dense")
        out["retrievers"][f"Hybrid: {labels[emb]} + BM25 (RRF)"] = evaluate_retrieval(idx, questions, "hybrid")
        for llm in answer_llms:
            tag = f"Hybrid: {labels[emb]} + BM25 (RRF) | {llm}"
            out["answers"][tag] = evaluate_answers(idx, questions, "hybrid", llm)
    for emb, idx in indexes.items():
        for cs in chunk_sizes:
            ch = load_chunks(docs, cs, max(10, cs // 4))
            sweep_idx = HybridIndex.build(ch, emb)
            r = evaluate_retrieval(sweep_idx, questions, "hybrid")["overall"]
            out["chunk_size_sweep"].setdefault(labels[emb], {})[str(cs)] = {**r, "n_chunks": len(ch)}
    return out
