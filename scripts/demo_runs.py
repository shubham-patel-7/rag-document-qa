"""Three demo runs of the question-answering pipeline on the sample corpus.

  1. a factoid question        (shares words with the answer)
  2. a paraphrased question    (almost no shared words with the answer)
  3. a question the corpus cannot answer (the assistant should abstain, not make something up)

Each question is run through BM25-only, dense-only and hybrid retrieval, then answered by the offline
extractive answerer (no API key needed).

    python scripts/demo_runs.py [--embedders tfidf minilm]     # writes results/demo_runs.json / .md
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docqa import config as C  # noqa: E402
from docqa.index import HybridIndex  # noqa: E402
from docqa.ingest import load_chunks  # noqa: E402
from docqa.llm import NOT_FOUND  # noqa: E402
from docqa.pipeline import DocQA  # noqa: E402

DEMOS = [
    ("1. Factoid", "Who developed the CAN protocol and in what year was it introduced?",
     {"doc": "can_bus_fundamentals.pdf", "page": 1}),
    ("2. Paraphrase", "How does CAN decide which of two simultaneous senders goes first?", None),   # gold looked up below
    ("3. Not in the documents", "What is the price of a Cummins X15 engine?", None),
]


def gold_for(qid_text):
    qs = json.loads((C.ROOT / "data" / "eval" / "questions.json").read_text())
    for q in qs:
        if q["question"] == qid_text:
            return q["gold"][0]
    return None


def rank_of_gold(hits, gold):
    for h in hits:
        if h.chunk.doc == gold["doc"] and h.chunk.page == gold["page"]:
            return h.rank
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", default=["tfidf"])
    args = ap.parse_args(argv)
    chunks = load_chunks(C.SAMPLE_DOCS)
    out = []
    for emb in args.embedders:
        try:
            idx = HybridIndex.build(chunks, emb)
        except Exception as e:   # e.g. MiniLM weights not downloadable offline
            print(f"skipping {emb}: {e}")
            continue
        qa = DocQA(idx, llm="extractive", mode="hybrid")
        for title, question, gold in DEMOS:
            gold = gold or gold_for(question)
            ranks = {m: (rank_of_gold(idx.search(question, 10, m), gold) if gold else None) for m in ("bm25", "dense", "hybrid")}
            t0 = time.perf_counter()
            ans = qa.ask(question)
            ms = (time.perf_counter() - t0) * 1000
            cited = [s for s in ans.sources if s.cited]
            out.append({
                "demo": title, "embedder": emb, "question": question, "gold": gold, "gold_rank": ranks,
                "answer": ans.text, "abstained": ans.text == NOT_FOUND,
                "top_source": f"{ans.sources[0].doc} p.{ans.sources[0].page}" if ans.sources else None,
                "top_source_score": ans.sources[0].score if ans.sources else None,
                "cited": [f"{s.doc} p.{s.page}" for s in cited],
                "cites_gold_page": bool(gold and any(s.doc == gold["doc"] and s.page == gold["page"] for s in cited)),
                "ms": ms,
            })
    (C.RESULTS_DIR / "demo_runs.json").write_text(json.dumps(out, indent=2))
    fmt = lambda r: "not in top 10" if r is None else f"rank {r}"  # noqa: E731
    L = ["| Demo run | Embedder | Gold page rank (BM25 / dense / hybrid) | Answer | Cites gold page | Latency |", "|---|---|---|---|---|---|"]
    for r in out:
        gr = r["gold_rank"]
        ranks = "n/a (no gold page)" if r["gold"] is None else f"{fmt(gr['bm25'])} / {fmt(gr['dense'])} / {fmt(gr['hybrid'])}"
        cites = "n/a" if r["gold"] is None else ("yes" if r["cites_gold_page"] else "no")
        L.append(f"| {r['demo']} | {r['embedder']} | {ranks} | {r['answer'][:110].strip()} | {cites} | {r['ms']:.0f} ms |")
    md = "\n".join(L)
    (C.RESULTS_DIR / "demo_runs.md").write_text(md + "\n")
    print(md)
    for r in out:
        print(f"\n[{r['demo']} / {r['embedder']}] {r['question']}\n  -> {r['answer']}\n  cited={r['cited']} top={r['top_source']} score={r['top_source_score']:.3f}")


if __name__ == "__main__":
    main()
