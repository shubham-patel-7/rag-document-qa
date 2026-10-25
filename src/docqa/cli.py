"""Command line interface:  docqa ingest | ask | eval"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as C


def cmd_ingest(a):
    from .index import HybridIndex
    from .ingest import load_chunks

    chunks = load_chunks(a.docs, a.chunk_words, a.overlap_words)
    idx = HybridIndex.build(chunks, a.embedder)
    idx.save(a.index)
    print(f"Indexed {len(chunks)} chunks from {idx.n_docs} document(s) with the '{a.embedder}' embedder -> {a.index}")


def cmd_ask(a):
    from .pipeline import DocQA

    qa = DocQA.load(a.index, llm=a.llm, mode=a.mode, k=a.k)
    ans = qa.ask(a.question)
    print(f"\n{ans.text}\n\n[{ans.llm} answerer, {ans.mode} retrieval]")
    for s in ans.sources:
        mark = "*" if s.cited else " "
        print(f" {mark}[{s.n}] {s.doc}, page {s.page}  (score {s.score:.3f})")
    print(" (* = cited in the answer)")


def cmd_eval(a):
    from .evaluate import load_questions, run_eval
    from .report import write_report

    res = run_eval(a.docs, load_questions(a.questions), a.embedders, answer_llms=a.answer_llm)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "eval_results.json").write_text(json.dumps(res, indent=2))
    md = write_report(res)
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docqa", description="Document Q&A with cited answers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="build and save an index from PDFs")
    p.add_argument("--docs", type=Path, default=C.SAMPLE_DOCS, help="PDF file or directory of PDFs")
    p.add_argument("--index", type=Path, default=C.INDEX_DIR / "sample")
    p.add_argument("--embedder", choices=["tfidf", "minilm"], default="tfidf")
    p.add_argument("--chunk-words", type=int, default=C.CHUNK_WORDS)
    p.add_argument("--overlap-words", type=int, default=C.OVERLAP_WORDS)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question against a saved index")
    p.add_argument("question")
    p.add_argument("--index", type=Path, default=C.INDEX_DIR / "sample")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("--llm", choices=["auto", "extractive", "anthropic", "openai"], default="auto")
    p.add_argument("-k", type=int, default=C.TOP_K)
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("eval", help="measure retrieval quality on the labelled question set")
    p.add_argument("--docs", type=Path, default=C.SAMPLE_DOCS)
    p.add_argument("--questions", type=Path, default=C.EVAL_QUESTIONS)
    p.add_argument("--embedders", nargs="+", choices=["tfidf", "minilm"], default=["tfidf"])
    p.add_argument("--answer-llm", nargs="+", choices=["extractive", "anthropic", "openai"], default=["extractive"],
                   help="answer backends to score (anthropic/openai need an API key)")
    p.set_defaults(fn=cmd_eval)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
