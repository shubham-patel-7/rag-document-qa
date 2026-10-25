"""Turn eval results into a markdown report and figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 10.5,
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 9.5, "legend.frameon": False, "figure.dpi": 150, "lines.linewidth": 2,
    })


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _wrap(label: str) -> str:
    return label.replace(" (", "\n(").replace(": ", ":\n")


def fig_retrievers(res):
    names = list(res["retrievers"])
    metrics = [("hit@1", "Hit@1", BLUE), ("hit@3", "Hit@3", ORANGE), ("hit@5", "Hit@5", AQUA)]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(names)), 3.8))
    for j, (k, lab, col) in enumerate(metrics):
        vals = [res["retrievers"][n]["overall"][k] for n in names]
        ax.bar(x + (j - 1) * w, vals, w * 0.92, color=col, label=lab, edgecolor=SURFACE, linewidth=1.2)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 1) * w, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(n) for n in names], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of questions")
    ax.set_title(f"Page-level retrieval hit rate ({res['n_questions']} questions)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.36))
    fig.tight_layout()
    _save(fig, "retrieval_comparison.png")


def fig_by_type(res):
    names = list(res["retrievers"])
    types = [t for t in ("factoid", "paraphrase") if t in res["retrievers"][names[0]]]
    fig, axes = plt.subplots(1, len(types), figsize=(5.2 * len(types), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, types):
        vals = [res["retrievers"][n][t]["mrr"] for n in names]
        n_q = res["retrievers"][names[0]][t]["n"]
        ax.barh(range(len(names)), vals, color=BLUE, height=0.55, edgecolor=SURFACE)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("MRR")
        ax.set_title(f"{t.capitalize()} questions (n={n_q})")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "by_question_type.png")


def fig_chunk_sweep(res):
    sweep = res["chunk_size_sweep"]
    colors = {"TF-IDF": BLUE, "MiniLM": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, key, title in [(axes[0], "hit@1", "Hit@1"), (axes[1], "mrr", "MRR")]:
        for name, d in sweep.items():
            sizes = sorted(d, key=int)
            ax.plot([int(s) for s in sizes], [d[s][key] for s in sizes], marker="o", ms=6, color=colors.get(name, INK2), label=f"Hybrid: {name} + BM25")
        ax.set_xlabel("Chunk size (words)")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "chunk_size_sweep.png")


def write_report(res) -> str:
    _style()
    fig_retrievers(res)
    fig_by_type(res)
    if res["chunk_size_sweep"]:
        fig_chunk_sweep(res)
    L = [f"## Retrieval evaluation ({res['n_questions']} questions, {res['n_docs']} documents, {res['n_chunks']} chunks)", "",
         "| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR | Context recall@5 | Hit@3 (factoid) | Hit@3 (paraphrase) |",
         "|---|---|---|---|---|---|---|---|"]
    for n, r in res["retrievers"].items():
        o = r["overall"]
        f = r.get("factoid", {}).get("hit@3", float("nan"))
        p = r.get("paraphrase", {}).get("hit@3", float("nan"))
        L.append(f"| {n} | {o['hit@1']:.2f} | {o['hit@3']:.2f} | {o['hit@5']:.2f} | {o['mrr']:.2f} | {o['context_recall@5']:.2f} | {f:.2f} | {p:.2f} |")
    if res["answers"]:
        L += ["", "### Answer check (hybrid retrieval, top-5)", "",
              "| Retriever | Answerer | Answer contains all gold keywords | Answer cites the gold page |", "|---|---|---|---|"]
        for n, a in res["answers"].items():
            ret, _, llm = n.partition(" | ")
            L.append(f"| {ret} | {llm or 'extractive'} | {a['keyword_accuracy']:.2f} | {a['cites_gold_page']:.2f} |")
    if res["chunk_size_sweep"]:
        L += ["", "### Chunk-size sweep (hybrid retrieval)", "", "| Embedder | Chunk words | #chunks | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|---|---|"]
        for emb, d in res["chunk_size_sweep"].items():
            for cs in sorted(d, key=int):
                r = d[cs]
                L.append(f"| {emb} | {cs} | {r['n_chunks']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['mrr']:.2f} |")
    if res["skipped"]:
        L += ["", "Skipped: " + "; ".join(f"{s['embedder']} ({s['reason']})" for s in res["skipped"])]
    md = "\n".join(L)
    (C.RESULTS_DIR / "eval_report.md").write_text(md + "\n")
    return md
