"""Chat with your PDFs - answers with page-level citations.

Run:  streamlit run app/streamlit_app.py
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from docqa import config as C
from docqa.embedders import minilm_available
from docqa.pipeline import DocQA

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Checking for the MiniLM embedding model…")
def has_minilm() -> bool:
    return minilm_available()


@st.cache_resource(show_spinner="Reading PDFs and building the index…")
def build_qa(key: str, files: tuple, embedder: str, chunk_words: int, llm: str, mode: str, k: int):
    """`files` is a tuple of (name, bytes) or a Path to a folder; `key` makes the cache key explicit."""
    sources = [(n, b) for n, b in files] if files else C.SAMPLE_DOCS
    return DocQA.from_pdfs(sources, embedder=embedder, chunk_words=chunk_words,
                           overlap_words=max(10, chunk_words // 4), llm=llm, mode=mode, k=k)


def main():
    st.title("📚 Ask your documents")
    st.caption("Hybrid retrieval (FAISS vectors + BM25) → answer with page-level citations.")

    st.sidebar.header("Documents")
    uploads = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    use_sample = st.sidebar.checkbox("Use the bundled sample corpus (6 short PDFs)", value=not uploads)

    st.sidebar.header("Retrieval")
    minilm_ok = has_minilm()
    embedder = st.sidebar.selectbox(
        "Embedder", ["minilm", "tfidf"] if minilm_ok else ["tfidf"],
        help="MiniLM = neural sentence embeddings (understands paraphrases). TF-IDF = word statistics, no model download.")
    if not minilm_ok:
        st.sidebar.info("MiniLM isn't available here (not installed or no internet to download it), so TF-IDF is used.")
    mode = st.sidebar.radio("Mode", ["hybrid", "dense", "bm25"], help="hybrid fuses the dense and BM25 rankings with Reciprocal Rank Fusion.")
    k = st.sidebar.slider("Passages to retrieve (k)", 1, 10, 5)
    chunk_words = st.sidebar.select_slider("Chunk size (words)", [60, 90, 120, 160, 200], value=120)

    st.sidebar.header("Answering")
    llm = st.sidebar.selectbox("Backend", ["auto", "extractive", "anthropic", "openai"],
                               help="auto = Anthropic or OpenAI-compatible if an API key is set, else the offline extractive answerer.")

    files = tuple((u.name, u.getvalue()) for u in uploads) if uploads else ()
    if not files and not use_sample:
        st.info("Upload one or more PDFs, or tick the sample corpus in the sidebar.")
        st.stop()
    key = hashlib.md5(b"".join(b for _, b in files)).hexdigest() if files else "sample"
    try:
        qa = build_qa(key, files, embedder, chunk_words, llm, mode, k)
    except Exception as e:
        st.error(f"Could not build the index: {e}")
        st.stop()
    st.sidebar.success(f"{len(qa.index.chunks)} chunks from {qa.index.n_docs} document(s)")

    tab_ask, tab_eval = st.tabs(["Ask", "Evaluation"])
    with tab_ask:
        examples = ["How is coolant temperature encoded in J1939?", "What makes a faulty CAN node stop disrupting the network?",
                    "Why is predicting failure too late worse than too early?", "What is the capital of France?"]
        if "q" not in st.session_state:
            st.session_state.q = ""
        cols = st.columns(len(examples))
        for c, ex in zip(cols, examples):
            if c.button(ex, use_container_width=True):
                st.session_state.q = ex
        question = st.text_input("Your question", key="q", placeholder="Ask something about the documents…")
        if question:
            ans = qa.ask(question)
            st.markdown("### Answer")
            st.markdown(re.sub(r"\[(\d+)\]", r"**[\1]**", ans.text))
            st.caption(f"{ans.llm} answerer · {ans.mode} retrieval · k={len(ans.sources)}")
            st.markdown("### Sources")
            for s in ans.sources:
                badge = "✅ cited" if s.cited else "retrieved"
                with st.expander(f"[{s.n}] {s.doc} · page {s.page} · {badge} · score {s.score:.3f}", expanded=s.cited):
                    st.write(s.text)

    with tab_eval:
        rep = C.RESULTS_DIR / "eval_report.md"
        if rep.exists():
            st.markdown(rep.read_text())
            for fig in ("retrieval_comparison.png", "by_question_type.png", "chunk_size_sweep.png"):
                if (C.FIG_DIR / fig).exists():
                    st.image(str(C.FIG_DIR / fig))
        else:
            st.info("Run `docqa eval` to generate the evaluation report.")


main()
