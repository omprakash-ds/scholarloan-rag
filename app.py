"""Streamlit UI. Run: streamlit run app.py"""
import os
import time
from pathlib import Path
import streamlit as st
import settings
from answer import answer_question
from llm import LLMClient
from retrieval import build_retriever
from sources import load_sources
def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load key=value pairs from a local .env file without extra dependencies."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_dotenv()
st.set_page_config(page_title="ScholarLoan RAG", page_icon=":mortar_board:", layout="centered")
SAMPLE_QUESTIONS = [
    ("Pragati Amount", "How much scholarship does the AICTE Pragati scheme give per year?"),
    ("IBA Loan Margin (OCR)", "What is the minimum margin required for an education loan up to Rs. 4 lakh under the IBA scheme?"),
    ("PM-Vidyalaxmi", "Does PM-Vidyalaxmi require collateral or a guarantor for education loans at QHEIs?"),
    ("Refusal Test", "What processing fee does HDFC Credila charge for US education loans?"),
]
@st.cache_resource(show_spinner="Loading documents and models...")
def load_system():
    cfg = settings.load_config()
    retriever = build_retriever(settings.CHUNKS_PATH, cfg.get("embedder", settings.EMBED_MODEL), settings.DATA_DIR)
    return retriever, cfg, load_sources(), LLMClient.from_env()
def show_citation(c):
    label = f"[{c.n}] {c.title or c.doc} - page {c.page}"
    with st.expander(label):
        meta = " | ".join(x for x in (c.publisher, f"as of {c.as_of}" if c.as_of else "") if x)
        if meta:
            st.caption(meta)
        if c.url:
            st.markdown(f"[Open source document]({c.url})")
        st.write(c.text)
        st.caption(f"similarity {c.score:.3f}")
st.title("ScholarLoan RAG")
st.caption("Cited answers about Indian education loans and scholarships, taken only from the public documents listed in the sidebar.")
try:
    retriever, cfg, sources, llm = load_system()
except FileNotFoundError:
    st.error("No index found. Run `python ingest.py` and then `python evaluate.py --apply-best`.")
    st.stop()
doc_count = len({c["doc"] for c in retriever.chunks})
c1, c2, c3 = st.columns(3)
c1.metric("Official PDFs", doc_count)
c2.metric("Indexed Passages", len(retriever.chunks))
c3.metric("Answer Mode", "LLM + Citations" if llm else "Extractive")
with st.sidebar:
    st.header("Documents")
    for doc in sorted({c["doc"] for c in retriever.chunks}):
        meta = sources.get(doc, {})
        when = f" ({meta['as_of']})" if meta.get("as_of") else ""
        st.markdown(f"**{meta.get('title') or doc}**{when}  \n{meta.get('publisher', '')}")
    st.divider()
    methods = retriever.available_methods()
    default = cfg.get("method") if cfg.get("method") in methods else methods[0]
    method = st.selectbox("Retrieval method", methods, index=methods.index(default))
    k = st.slider("Passages to retrieve", 3, 8, settings.TOP_K)
    tau = cfg.get("tau")
    if tau is None:
        st.warning("Refusal threshold not tuned yet. Run `python evaluate.py --apply-best`.")
        tau = 0.0
    else:
        st.caption(f"Refusal threshold {tau} (tuned for '{cfg.get('method')}').")
    st.caption("Answer mode: " + ("language model with citations" if llm else "extracted passages (no API key set)"))
with st.form("ask"):
    question = st.text_input("Ask about a loan or scholarship", placeholder="e.g. Is interest subsidy available for both UG and PG?")
    submitted = st.form_submit_button("Ask", type="primary")
st.caption("Try a sample question:")
sample_clicked = None
cols = st.columns(len(SAMPLE_QUESTIONS))
for col, (label, q_text) in zip(cols, SAMPLE_QUESTIONS):
    if col.button(label, use_container_width=True):
        sample_clicked = q_text
active_question = sample_clicked if sample_clicked else (question.strip() if submitted else "")
if active_question:
    if sample_clicked:
        st.markdown(f"**Question:** {active_question}")
    t0 = time.perf_counter()
    result = answer_question(active_question, retriever, method=method, tau=tau, k=k, llm=llm, sources=sources)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    top_hit = result.citations[0] if result.citations else (result.closest[0] if result.closest else None)
    top_score_str = f"{top_hit.score:.3f}" if top_hit else "N/A"
    mode_label = "LLM + Citations" if (llm and not result.note) else "Extractive"
    if result.answered:
        st.subheader("Answer")
        with st.container(border=True):
            st.markdown(result.text)
        st.caption(f"Latency: {elapsed_ms:.0f} ms | Top confidence: {top_score_str} (threshold: {tau:.3f}) | Method: {method} | Mode: {mode_label}")
        if result.note:
            st.info(result.note)
        st.subheader("Sources")
        for c in result.citations:
            show_citation(c)
    else:
        st.warning(result.text)
        st.caption(f"Latency: {elapsed_ms:.0f} ms | Top confidence: {top_score_str} (below threshold: {tau:.3f}) | Method: {method}")
        if result.note:
            st.caption(result.note)
        if result.closest:
            with st.expander("Closest passages (low confidence)"):
                for c in result.closest:
                    show_citation(c)
    st.caption("Scheme rules change. Always confirm on the official source before applying. This is not financial or legal advice.")
elif submitted:
    st.info("Type a question first.")
