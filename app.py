"""Streamlit UI. Run: streamlit run app.py"""
import streamlit as st

import settings
from answer import answer_question
from llm import LLMClient
from retrieval import build_retriever
from sources import load_sources

st.set_page_config(page_title="ScholarLoan RAG", page_icon=":mortar_board:", layout="centered")


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
        st.caption(f"similarity {c.score:.2f}")


st.title("ScholarLoan RAG")
st.caption("Cited answers about Indian education loans and scholarships, taken only from the public documents listed in the sidebar.")

try:
    retriever, cfg, sources, llm = load_system()
except FileNotFoundError:
    st.error("No index found. Run `python ingest.py` and then `python evaluate.py --apply-best`.")
    st.stop()

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
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    result = answer_question(question.strip(), retriever, method=method, tau=tau, k=k, llm=llm, sources=sources)
    if result.answered:
        st.subheader("Answer")
        st.markdown(result.text)
        if result.note:
            st.info(result.note)
        st.subheader("Sources")
        for c in result.citations:
            show_citation(c)
    else:
        st.warning(result.text)
        if result.note:
            st.caption(result.note)
        if result.closest:
            with st.expander("Closest passages (low confidence)"):
                for c in result.closest:
                    show_citation(c)
    st.caption("Scheme rules change. Always confirm on the official source before applying. This is not financial or legal advice.")
elif submitted:
    st.info("Type a question first.")
