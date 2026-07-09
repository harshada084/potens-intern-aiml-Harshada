"""
app_streamlit.py

Simple UI to try /ask and /contradict without needing Postman or curl.
Calls the qa.py and contradict.py functions directly (no need for the
FastAPI server to be running).

Run with:  streamlit run app_streamlit.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
from translate import ask_multilingual
from contradict import contradict, list_available_documents

st.set_page_config(page_title="Document Q&A with Citations", layout="wide")

st.title("📄 Document Q&A with Citations")
st.caption("RAG system over fraud-detection research papers — ask questions (any language) or check for contradictions between papers.")

tab1, tab2 = st.tabs(["Ask a Question", "Check for Contradictions"])

with tab1:
    st.subheader("Ask a question about the documents")
    st.caption("Works in any language — the question is translated automatically if needed.")
    question = st.text_input("Your question:", placeholder="e.g. What is SMOTE used for?")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                result = ask_multilingual(question)

            if result.get("detected_language", "en") != "en":
                st.caption(f"Detected language: {result['detected_language']} — translated for retrieval, answer translated back.")

            if result["no_answer_found"]:
                st.info("🚫 " + result["answer"])
            else:
                st.success("**Answer:**")
                st.write(result["answer"])

                st.markdown("---")
                st.markdown("**Citations:**")
                for c in result["citations"]:
                    with st.container(border=True):
                        st.markdown(f"**Source:** {c['source']} &nbsp;&nbsp; **Ref:** {c['chunk_ref']}")
                        st.caption(c["snippet"] + "...")

with tab2:
    st.subheader("Compare two documents for contradictions")

    docs = list_available_documents()

    if not docs:
        st.warning("No documents found. Run `python src/ingest.py` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            doc_a = st.selectbox("Document A", docs, key="doc_a")
        with col2:
            doc_b = st.selectbox("Document B", docs, index=min(1, len(docs) - 1), key="doc_b")

        if st.button("Check for Contradiction", type="primary"):
            if doc_a == doc_b:
                st.warning("Please select two different documents.")
            else:
                with st.spinner("Analyzing both documents..."):
                    result = contradict(doc_a, doc_b)

                if "error" in result:
                    st.error(result["error"])
                else:
                    if result["conflict"]:
                        st.error(f"⚠️ Conflict detected — Topic: {result['topic']}")
                    else:
                        st.success("✅ No conflict detected")

                    st.markdown("**Reasoning:**")
                    st.write(result["reasoning"])

st.markdown("---")
st.caption("Built for the Potens AI/ML internship assignment — Document Q&A with Citations (Q1)")