"""Streamlit web UI for the RAG chatbot — a clickable live demo.

Deploy free on Streamlit Community Cloud (share.streamlit.io): point it at this
repo, set ANTHROPIC_API_KEY in the app's Secrets, and it gives you a public URL.
"""

from __future__ import annotations

import os

import streamlit as st

from rag_chatbot import Retriever, chunk_text

st.set_page_config(page_title="RAG Chatbot — Chat with your docs", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.caption("Ask questions about a document. Built with Python + Claude (Retrieval-Augmented Generation).")

# API key from Streamlit secrets (deployed) or environment (local).
api_key = ""
try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception:
    pass
api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

# Document source: upload or use the bundled sample.
uploaded = st.file_uploader("Upload a .txt document (or use the sample below)", type=["txt"])
if uploaded is not None:
    text = uploaded.read().decode("utf-8", "ignore")
    st.success(f"Loaded your file: {uploaded.name}")
else:
    with open("sample_docs/about_rag.txt", encoding="utf-8") as fh:
        text = fh.read()
    st.info("Using the sample document (about RAG). Upload your own .txt above to try yours.")

chunks = chunk_text(text)
retriever = Retriever(chunks)

question = st.text_input("Ask a question about the document:")

if question:
    if not api_key:
        st.error("No ANTHROPIC_API_KEY configured. Add it in the app's Secrets (or your local .env).")
    else:
        context = "\n\n".join(retriever.top_k(question))
        if not context:
            st.warning("I couldn't find anything relevant in the document.")
        else:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            prompt = (
                "Answer the question using ONLY the context below. "
                "If the answer is not in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            with st.spinner("Thinking..."):
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
            answer_text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            ).strip()
            st.markdown("### Answer")
            st.write(answer_text)
            with st.expander("See the passages it retrieved"):
                st.write(context)
