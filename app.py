
"""app.py

Streamlit web app — Streamlit Cloud ready.
Loads a default 'Port Operations Reference Manual.txt' at startup so the app is demo-ready.
Lightweight defaults for embeddings and HF generator.
"""

import os
import time
import json
from pathlib import Path

import streamlit as st

from rag_engine import RAGEngine

# --- Constants
CHROMA_DIR = "chroma_persist"
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DOC = "Port Operations Reference Manual.txt"

st.set_page_config(page_title="PortOps RAG Chatbot (Demo)", layout="wide", initial_sidebar_state="expanded")

# Header
st.markdown("""<div style='padding:12px;border-radius:8px;background:linear-gradient(90deg,#0b5cff,#00b7ff);color:white'>
<h2>PortOps RAG Chatbot — Demo (Ready to chat)</h2></div>""", unsafe_allow_html=True)
st.markdown("Upload new documents or use the built-in reference manual to try the chatbot immediately.")

# Sidebar controls
with st.sidebar:
    st.subheader("Configuration & Upload")
    model_choice = st.selectbox("LLM Backend", options=["Auto (OpenAI if available, else HF)", "OpenAI", "Hugging Face"], index=0)
    hf_model = st.text_input("Hugging Face model (if using HF)", value="google/flan-t5-small")
    persist_dir = st.text_input("Chroma persist directory", value=CHROMA_DIR)
    embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    max_tokens = st.slider("LLM max tokens (response)", min_value=64, max_value=512, value=128, step=32)
    top_k = st.slider("Retriever top_k (documents)", min_value=1, max_value=6, value=3)
    st.markdown("---")
    uploaded_files = st.file_uploader("Upload New Document", type=["pdf", "txt"], accept_multiple_files=True)
    build_index_btn = st.button("Add & Index Uploaded Documents")
    clear_index_btn = st.button("Clear Index (delete Chroma DB)")

# Initialize engine in session state
if "engine" not in st.session_state:
    try:
        st.session_state.engine = RAGEngine(persist_directory=CHROMA_DIR, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", hf_model_default="google/flan-t5-small")
        # Ensure default doc is loaded if index empty
        default_path = Path(DEFAULT_DOC)
        if default_path.exists():
            st.session_state.engine.add_default_if_empty(str(default_path))
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {e}")
        st.stop()

engine = st.session_state.engine

# Clear index
if clear_index_btn:
    try:
        engine.clear_persisted_index()
        st.success("Cleared persisted Chroma index.")
    except Exception as e:
        st.error(f"Error clearing index: {e}")

# Build index from uploaded files
if build_index_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file first.")
    else:
        progress = st.progress(0)
        total = len(uploaded_files)
        added_docs = []
        for i, uploaded in enumerate(uploaded_files):
            safe_name = uploaded.name.replace(" ", "_")
            dest = UPLOADS_DIR / safe_name
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            st.info(f"Saved {uploaded.name} → {dest}")
            try:
                doc_count = engine.add_documents([str(dest)])
                added_docs.append((uploaded.name, doc_count))
            except Exception as e:
                st.error(f"Failed to index {uploaded.name}: {e}")
            progress.progress(int(((i + 1) / total) * 100))
        st.success(f"Indexing finished. Added docs: {len(added_docs)}")
        for name, count in added_docs:
            st.write(f"- {name} → {count} chunks indexed")

# Main layout - chat and stats
left_col, right_col = st.columns([3, 1])

with left_col:
    st.subheader("Chat Interface")
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    st.markdown('<div style="background:#ffffff;padding:12px;border-radius:8px;height:60vh;overflow:auto">', unsafe_allow_html=True)
    for turn in st.session_state.history:
        st.markdown(f"**You:** {turn['query']}")
        st.markdown(f"**Bot:** {turn['answer']}")
        if turn.get('sources'):
            st.markdown("**Sources:**")
            for src in turn['sources']:
                src_line = f"- {src.get('source', 'unknown')}"
                if src.get('page'):
                    src_line += f" (page {src['page']})"
                st.markdown(f"<div style='background:#f1f5f9;padding:6px;border-radius:6px'>{src_line}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # User input
    query = st.text_input("Ask about the Port Operations Reference Manual or your uploaded docs", key="query_input")
    if st.button("Send"):
        if not query or query.strip() == "": 
            st.warning("Please type a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    result = engine.answer_query(query, top_k=top_k, max_tokens=max_tokens, hf_model=hf_model, force_openai=(model_choice=="OpenAI"), force_hf=(model_choice=="Hugging Face"))
                    answer = result.get('answer','')
                    sources = result.get('sources',[])
                    st.session_state.history.append({'query': query, 'answer': answer, 'sources': sources})
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error while answering: {e}")

with right_col:
    st.subheader("Statistics & Index")
    try:
        stats = engine.get_stats()
        st.metric("Documents indexed", stats.get('documents', 0))
        st.metric("Total vectors", stats.get('vectors', 0))
        st.write(f"- Persist directory: `{engine.persist_directory}`")
        st.write(f"- Embedding model: {engine.embedding_model_name}")
        st.write(f"- Index last update: {stats.get('last_index_time','N/A')}")
    except Exception as e:
        st.error(f"Failed to read stats: {e}")

    if st.button("Save conversation to file"):
        fname = f"conversation_{int(time.time())}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
        st.success(f"Saved conversation to {fname}")
\n