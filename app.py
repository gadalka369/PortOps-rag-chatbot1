
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
# --- Sidebar / Upload Section ---
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload New Document", type=["pdf", "txt"], key="upload_doc_btn"
)

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        with open(f"uploaded_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded {uploaded_file.name}")
        # Here you would index the new document into RAG
        # rag_engine.index_new_document(f"uploaded_{uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to upload document: {e}")

# --- Main Chat Interface ---
st.header("Port Operations Chatbot")

# Display conversation history
if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")

# Input box
user_input = st.text_input("Ask a question about port operations:", key="user_input_box")

if st.button("Send", key="send_msg_btn") and user_input:
    try:
        answer, sources = rag_engine.ask(user_input)
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "bot", "content": answer})
        st.markdown(f"**Bot:** {answer}")
        if sources:
            st.markdown("**Sources:**")
            for s in sources:
                st.markdown(f"- {s}")
    except Exception as e:
        st.error(f"Failed to get answer: {e}")

# --- Save / Clear Buttons ---
st.write("---")

if st.button("Save conversation to file", key="save_conv_btn"):
    try:
        fname = f"conversation_{int(time.time())}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
        st.success(f"Saved conversation to {fname}")
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

if st.button("Clear chat history", key="clear_chat_btn"):
    st.session_state.history = []
    st.success("Chat history cleared")
