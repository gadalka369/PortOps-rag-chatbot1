"""app.py - SIMPLIFIED VERSION

Minimal code, maximum reliability.
"""

import os
import streamlit as st
from pathlib import Path
from rag_engine import RAGEngine

st.set_page_config(page_title="Port Operations Chatbot", layout="wide", page_icon="🚢")

# Initialize
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_rag():
    """Initialize RAG engine."""
    try:
        # Get token
        token = None
        if hasattr(st, 'secrets'):
            token = st.secrets.get('HUGGINGFACEHUB_API_TOKEN') or st.secrets.get('HUGGINGFACE_API_TOKEN')
        if not token:
            token = os.environ.get('HUGGINGFACEHUB_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')
        
        if not token:
            st.error("❌ No HuggingFace token found!")
            st.info("Add HUGGINGFACEHUB_API_TOKEN to Streamlit secrets")
            return None
        
        # Create engine
        engine = RAGEngine(hf_token=token)
        
        # Load default doc
        default_doc = "Port Operations Reference Manual.txt"
        if Path(default_doc).exists():
            stats = engine.get_stats()
            if stats['documents'] == 0:
                with st.spinner("Loading default document..."):
                    engine.index_documents([default_doc])
        
        return engine
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Initialize engine
if st.session_state.rag_engine is None:
    st.session_state.rag_engine = init_rag()

engine = st.session_state.rag_engine

# UI
st.title("🚢 Port Operations AI Chatbot")
st.markdown("Ask questions about port operations")

# Sidebar
with st.sidebar:
    st.header("📊 Stats")
    if engine:
        stats = engine.get_stats()
        st.metric("Documents", stats['documents'])
        st.metric("Vectors", stats['vectors'])
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Check if ready
if not engine:
    st.warning("⚠️ RAG engine not ready. Check configuration.")
    st.stop()

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

# Chat input
if prompt := st.chat_input("Ask about port operations..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = engine.ask(prompt)
                st.write(answer)
                
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.write(f"- {s}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                error = f"Error: {e}"
                st.error(error)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error,
                    "sources": []
                })

# Example questions
st.markdown("---")
st.subheader("💡 Example Questions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Crane Safety"):
        st.session_state.messages.append({"role": "user", "content": "What are crane safety procedures?"})
        st.rerun()

with col2:
    if st.button("Emergency Contacts"):
        st.session_state.messages.append({"role": "user", "content": "List emergency contacts"})
        st.rerun()

with col3:
    if st.button("Weather Limits"):
        st.session_state.messages.append({"role": "user", "content": "What are weather restrictions?"})
        st.rerun()
