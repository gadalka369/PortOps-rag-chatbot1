"""app.py - Fixed Version

Streamlit web app – Streamlit Cloud ready.
Loads a default 'Port Operations Reference Manual.txt' at startup so the app is demo-ready.
"""

import os
import time
import json
from pathlib import Path

import streamlit as st
from rag_engine import RAGEngine

# --- Constants ---
CHROMA_DIR = "chroma_persist"
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DOC = "Port Operations Reference Manual.txt"

# --- Page Config ---
st.set_page_config(
    page_title="PortOps RAG Chatbot", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🚢"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #0b5cff, #00b7ff);
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "history" not in st.session_state:
    st.session_state.history = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# --- Initialize RAG Engine ---
@st.cache_resource
def initialize_rag_engine():
    """Initialize RAG engine and load default document"""
    try:
        # Get Hugging Face token from secrets or environment
        hf_token = None
        if hasattr(st, 'secrets') and 'HUGGINGFACE_API_TOKEN' in st.secrets:
            hf_token = st.secrets['HUGGINGFACE_API_TOKEN']
        elif 'HUGGINGFACE_API_TOKEN' in os.environ:
            hf_token = os.environ['HUGGINGFACE_API_TOKEN']
        
        # Initialize engine
        engine = RAGEngine(
            persist_directory=CHROMA_DIR,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            hf_model="google/flan-t5-small",
            hf_token=hf_token,
            use_openai=False
        )
        
        # Check if default document needs to be loaded
        if Path(DEFAULT_DOC).exists():
            # Check if vectorstore is empty
            stats = engine.get_stats()
            if stats.get('documents', 0) == 0:
                with st.spinner(f"Loading default document: {DEFAULT_DOC}..."):
                    engine.index_documents([DEFAULT_DOC])
                st.success("✅ Default document loaded!")
        
        return engine
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG engine: {str(e)}")
        st.info("💡 Make sure you've added HUGGINGFACE_API_TOKEN to Streamlit secrets")
        return None

# Initialize engine
if st.session_state.rag_engine is None:
    st.session_state.rag_engine = initialize_rag_engine()
    st.session_state.initialized = True

rag_engine = st.session_state.rag_engine

# --- Header ---
st.markdown("""
<div class='main-header'>
    <h1>🚢 PortOps RAG Chatbot</h1>
    <p>Ask questions about port operations - Powered by AI</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Document Management")
    
    # Show stats
    if rag_engine:
        try:
            stats = rag_engine.get_stats()
            st.metric("Documents Indexed", stats.get('documents', 0))
            st.metric("Vector Embeddings", stats.get('vectors', 0))
        except:
            st.info("Stats unavailable")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload New Document", 
        type=["pdf", "txt"],
        help="Upload PDF or TXT files"
    )
    
    if uploaded_file is not None and rag_engine:
        try:
            # Save uploaded file
            upload_path = UPLOADS_DIR / uploaded_file.name
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Index document
            with st.spinner(f"Processing {uploaded_file.name}..."):
                rag_engine.index_documents([str(upload_path)])
            
            st.success(f"✅ Indexed: {uploaded_file.name}")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to process: {str(e)}")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.success("Chat cleared!")
        st.rerun()
    
    # Save conversation button
    if st.button("💾 Save Conversation"):
        if st.session_state.history:
            try:
                fname = f"conversation_{int(time.time())}.json"
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
                st.success(f"Saved: {fname}")
            except Exception as e:
                st.error(f"Save failed: {e}")
        else:
            st.warning("No conversation to save")

# --- Main Chat Interface ---
st.header("💬 Chat Interface")

# Check if engine is ready
if not rag_engine:
    st.error("⚠️ RAG engine not initialized. Please check your configuration.")
    st.info("""
    **Setup Instructions for Streamlit Cloud:**
    1. Go to App Settings → Secrets
    2. Add your Hugging Face token:
    ```
    HUGGINGFACE_API_TOKEN = "hf_your_token_here"
    ```
    3. Get a free token at: https://huggingface.co/settings/tokens
    """)
    st.stop()

# Display conversation history
for message in st.session_state.history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(message['content'])
            if 'sources' in message and message['sources']:
                with st.expander("📚 View Sources"):
                    for source in message['sources']:
                        st.markdown(f"- `{source}`")

# Chat input
user_input = st.chat_input("Ask a question about port operations...")

if user_input:
    # Add user message
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = rag_engine.ask(user_input)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.markdown(f"- `{source}`")
                
                # Save to history
                st.session_state.history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.history.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "sources": []
                })

# --- Footer with Example Questions ---
st.markdown("---")
st.subheader("💡 Example Questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What are crane safety procedures?"):
        st.session_state.example_query = "What are crane safety procedures?"
        st.rerun()

with col2:
    if st.button("List emergency contacts"):
        st.session_state.example_query = "List emergency contacts"
        st.rerun()

with col3:
    if st.button("What weather restrictions apply?"):
        st.session_state.example_query = "What weather restrictions are specified?"
        st.rerun()

# Handle example query
if 'example_query' in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query
    
    # Add to history and get response
    st.session_state.history.append({"role": "user", "content": query})
    
    try:
        answer, sources = rag_engine.ask(query)
        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
    except Exception as e:
        st.session_state.history.append({
            "role": "assistant",
            "content": f"Error: {str(e)}",
            "sources": []
        })
    
    st.rerun()

# --- Info Section ---
with st.expander("ℹ️ About This Chatbot"):
    st.markdown("""
    ### How It Works
    This chatbot uses **Retrieval-Augmented Generation (RAG)** technology:
    
    1. 📄 **Documents** are split into chunks
    2. 🧠 **Embeddings** are created using Sentence Transformers
    3. 💾 **Stored** in ChromaDB vector database
    4. 🔍 **Retrieval** finds relevant context for your question
    5. 🤖 **LLM** generates accurate answers with sources
    
    ### Technology Stack
    - **Frontend:** Streamlit
    - **Embeddings:** Sentence Transformers (MiniLM-L6-v2)
    - **Vector DB:** ChromaDB
    - **LLM:** Hugging Face (Flan-T5-Small)
    - **Framework:** LangChain
    
    ### Features
    ✅ Semantic search over documents  
    ✅ Source citations for verification  
    ✅ Upload custom documents  
    ✅ Persistent storage  
    ✅ Free to use (no API costs with HF)
    """)
