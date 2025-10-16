"""rag_engine.py - ULTRA SIMPLE VERSION THAT WORKS

No complex LangChain wrappers, just basic components that work.
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class RAGEngine:
    """Simple RAG Engine that actually works."""
    
    def __init__(
        self,
        persist_directory: str = "chroma_persist",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_model: str = "microsoft/Phi-3-mini-4k-instruct",
        hf_token: Optional[str] = None,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        self.persist_directory = persist_directory
        self.hf_model = hf_model
        
        # Get token
        self.hf_token = hf_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
        
        if not self.hf_token:
            raise ValueError("HuggingFace token required. Get one at: https://huggingface.co/settings/tokens")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Embeddings
        print("🔧 Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Vector store
        print("🔧 Loading vector store...")
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        # API setup
        self.api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        print("✅ RAG Engine ready!")
    
    def _call_hf_api(self, prompt: str) -> str:
        """Call HuggingFace API with retries."""
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "temperature": 0.3}
        }
        
        for i in range(3):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('generated_text', str(result[0]))
                        return text.strip()
                    return str(result).strip()
                
                elif response.status_code == 503:
                    print(f"⏳ Model loading, waiting... (attempt {i+1}/3)")
                    time.sleep(10)
                else:
                    print(f"⚠️ API error {response.status_code}: {response.text[:200]}")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"⚠️ Error: {e}")
                if i < 2:
                    time.sleep(5)
                else:
                    raise
        
        return "Sorry, I couldn't generate an answer. Please try again."
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """Load text documents."""
        all_docs = []
        for path in file_paths:
            try:
                loader = TextLoader(path, encoding='utf-8')
                docs = loader.load()
                all_docs.extend(docs)
                print(f"✅ Loaded: {Path(path).name}")
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
        return all_docs
    
    def index_documents(self, file_paths: List[str]):
        """Index documents."""
        docs = self.load_documents(file_paths)
        if not docs:
            raise ValueError("No documents loaded")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"📦 Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata['source'] = chunk.metadata.get('source', f'doc_{i}')
        
        self.vectorstore.add_documents(chunks)
        try:
            self.vectorstore.persist()
        except:
            pass
        print("✅ Documents indexed")
    
    def ask(self, query: str) -> tuple:
        """Ask a question."""
        try:
            # Get relevant docs
            docs = self.vectorstore.similarity_search(query, k=3)
            
            if not docs:
                return "I don't have information to answer this question.", []
            
            # Build context
            context = "\n\n".join([d.page_content[:500] for d in docs])
            
            # Create prompt
            prompt = f"""Answer this question based on the context below. Be specific and concise.

Context: {context}

Question: {query}

Answer:"""
            
            # Get answer
            answer = self._call_hf_api(prompt)
            
            # Get sources
            sources = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                if source not in sources:
                    sources.append(source)
            
            return answer, sources[:3]
            
        except Exception as e:
            raise Exception(f"Error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        try:
            count = self.vectorstore._collection.count()
            return {"documents": count, "vectors": count, "last_index_time": "N/A"}
        except:
            return {"documents": 0, "vectors": 0, "last_index_time": "N/A"}
