"""rag_engine.py - SIMPLIFIED WORKING VERSION

Uses direct Hugging Face Inference API instead of LangChain's wrapper.
This avoids the InferenceClient compatibility issues.
"""

import os
import time
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for document Q&A
    Uses direct HuggingFace API calls to avoid compatibility issues
    """
    
    def __init__(
        self,
        persist_directory: str = "chroma_persist",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_model: str = "google/flan-t5-small",
        hf_token: Optional[str] = None,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the RAG engine."""
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.hf_model = hf_model
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        
        # Get HF token
        if hf_token:
            self.hf_token = hf_token
        else:
            self.hf_token = (
                os.environ.get("HUGGINGFACEHUB_API_TOKEN") or 
                os.environ.get("HUGGINGFACE_API_TOKEN") or
                None
            )
        
        if not self.hf_token:
            raise ValueError(
                "Hugging Face API token is required. "
                "Get a free token at: https://huggingface.co/settings/tokens"
            )
        
        # Create persist directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings
        print("🔧 Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vectorstore
        print("🔧 Initializing vector store...")
        self.vectorstore = self._init_vectorstore()
        
        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Set up HuggingFace API endpoint
        self.api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        print("✅ RAG Engine initialized successfully!")
    
    def _init_vectorstore(self) -> Chroma:
        """Initialize or load Chroma vectorstore."""
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded vectorstore from {self.persist_directory}")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Creating new vectorstore: {e}")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return vectorstore
    
    def _query_huggingface(self, prompt: str, max_retries: int = 3) -> str:
        """
        Query HuggingFace API directly with retry logic.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                        elif isinstance(result[0], str):
                            return result[0].strip()
                    elif isinstance(result, dict) and 'generated_text' in result:
                        return result['generated_text'].strip()
                    
                    return str(result).strip()
                
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    wait_time = 5 * (attempt + 1)
                    print(f"⏳ Model loading... waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    if attempt == max_retries - 1:
                        raise Exception(error_msg)
                    print(f"⚠️ {error_msg}, retrying...")
                    time.sleep(2)
                    
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise Exception("Request timed out after 30 seconds")
                print(f"⏳ Timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"HuggingFace API error: {str(e)}")
                print(f"⚠️ Error: {e}, retrying...")
                time.sleep(2)
        
        raise Exception("Failed to get response after all retries")
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """Load documents from file paths."""
        all_docs = []
        
        for path in file_paths:
            try:
                ext = Path(path).suffix.lower()
                
                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path, encoding='utf-8')
                else:
                    print(f"⚠️ Skipping unsupported file: {path}")
                    continue
                
                docs = loader.load()
                all_docs.extend(docs)
                print(f"✅ Loaded: {Path(path).name} ({len(docs)} pages)")
                
            except Exception as e:
                print(f"❌ Error loading {path}: {str(e)}")
                continue
        
        return all_docs
    
    def index_documents(self, file_paths: List[str]):
        """Index documents into the vector store."""
        print(f"📄 Loading {len(file_paths)} document(s)...")
        docs = self.load_documents(file_paths)
        
        if not docs:
            raise ValueError("No documents loaded successfully")
        
        # Split documents
        print("✂️ Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        print(f"📦 Created {len(chunks)} chunks")
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = f"document_{i}"
        
        # Add to vectorstore
        print("💾 Adding to vector store...")
        self.vectorstore.add_documents(chunks)
        
        try:
            self.vectorstore.persist()
            print("✅ Vector store updated and persisted")
        except Exception as e:
            print(f"⚠️ Persistence warning: {e}")
    
    def ask(self, query: str) -> tuple[str, List[str]]:
        """
        Ask a question and get an answer with sources.
        """
        try:
            print(f"🔍 Processing query: {query}")
            
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(query)
            
            if not relevant_docs:
                return "I don't have enough information to answer this question.", []
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # Create prompt
            prompt = f"""You are an AI assistant for port operations. Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer (be specific and concise):"""
            
            # Query HuggingFace API
            print("🤖 Generating answer...")
            answer = self._query_huggingface(prompt)
            
            # Extract sources
            sources = []
            for doc in relevant_docs[:3]:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', '')
                
                if page:
                    source_info = f"{source} (Page {page})"
                else:
                    source_info = source
                
                sources.append(source_info)
            
            print(f"✅ Generated answer with {len(sources)} sources")
            
            return answer, sources
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            collection = self.vectorstore._collection
            
            doc_count = 0
            try:
                doc_count = collection.count()
            except:
                try:
                    docs = collection.get()
                    doc_count = len(docs.get('ids', []))
                except:
                    pass
            
            last_modified = "N/A"
            if os.path.exists(self.persist_directory):
                try:
                    last_modified = time.ctime(os.path.getmtime(self.persist_directory))
                except:
                    pass
            
            return {
                "documents": doc_count,
                "vectors": doc_count,
                "last_index_time": last_modified,
                "embedding_model": self.embedding_model,
                "llm_model": self.hf_model
            }
            
        except Exception as e:
            print(f"⚠️ Stats error: {e}")
            return {
                "documents": 0,
                "vectors": 0,
                "last_index_time": "N/A",
                "embedding_model": self.embedding_model,
                "llm_model": self.hf_model
            }


# Test function
if __name__ == "__main__":
    print("🧪 Testing RAG Engine...")
    
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        print("❌ HuggingFace token not found")
        print("Get a free token at: https://huggingface.co/settings/tokens")
        exit(1)
    
    try:
        engine = RAGEngine(hf_token=hf_token)
        
        if Path("Port Operations Reference Manual.txt").exists():
            print("\n📄 Loading test document...")
            engine.index_documents(["Port Operations Reference Manual.txt"])
            
            print("\n❓ Testing query...")
            answer, sources = engine.ask("What are the crane safety procedures?")
            
            print(f"\n💬 Answer: {answer}")
            print(f"\n📚 Sources: {sources}")
            print("\n✅ RAG Engine test complete!")
        else:
            print("⚠️ Test document not found")
            
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
