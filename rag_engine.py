"""rag_engine.py - FINAL FIXED VERSION

This version properly handles HuggingFace token for Streamlit Cloud deployment.
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Updated LangChain imports (compatible with latest versions)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for document Q&A
    Handles both HUGGINGFACEHUB_API_TOKEN and HUGGINGFACE_API_TOKEN
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
        """
        Initialize the RAG engine.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            embedding_model: Sentence transformer model for embeddings
            hf_model: Hugging Face model for generation
            hf_token: Hugging Face API token (optional, will check env vars)
            use_openai: Whether to use OpenAI (not implemented in this version)
            openai_api_key: OpenAI API key (not used in this version)
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.hf_model = hf_model
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        
        # Get HF token - check multiple possible env var names
        if hf_token:
            self.hf_token = hf_token
        else:
            # Try different environment variable names
            self.hf_token = (
                os.environ.get("HUGGINGFACEHUB_API_TOKEN") or 
                os.environ.get("HUGGINGFACE_API_TOKEN") or
                None
            )
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings
        print("🔧 Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize or load vectorstore
        print("🔧 Initializing vector store...")
        self.vectorstore = self._init_vectorstore()
        
        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
        # Initialize LLM
        print("🔧 Initializing LLM...")
        self.llm = self._init_llm()
        
        # Build QA chain with custom prompt
        print("🔧 Building QA chain...")
        self.qa_chain = self._build_qa_chain()
        
        print("✅ RAG Engine initialized successfully!")
    
    def _init_vectorstore(self) -> Chroma:
        """
        Initialize or load Chroma vectorstore with persistence.
        """
        try:
            # Try to load existing vectorstore
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded existing vectorstore from {self.persist_directory}")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Creating new vectorstore (error loading existing: {e})")
            # Create new empty vectorstore
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return vectorstore
    
    def _init_llm(self):
        """
        Initialize the Language Model (Hugging Face).
        """
        if not self.hf_token:
            raise ValueError(
                "Hugging Face API token is required. "
                "Please add HUGGINGFACEHUB_API_TOKEN to your Streamlit secrets.\n"
                "Get a free token at: https://huggingface.co/settings/tokens"
            )
        
        try:
            # Initialize HuggingFace Hub with explicit token and task parameters
            llm = HuggingFaceHub(
                repo_id=self.hf_model,
                task="text2text-generation",  # Required for Flan-T5 models
                model_kwargs={
                    "temperature": 0.3,
                    "max_length": 512,
                },
                huggingfacehub_api_token=self.hf_token
            )
            print(f"✅ Initialized Hugging Face model: {self.hf_model}")
            return llm
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {str(e)}")
    
    def _build_qa_chain(self):
        """
        Build the Question-Answering chain with custom prompt.
        """
        # Custom prompt for port operations
        prompt_template = """You are an AI assistant for port operations. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer (be specific and cite information from the context):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """
        Load documents from file paths (PDF or TXT).
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        all_docs = []
        
        for path in file_paths:
            try:
                ext = Path(path).suffix.lower()
                
                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path, encoding='utf-8')
                else:
                    print(f"⚠️ Skipping unsupported file type: {path}")
                    continue
                
                docs = loader.load()
                all_docs.extend(docs)
                print(f"✅ Loaded: {Path(path).name} ({len(docs)} pages/sections)")
                
            except Exception as e:
                print(f"❌ Error loading {path}: {str(e)}")
                continue
        
        return all_docs
    
    def index_documents(self, file_paths: List[str]):
        """
        Index documents into the vector store.
        
        Args:
            file_paths: List of file paths to index
        """
        print(f"📄 Loading {len(file_paths)} document(s)...")
        docs = self.load_documents(file_paths)
        
        if not docs:
            raise ValueError("No documents loaded successfully")
        
        # Split documents into chunks
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
            # Ensure source is tracked
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = f"document_{i}"
        
        # Add to vectorstore
        print("💾 Adding to vector store...")
        self.vectorstore.add_documents(chunks)
        
        # Persist changes
        try:
            self.vectorstore.persist()
            print("✅ Vector store updated and persisted")
        except Exception as e:
            print(f"⚠️ Persistence warning: {e}")
    
    def ask(self, query: str) -> tuple[str, List[str]]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            query: The question to ask
            
        Returns:
            Tuple of (answer, list of source documents)
        """
        try:
            print(f"🔍 Processing query: {query}")
            
            # Get answer from QA chain
            result = self.qa_chain.invoke({"query": query})
            
            # Extract answer
            answer = result.get("result", "")
            
            # Extract sources
            source_docs = result.get("source_documents", [])
            sources = []
            
            for doc in source_docs[:3]:  # Limit to top 3 sources
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
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats
        """
        try:
            # Try to get collection stats
            collection = self.vectorstore._collection
            
            # Get document count
            doc_count = 0
            try:
                doc_count = collection.count()
            except:
                try:
                    docs = collection.get()
                    doc_count = len(docs.get('ids', []))
                except:
                    pass
            
            # Get last modification time
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
    
    # Check for HF token
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        print("❌ HuggingFace token not found in environment")
        print("Set one of these:")
        print("  export HUGGINGFACEHUB_API_TOKEN='your_token'")
        print("  export HUGGINGFACE_API_TOKEN='your_token'")
        print("\nGet a free token at: https://huggingface.co/settings/tokens")
        exit(1)
    
    # Initialize engine
    try:
        engine = RAGEngine(hf_token=hf_token)
        
        # Test with default document if it exists
        if Path("Port Operations Reference Manual.txt").exists():
            print("\n📄 Loading test document...")
            engine.index_documents(["Port Operations Reference Manual.txt"])
            
            print("\n❓ Testing query...")
            answer, sources = engine.ask("What are the crane safety procedures?")
            
            print(f"\n💬 Answer: {answer}")
            print(f"\n📚 Sources: {sources}")
            
            print("\n✅ RAG Engine test complete!")
        else:
            print("⚠️ Test document not found, but engine initialized successfully")
            
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
