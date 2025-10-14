import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

# --- Data Classes ---
@dataclass
class DocumentMetadata:
    source: str
    page: Optional[int] = None

# Initialize RAG engine
rag_engine = RAGEngine(
    persist_directory="chroma_persist",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    hf_model="google/flan-t5-small",
    use_openai=False  # Set True if you have an OpenAI API key
)

# --- RAG Engine ---
class RAGEngine:
    def __init__(
        self,
        persist_directory: str = "chroma_persist",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_model: str = "google/flan-t5-small",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the RAG engine.
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.hf_model = hf_model
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key

        # Embeddings
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)

        # Load or create vectorstore
        self.vectorstore = self._init_vectorstore()

        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever()

        # Initialize LLM
        if self.use_openai:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required when use_openai=True")
            self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0)
        else:
            self.llm = HuggingFaceHub(repo_id=self.hf_model, model_kwargs={"temperature": 0})

        # Build QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.retriever, return_source_documents=True
        )

    # --- Vectorstore initialization ---
    def _init_vectorstore(self) -> Chroma:
        """
        Initialize or load Chroma vectorstore (persisted).
        """
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        else:
            os.makedirs(self.persist_directory, exist_ok=True)
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

    # --- Document loading ---
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load PDF or TXT documents and return LangChain Document objects.
        """
        all_docs = []
        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path)
            else:
                print(f"Skipping unsupported file: {path}")
                continue
            docs = loader.load()
            all_docs.extend(docs)
        return all_docs

    # --- Index documents ---
    def index_documents(self, file_paths: List[str]):
        """
        Index a list of documents into Chroma vectorstore.
        """
        docs = self.load_documents(file_paths)

        # Chunk text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Add metadata for source tracking
        for i, doc in enumerate(chunks):
            if not hasattr(doc, "metadata"):
                doc.metadata = {}
            doc.metadata["source"] = getattr(doc, "source", f"doc_{i}")

        # Add to vectorstore
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

    # --- Ask a question ---
    def ask(self, query: str) -> (str, List[str]):
        """
        Ask a question and return the answer with source documents.
        """
        result = self.qa_chain({"query": query})
        answer = result.get("result", "")
        sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
        return answer, sources

    # --- Get vectorstore stats ---
    def get_stats(self) -> Dict[str, Any]:
        """
        Return simple stats about the current vectorstore.
        """
        try:
            collection = self.vectorstore._collection
            vectors = collection.count() if hasattr(collection, "count") else None
            documents = collection.get()["metadatas"] if hasattr(collection, "get") else None
            doc_count = len(documents) if documents else 0
            last_index_time = time.ctime(os.path.getmtime(self.persist_directory)) if os.path.exists(self.persist_directory) else "N/A"
            return {
                "documents": doc_count,
                "vectors": vectors or doc_count,
                "last_index_time": last_index_time,
                "retriever_type": "chroma_retriever",
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {"documents": 0, "vectors": 0, "last_index_time": "N/A", "retriever_type": "chroma_retriever"}

