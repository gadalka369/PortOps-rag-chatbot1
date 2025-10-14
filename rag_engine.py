
"""rag_engine.py (Streamlit Cloud verified)

Lightweight defaults:
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2
  - HF generator: google/flan-t5-small

The engine exposes:
  - add_documents(filepaths: List[str]) -> int
  - answer_query(query: str, top_k=3, max_tokens=128, hf_model=None, force_openai=False, force_hf=False)
  - get_stats() -> Dict
  - clear_persisted_index()
  - add_default_if_empty(default_filepath: str)
"""

import os
import shutil
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    source: str
    page: Optional[int] = None

class RAGEngine:
    def __init__(self, persist_directory: str = "chroma_persist", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hf_model_default: str = "google/flan-t5-small"):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.hf_model_default = hf_model_default
        # embeddings
        self.embedding = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        # initialize or create chroma persist dir
        self._init_vectorstore()
        # text splitter tuned for shorter chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        # prompt template - use triple-quoted string to avoid escape issues
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful expert on port operations. Use ONLY the provided context to answer.
If the answer is not in the context, say 'I don't know based on the provided documents.'

Context:
{context}

Question: {question}

Answer:"""
        )

    def _init_vectorstore(self):
        try:
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            logger.info("Chroma initialized.")
        except Exception as e:
            logger.error(f"Chroma init failed: {e}")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)

    def clear_persisted_index(self):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)
        self._init_vectorstore()

    def _load_file(self, filepath: str) -> List:
        p = Path(filepath)
        ext = p.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(p))
            docs = loader.load_and_split()
        elif ext == ".txt":
            loader = TextLoader(str(p), encoding="utf-8")
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return docs

    def add_documents(self, filepaths: List[str]) -> int:
        to_add_docs = []
        for fp in filepaths:
            try:
                loaded = self._load_file(fp)
            except Exception as e:
                logger.error(f"Failed loading {fp}: {e}")
                continue
            if not isinstance(loaded, list):
                loaded = [loaded]
            for doc in loaded:
                chunks = self.text_splitter.split_documents([doc])
                for c in chunks:
                    c.metadata["__source_file"] = Path(fp).name
                    if "page" in c.metadata:
                        try:
                            c.metadata["__source_page"] = int(c.metadata.get("page"))
                        except:
                            c.metadata["__source_page"] = None
                to_add_docs.extend(chunks)
        if to_add_docs:
            self.vectorstore.add_documents(to_add_docs)
        return len(to_add_docs)

    def add_default_if_empty(self, default_filepath: str) -> int:
        """If the index is empty, add the default file and return number of chunks added."""
        try:
            collection = self.vectorstore._collection
            has_data = False
            try:
                get_res = collection.get()
                metadatas = get_res.get("metadatas", [])
                if metadatas:
                    has_data = True
            except Exception:
                has_data = False
            if not has_data and os.path.exists(default_filepath):
                return self.add_documents([default_filepath])
            return 0
        except Exception as e:
            logger.error(f"add_default_if_empty error: {e}")
            return 0

    def _get_llm(self, max_tokens: int = 128, hf_model: Optional[str] = None, force_openai: bool = False, force_hf: bool = False):
        openai_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_key and not force_hf:
            try:
                llm = OpenAI(openai_api_key=openai_key, max_tokens=max_tokens, temperature=0.1)
                logger.info("Using OpenAI LLM.")
                return llm
            except Exception as e:
                logger.warning(f"OpenAI init failed, falling back to HF: {e}")
        hf_model = hf_model or self.hf_model_default
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
            pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_length=max_tokens)
            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Using HF model: {hf_model}")
            return llm
        except Exception as e:
            logger.error(f"Failed to instantiate HF pipeline for {hf_model}: {e}")
            raise

    def answer_query(self, query: str, top_k: int = 3, max_tokens: int = 128, hf_model: Optional[str] = None, force_openai: bool = False, force_hf: bool = False) -> Dict[str, Any]:
        if not query or query.strip() == "":
            return {"answer": "Please ask a non-empty question.", "sources": []}
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        llm = self._get_llm(max_tokens=max_tokens, hf_model=hf_model, force_openai=force_openai, force_hf=force_hf)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": self.prompt})
        result = qa_chain({"query": query})
        answer = result.get("result") or result.get("answer") or ""
        source_docs = result.get("source_documents", []) or []
        sources = []
        seen = set()
        for sd in source_docs:
            md = sd.metadata or {}
            src_name = md.get("__source_file") or md.get("source") or "unknown"
            page = md.get("__source_page") or md.get("page")
            key = f"{src_name}:{page}"
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": src_name, "page": page})
        return {"answer": answer, "sources": sources}

    def get_stats(self) -> Dict[str, Any]:
        try:
            collection = self.vectorstore._collection
            vectors = collection.count() if hasattr(collection, "count") else None
            documents = collection.get()["metadatas"] if hasattr(collection, "get") else None
            doc_count = len(documents) if documents else 0
            return {"documents": doc_count, "vectors": vectors or doc_count, "last_index_time": time.ctime(os.path.getmtime(self.persist_directory)) if os.path.exists(self.persist_directory) else "N/A", "retriever_type": "chroma_retriever"}
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"documents": 0, "vectors": 0, "last_index_time": "N/A", "retriever_type": "chroma_retriever"}
          
