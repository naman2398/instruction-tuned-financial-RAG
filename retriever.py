"""
Financial RAG Pipeline - Retriever Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Implements document retrieval functionality using both BM25 and Vector search.
  Sets up a HuggingFace embedding model for vector embeddings and manages
  the Pinecone index creation and document ingestion. Provides methods to
  retrieve documents using either BM25 or vector similarity.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - BM25 uses token-based matching (lexical/syntactic approach)
  II. Semantics | Probabilistic Models: 
     - BM25 implements probabilistic relevance model
     - Vector embeddings capture semantic relationships between texts
     - Word2Vec-like approaches for semantic representation (in embedding models)
  III. Language Modeling | Transformers: 
     - Uses transformer-based embeddings for vector representations
  IV. Applications | Custom Statistical or Symbolic: 
     - Domain-specific retrieval for financial information

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import os
from typing import List

from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY       = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT   = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME    = os.environ.get("PINECONE_INDEX_NAME", "my-index")
EMBEDDING_MODEL_NAME   = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE                 = os.environ.get("DEVICE", "cpu")
TOP_K                  = int(os.environ.get("TOP_K", 4))


class Retriever:
    def __init__(self, documents: List[Document] = None):
        # 1) embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": DEVICE}
        )

        # 2) optional BM25
        self.bm25_retriever = None
        if documents:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = TOP_K
            print(f"BM25 initialized on {len(documents)} docs")

        # 3) pinecone client/wrapper
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

        # 4) check or create index
        needs_ingest = False
        if not self.pinecone.has_index(PINECONE_INDEX_NAME):
            print(f"Index '{PINECONE_INDEX_NAME}' not found → creating")
            dim = self.embeddings.client.get_sentence_embedding_dimension()
            self.pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            needs_ingest = True
        else:
            # exists: check if empty
            idx = self.pinecone.Index(PINECONE_INDEX_NAME)
            stats = idx.describe_index_stats()
            total = sum(ns["vector_count"] for ns in stats["namespaces"].values())
            if total == 0:
                print(f"Index '{PINECONE_INDEX_NAME}' is empty → will ingest")
                needs_ingest = True
            else:
                print(f"Connected to non-empty index '{PINECONE_INDEX_NAME}' ({total} vectors)")

        # 5) connect vector store
        idx = self.pinecone.Index(PINECONE_INDEX_NAME)
        self.vector_store = PineconeVectorStore(index=idx, embedding=self.embeddings)

        # 6) ingest if needed
        if needs_ingest:
            if not documents:
                raise ValueError(
                    f"Pinecone index '{PINECONE_INDEX_NAME}' is empty and no documents were provided for ingestion"
                )

            batch_size = 32  # adjust up/down based on average document size
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                self.vector_store.add_documents(batch)
            print(f"Ingested {len(documents)} documents into Pinecone")

    def get_retriever(self, method: str):
        method = method.lower()
        if method == "bm25":
            if not self.bm25_retriever:
                raise ValueError("BM25 not initialized (pass documents at init).")
            return self.bm25_retriever
        if method == "vector":
            return self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        raise ValueError("Unknown method; choose 'bm25' or 'vector'")

    def retrieve(self, query: str, method: str = "vector") -> List[Document]:
        retr = self.get_retriever(method)
        return retr.get_relevant_documents(query)


