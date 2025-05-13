"""
Financial RAG Pipeline - Document Ingestion Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Handles the loading and processing of financial documents.
  Uses LangChain's DirectoryLoader and PyPDFLoader to load PDF documents,
  and splits documents into smaller chunks using RecursiveCharacterTextSplitter.
  Returns document chunks for further processing in the RAG pipeline.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Document chunking based on syntactic boundaries (sentences, paragraphs)
  II. Semantics | Probabilistic Models: 
     - Not directly applied in this module
  III. Language Modeling | Transformers: 
     - Not directly applied in this module
  IV. Applications | Custom Statistical or Symbolic: 
     - Application-specific document processing for financial domain

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import os
from typing import List

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_DIR        = os.environ.get("DATA_DIR", "documents")
CHUNK_SIZE      = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP   = int(os.environ.get("CHUNK_OVERLAP", 200))


def ingest_documents() -> List[Document]:
    """
    Load all PDFs under DATA_DIR, split them into text chunks, and return them.
    """
    print(f"Ingesting PDF documents from {DATA_DIR}…")

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"  • Loaded {len(docs)} raw PDF pages/documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"  • Split into {len(chunks)} chunks")

    return chunks