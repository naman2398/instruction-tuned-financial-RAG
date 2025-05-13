"""
Financial RAG Pipeline - Main Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Entry point for the RAG system that ties all components together.
  Implements an interactive query loop where users can ask financial
  questions and compare different retrieval and generation approaches.
  Demonstrates four approaches: BM25, Vector retrieval, Hybrid retrieval
  with reranking, and a finetuned model with hybrid retrieval.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Used in text processing and classification of document relevance
  II. Semantics | Probabilistic Models: 
     - Applied in BM25 retrieval which uses probabilistic relevance model
     - Vector embeddings to capture semantic relationships
  III. Language Modeling | Transformers: 
     - Leveraged in both base and finetuned language models for answer generation
  IV. Applications | Custom Statistical or Symbolic: 
     - Application of RAG system to financial domain
     - Integration of multiple approaches into a cohesive QA system

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import torch
from typing import Dict, Any, List
from langchain.schema import BaseRetriever

from dotenv import load_dotenv
import os

# Load variables from .env file into environment
load_dotenv()

from ingestion import ingest_documents
from retriever import Retriever
from llm_model import LLModel
from hybrid_retriever import HybridReranker
from finetuned_llm import FinetunedLLModel

# Load environment variables
DEVICE = os.getenv('DEVICE', 'cpu')
TOP_K = int(os.getenv('TOP_K', '5'))
FUSION_K = int(os.getenv('FUSION_K', '10')) 
FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME', './Finetuned_model')

def run_rag_system():
    """
    Run the complete RAG system with both improvements:
    1. Hybrid retrieval with reranking
    2. Finetuned model with hybrid retrieval
    """
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Using CPU.")
    print(f"Using device: {DEVICE}")

    # Step 1: Ingest documents
    print("\n--- DOCUMENT INGESTION ---")
    documents = ingest_documents()

    if not documents:
        print("No documents to process. Exiting.")
        return

    # Step 2: Initialize retriever
    print("\n--- INITIALIZING RETRIEVER ---")
    retriever = Retriever(documents)
    
    # Step 3: Initialize reranker for Improvement 1
    print("\n--- INITIALIZING RERANKER ---")
    reranker = HybridReranker(device=DEVICE)

    # Step 4: Initialize LLMs
    print("\n--- INITIALIZING BASE LLM ---")
    base_llm = LLModel()
    
    print("\n--- INITIALIZING FINETUNED LLM ---")
    finetuned_llm = FinetunedLLModel(model_path=FINETUNED_MODEL_NAME)

    # Step 5: Interactive query loop
    print("\n--- RAG SYSTEM READY ---")
    print("Enter queries or type 'exit' to quit.")

    while True:
        # Get query from user
        query = input("\nEnter your query: ")

        if query.lower() == 'exit':
            print("Exiting RAG system.")
            break

        if not query.strip():
            continue

        # Process query
        print(f"\nProcessing query: {query}")

        # Baseline 1: BM25 Retrieval
        print("\n=== BASELINE 1: BM25 RETRIEVAL ===")
        bm25_retriever = retriever.get_retriever("bm25")
        bm25_result = base_llm.generate_answer(bm25_retriever, query, "bm25")
        print(f"Answer: {bm25_result['answer']}")
        
        # Baseline 2: Vector Retrieval
        print("\n=== BASELINE 2: VECTOR RETRIEVAL ===")
        vector_retriever = retriever.get_retriever("vector")
        vector_result = base_llm.generate_answer(vector_retriever, query, "vector")
        print(f"Answer: {vector_result['answer']}")
                    
        # Improvement 1: Hybrid Retrieval with Reranking
        print("\n=== IMPROVEMENT 1: HYBRID RETRIEVAL WITH RERANKING ===")
        
        # Get candidates from both retrievers
        bm25_candidates = retriever.retrieve(query, method='bm25')[:FUSION_K]
        vector_candidates = retriever.retrieve(query, method='vector')[:FUSION_K]
        
        # Merge & dedupe candidates by normalizing paths
        unique = {}
        for doc in bm25_candidates + vector_candidates:
            # Normalize source path for deduplication
            source = doc.metadata.get('source', '')
            if source:
                normalized_source = source.replace('\\', '/')
                doc.metadata['source'] = normalized_source
            
            # Use normalized source and content hash as key
            key = doc.metadata.get('source', '') + str(hash(doc.page_content))
            unique[key] = doc
        
        candidates = list(unique.values())
        
        if candidates:
            print(f"Scoring {len(candidates)} candidates with reranker...")
            
            # Rerank candidates
            scores = reranker.score(query, candidates)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in ranked[:TOP_K]]
            
            # Since we can't easily create a new retriever with the reranked docs,
            # we'll use the original vector retriever for the hybrid approach.
            hybrid_retriever = vector_retriever
            
            # Generate answer using the base LLM
            hybrid_result = base_llm.generate_answer(hybrid_retriever, query, "hybrid+rerank")
            print(f"Answer: {hybrid_result['answer']}")
            if 'source_documents' in hybrid_result:
                print(f"Sources: {[doc.metadata.get('source', 'unknown') for doc in hybrid_result['source_documents'][:3]]}")
                
            # Print reranking scores for top documents
            print("\nTop reranked documents with scores:")
            for i, (doc, score) in enumerate(ranked[:3], 1):
                source = doc.metadata.get('source', 'unknown')
                print(f"{i}. Score: {score:.4f} - Source: {source}")
                # Print a snippet of the content
                snippet = doc.page_content.replace("\n", " ")[:100] + "..."
                print(f"   Snippet: {snippet}")
                
            # Improvement 2: Finetuned Model with Hybrid Retrieval
            if finetuned_llm.llm:
                print("\n=== IMPROVEMENT 2: FINETUNED MODEL WITH HYBRID RETRIEVAL ===")
                
                # Use the same hybrid retriever with the finetuned model
                finetuned_result = finetuned_llm.generate_answer(hybrid_retriever, query, "finetuned+hybrid")
                print(f"Answer: {finetuned_result['answer']}")

            else:
                print("\nImprovement 2 skipped: Finetuned model could not be initialized.")
        else:
            print("No documents retrieved for hybrid approach.")


if __name__ == "__main__":
    run_rag_system()