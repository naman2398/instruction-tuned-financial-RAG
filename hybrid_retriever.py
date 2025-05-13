"""
Financial RAG Pipeline - Hybrid Retriever Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Implements a reranking system to improve retrieval quality.
  Uses a cross-encoder model to score document relevance to a query.
  Takes results from both BM25 and vector retrievers and reranks them,
  outputting a list of scores for candidate documents.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Classification of document relevance to queries
  II. Semantics | Probabilistic Models: 
     - Uses semantic similarity scoring
     - Combines multiple probabilistic models (BM25 and vector similarity)
  III. Language Modeling | Transformers: 
     - Cross-encoder model built on transformer architecture
     - Uses contextual embeddings for relevance scoring
  IV. Applications | Custom Statistical or Symbolic: 
     - Custom reranking approach adapted for financial document retrieval

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.schema import Document

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "cross-encoder/stsb-roberta-base")


class HybridReranker:
    """
    Cross-encoder reranker that scores document relevance for a query.
    """
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device.startswith("cuda"):
            self.model.to(device)
        self.device = device
        print(f"Initialized reranker model: {model_name} on {device}")

    def score(self, query: str, docs: List[Document]) -> List[float]:
        combined = [f"{query}{self.tokenizer.sep_token}{doc.page_content}" for doc in docs]
        inputs = self.tokenizer(
            combined,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        return logits.cpu().tolist()