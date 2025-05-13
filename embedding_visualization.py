"""
Financial RAG Pipeline - Embedding Visualization Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Visualizes embeddings for different retrieval methods.
  Generates sample financial queries, retrieves documents using
  BM25, vector, and hybrid approaches, and reduces embedding
  dimensions using t-SNE, PCA, or UMAP. Creates visualizations
  showing the relationship between query embeddings and
  document embeddings from different retrieval methods.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Not directly applied in this module
  II. Semantics | Probabilistic Models: 
     - Visualization of semantic embeddings
     - Vector space representations of text meaning
     - Dimensionality reduction of semantic spaces (t-SNE, PCA, UMAP)
  III. Language Modeling | Transformers: 
     - Analysis of transformer-derived embeddings
  IV. Applications | Custom Statistical or Symbolic: 
     - Visualization techniques for embedding analysis
     - Comparative visualization of different retrieval approaches
     - Domain-specific analysis of financial text embeddings

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Tuple
from langchain.schema import Document

load_dotenv()
from ingestion import ingest_documents
from retriever import Retriever
from hybrid_retriever import HybridReranker

DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
TOP_K = int(os.getenv('TOP_K', '5'))
FUSION_K = int(os.getenv('FUSION_K', '10'))
N_QUERIES = 5  # Number of sample queries to plot


def get_sample_queries():
    """Generate sample financial queries."""
    return [
        "What are the lending activities of The First of Long Island Corporation?",
        # "How does the company manage liquidity risk?",
        # "What were the key financial results in the latest quarter?",
        # "Explain the company's investment strategy.",
        # "What are the major risks facing the corporation?",
    ]


def normalize_embeddings(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize embedding vectors to unit length."""
    return [emb / np.linalg.norm(emb) for emb in embeddings]


def get_document_embeddings(retriever, query: str, method: str, k: int = TOP_K) -> Tuple[List[Document], List[np.ndarray]]:
    """Get documents and their embeddings for a given query and retrieval method."""
    docs = retriever.retrieve(query, method=method)[:k]
    
    embeddings = []
    for doc in docs:
        if hasattr(doc, 'embedding') and doc.embedding is not None:
            embeddings.append(doc.embedding)
        else:
            content = doc.page_content
            emb = retriever.embeddings.embed_query(content)
            embeddings.append(emb)
    
    return docs, embeddings


def get_hybrid_documents(retriever, reranker, query: str, k: int = TOP_K) -> Tuple[List[Document], List[np.ndarray], List[float]]:
    """Get hybrid retrieval documents and their embeddings."""
    bm25_candidates = retriever.retrieve(query, method='bm25')[:FUSION_K]
    vector_candidates = retriever.retrieve(query, method='vector')[:FUSION_K]
    
    unique = {}
    for doc in bm25_candidates + vector_candidates:
        # Use content hash as key for deduplication
        key = str(hash(doc.page_content))
        unique[key] = doc
    
    candidates = list(unique.values())
    
    embeddings = []
    for doc in candidates:
        if hasattr(doc, 'embedding') and doc.embedding is not None:
            embeddings.append(doc.embedding)
        else:
            content = doc.page_content
            emb = retriever.embeddings.embed_query(content)
            embeddings.append(emb)
    
    # Rerank
    scores = reranker.score(query, candidates)
    ranked = sorted(zip(candidates, embeddings, scores), key=lambda x: x[2], reverse=True)
    
    # Get top documents, embeddings, and scores
    top_docs = [doc for doc, _, _ in ranked[:k]]
    top_embeddings = [emb for _, emb, _ in ranked[:k]]
    top_scores = [score for _, _, score in ranked[:k]]
    
    return top_docs, top_embeddings, top_scores


def reduce_dimensions(embeddings: List[np.ndarray], method: str = 'tsne', n_components: int = 2):
    """Reduce dimensionality of embeddings for visualization."""

    if isinstance(embeddings[0], list) or isinstance(embeddings[0], np.ndarray):
        X = np.array(embeddings)
    else:
        X = np.array([np.array(emb) for emb in embeddings])
    

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=min(30, len(X)-1), random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
  
    reduced = reducer.fit_transform(X)
    return reduced


def plot_embeddings(query_embeddings, bm25_embeddings_list, vector_embeddings_list, 
                   hybrid_embeddings_list, hybrid_scores_list, queries, reduction='tsne'):
    """Plot embeddings for queries and different retrieval methods."""
    n_queries = len(queries)
    fig, axes = plt.subplots(n_queries, 1, figsize=(12, 5*n_queries))
    
    if n_queries == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):

        all_embeddings = [query_embeddings[i]] + bm25_embeddings_list[i] + vector_embeddings_list[i] + hybrid_embeddings_list[i]
        

        reduced = reduce_dimensions(all_embeddings, method=reduction)
        

        x_min, x_max = reduced[:, 0].min(), reduced[:, 0].max()
        y_min, y_max = reduced[:, 1].min(), reduced[:, 1].max()
        

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        

        x_range = x_max - x_min
        y_range = y_max - y_min
        

        zoom_factor = 3
        

        x_new_min = x_center - zoom_factor * x_range / 2
        x_new_max = x_center + zoom_factor * x_range / 2
        y_new_min = y_center - zoom_factor * y_range / 2
        y_new_max = y_center + zoom_factor * y_range / 2
        

        ax.set_xlim(x_new_min, x_new_max)
        ax.set_ylim(y_new_min, y_new_max)
        

        bm25_start = 1
        bm25_end = bm25_start + len(bm25_embeddings_list[i])
        vector_start = bm25_end
        vector_end = vector_start + len(vector_embeddings_list[i])
        hybrid_start = vector_end
        hybrid_end = hybrid_start + len(hybrid_embeddings_list[i])
        
        ax.scatter(reduced[0, 0], reduced[0, 1], c='purple', marker='*', s=300, 
                  label='Query', zorder=10, edgecolors='black', linewidths=1)
        
        ax.scatter(reduced[bm25_start:bm25_end, 0], reduced[bm25_start:bm25_end, 1], 
                  c='blue', marker='o', s=100, label='BM25', alpha=0.7, 
                  edgecolors='black', linewidths=0.5)
        
        ax.scatter(reduced[vector_start:vector_end, 0], reduced[vector_start:vector_end, 1], 
                  c='green', marker='s', s=100, label='Vector', alpha=0.7,
                  edgecolors='black', linewidths=0.5)
        
        # Plot Hybrid results (larger points with color gradient, no text)
        scatter = ax.scatter(reduced[hybrid_start:hybrid_end, 0], reduced[hybrid_start:hybrid_end, 1], 
                  c=hybrid_scores_list[i], marker='^', s=120, label='Hybrid', alpha=0.9, 
                  cmap='YlOrRd', vmin=min(hybrid_scores_list[i]), vmax=max(hybrid_scores_list[i]),
                  edgecolors='black', linewidths=0.5)
        
        # Add colorbar for hybrid scores
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Reranker Score')
        
        # Add query text and labels
        ax.set_title(f'Query: "{queries[i]}"', fontsize=14)
        ax.set_xlabel(f'{reduction.upper()} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{reduction.upper()} Dimension 2', fontsize=12)
        ax.legend(loc='best', fontsize=12)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_embeddings(reduction_method='tsne'):
    """Main function to generate embedding visualizations."""
    print("Loading documents...")
    documents = ingest_documents()
    
    print("Initializing retriever...")
    retriever = Retriever(documents)
    
    print("Initializing reranker...")
    reranker = HybridReranker(device=DEVICE)
    
    print("Getting sample queries...")
    queries = get_sample_queries()
    queries = queries[:N_QUERIES]  # Limit to N_QUERIES
    
    # Store results for all queries
    query_embeddings = []
    bm25_embeddings_list = []
    vector_embeddings_list = []
    hybrid_embeddings_list = []
    hybrid_scores_list = []
    
    for query in queries:
        print(f"Processing query: {query}")
        
        # Get query embedding
        query_emb = retriever.embeddings.embed_query(query)
        query_embeddings.append(query_emb)
        
        # Get BM25 documents and embeddings
        _, bm25_embeddings = get_document_embeddings(retriever, query, "bm25")
        bm25_embeddings_list.append(bm25_embeddings)
        
        # Get Vector documents and embeddings
        _, vector_embeddings = get_document_embeddings(retriever, query, "vector")
        vector_embeddings_list.append(vector_embeddings)
        
        # Get Hybrid documents, embeddings, and scores
        _, hybrid_embeddings, hybrid_scores = get_hybrid_documents(retriever, reranker, query)
        hybrid_embeddings_list.append(hybrid_embeddings)
        hybrid_scores_list.append(hybrid_scores)
    
    # Plot embeddings
    print(f"Plotting embeddings using {reduction_method}...")
    fig = plot_embeddings(query_embeddings, bm25_embeddings_list, vector_embeddings_list, 
                        hybrid_embeddings_list, hybrid_scores_list, queries, reduction=reduction_method)
    
    fig.savefig(f'embedding_visualization_{reduction_method}_clean.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved as embedding_visualization_{reduction_method}_clean.png")
    
    return fig


if __name__ == "__main__":
    # You can choose from 'tsne', 'pca', or 'umap' for dimensionality reduction
    # visualize_embeddings(reduction_method='tsne')
    
    # If you want to try other methods too:
    visualize_embeddings(reduction_method='pca')
    # visualize_embeddings(reduction_method='umap')