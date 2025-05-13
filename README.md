# Instruction Tuned Financial QnA RAG

A Retrieval-Augmented Generation (RAG) QnA system focused on financial context with hybrid retrieval methods and instruction-tuned large language models.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Components](#components)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Performance Evaluation](#performance-evaluation)
- [Visualization](#visualization)

## Overview

This project implements a comprehensive RAG (Retrieval-Augmented Generation) system designed specifically for financial question answering. The system leverages multiple retrieval methods (BM25, vector search, and hybrid approaches) and compares base language models with instruction-tuned alternatives to provide accurate answers to financial queries.

Key features:
- Multiple document retrieval methods (BM25, vector-based, hybrid)
- Neural reranking of retrieved documents
- Base and fine-tuned language models for answer generation
- Comprehensive evaluation framework
- Embedding visualizations

## System Architecture

The system follows a modular architecture with the following flow:

1. **Document Ingestion**: Financial documents (PDFs) are loaded and split into smaller chunks
2. **Indexing**: Documents are indexed using both BM25 and vector embeddings (Pinecone)
3. **Retrieval**: When a query is received, relevant documents are retrieved using one or more methods
4. **Reranking**: For hybrid approaches, retrieved documents are reranked using a cross-encoder model
5. **Answer Generation**: A language model generates an answer based on the retrieved context and question
6. **Evaluation**: The system's performance is measured using metrics like F1 and ROUGE-L scores

## Components

1. **Document Processing**: Handles loading and chunking of financial documents
2. **Retrieval Methods**:
   - BM25: Lexical retrieval based on keyword matching
   - Vector: Semantic retrieval using dense embeddings
   - Hybrid: Combination of BM25 and vector retrieval with reranking
3. **Language Models**:
   - Base Model: google/flan-t5-small
   - Instruction-tuned Model: Custom model instruction-tuned on financial data
4. **Evaluation Framework**: Measures system performance using standard metrics
5. **Visualization Tools**: For embedding space analysis

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DhairyaUmrania/FinancialRAGPipeline.git
cd FinancialRAGPipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file with the following):
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DEVICE=cuda
TOP_K=5
FUSION_K=10
FINETUNED_MODEL_NAME=./Finetuned_model
```

4. Create a `documents` directory and add your financial PDFs

## Usage

### Running the RAG System

To start the interactive query system:
```bash
python main.py
```

This will:
1. Load and process your documents
2. Initialize all retrieval methods
3. Start an interactive query loop where you can ask financial questions

### Fine-tuning a Model

To fine-tune the language model on financial data:
```bash
python NLP_Finetuned.py
```

### Running Evaluation

To evaluate system performance:
```bash
python evaluation.py
```

### Visualizing Embeddings

To generate embedding visualizations:
```bash
python embedding_visualization.py
```

## File Descriptions

### Main Components

- **`main.py`**: Entry point for the RAG system that ties all components together
- **`ingestion.py`**: Handles loading and processing of financial documents
- **`retriever.py`**: Implements BM25 and vector-based document retrieval
- **`hybrid_retriever.py`**: Implements reranking for improved retrieval quality
- **`llm_model.py`**: Implements the base language model for generating answers
- **`finetuned_llm.py`**: Uses a fine-tuned model for improved answer generation

### Auxiliary Components

- **`NLP_Finetuned.py`**: Contains code for fine-tuning the language model
- **`evaluation.py`**: Implements evaluation metrics for the RAG system
- **`embedding_visualization.py`**: Visualizes embeddings for different retrieval methods
- **`requirements.txt`**: Lists all Python package dependencies

## Performance Evaluation

The system compares four different retrieval-generation combinations:
1. BM25 + base LLM
2. Vector retrieval + base LLM
3. Hybrid retrieval + base LLM
4. Hybrid retrieval + instruction-tuned LLM

Evaluation metrics include:
- F1 score
- ROUGE-L score

## Visualization

The embedding visualization component creates plots showing the relationship between:
- Query embeddings
- BM25 retrieved document embeddings
- Vector retrieved document embeddings
- Hybrid retrieved document embeddings with reranker scores

Dimensionality reduction methods include:
- t-SNE
- PCA
- UMAP

