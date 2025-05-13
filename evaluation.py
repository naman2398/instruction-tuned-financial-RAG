"""
Financial RAG Pipeline - Evaluation Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Implements evaluation metrics for the RAG system.
  Loads a test dataset from FinLang/investopedia-instruction-tuning-dataset,
  implements F1 and ROUGE metrics for answer evaluation, and
  compares four different retrieval-generation combinations.
  Outputs performance metrics for each approach.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Token-based evaluation metrics (F1)
  II. Semantics | Probabilistic Models: 
     - Evaluation of semantic relevance in generated answers
  III. Language Modeling | Transformers: 
     - Evaluation of transformer-based language model outputs
     - Comparing performance of different model architectures
  IV. Applications | Custom Statistical or Symbolic: 
     - Domain-specific evaluation for financial QA task
     - ROUGE metrics for summarization/answer quality
     - Comparative evaluation of multiple system approaches

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import os
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline
import evaluate
from collections import Counter
from tqdm import tqdm

# Configuration variables
MAX_SAMPLES = 1000
TOP_K = 10
EMBED_MODEL = "FinLang/finance-embeddings-investopedia"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
BASE_MODEL = "google/flan-t5-small"
INSTR_MODEL = "Finetuned_model"
USE_CUDA = True
USE_RERANKER = True
MAX_LENGTH = 768
BATCH_SIZE = 8  # batch size for generation to maximize GPU utilization


def compute_f1(reference: str, prediction: str) -> float:
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    common = Counter(ref_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_data(max_samples: int):
    dataset = load_dataset("FinLang/investopedia-instruction-tuning-dataset", split="test")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset["Context"], dataset["Question"], dataset["Answer"]


def build_bm25(corpus):
    tokenized = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized)


def build_faiss(corpus, embed_model_name):
    embedder = SentenceTransformer(embed_model_name)
    embeddings = embedder.encode(corpus, show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embedder


def retrieve_bm25(bm25, corpus, query, top_k):
    scores = bm25.get_scores(query.split())
    idxs = np.argsort(scores)[::-1][:top_k]
    return [corpus[i] for i in idxs]


def retrieve_faiss(index, embedder, corpus, query, top_k):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    _, I = index.search(q_emb, top_k)
    return [corpus[i] for i in I[0]]


def retrieve_fusion(bm25, index, embedder, corpus, query, top_k, reranker=None):
    bm25_idxs = np.argsort(bm25.get_scores(query.split()))[::-1][:top_k]
    q_emb = embedder.encode([query], normalize_embeddings=True)
    _, vec_idxs = index.search(q_emb, top_k)
    candidates = set(bm25_idxs.tolist() + vec_idxs[0].tolist())
    if reranker:
        pairs = [(query, corpus[i]) for i in candidates]
        scores = reranker.predict(pairs)
        sorted_idxs = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
        final = sorted_idxs[:top_k]
    else:
        final = list(candidates)[:top_k]
    return [corpus[i] for i in final]


def main():
    # Load data
    contexts, questions, answers = load_data(MAX_SAMPLES)
    print(f"Loaded {len(questions)} samples.")

    # Build knowledge base
    corpus = contexts

    # Build retrieval structures
    print("Building BM25 index...")
    bm25 = build_bm25(corpus)
    print("Building FAISS index with embeddings...")
    faiss_index, embedder = build_faiss(corpus, EMBED_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL) if USE_RERANKER else None

    # Prepare LLM pipelines
    print("Loading generators...")
    device = 0 if USE_CUDA else -1
    gen_base = pipeline("text2text-generation", model=BASE_MODEL, tokenizer=BASE_MODEL, device=device, max_length=MAX_LENGTH)
    gen_instr = pipeline("text2text-generation", model=INSTR_MODEL, tokenizer=INSTR_MODEL, device=device, max_length=MAX_LENGTH)

    # Metrics
    rouge = evaluate.load("rouge")
    results = {}

    systems = [
        ("baseline1", lambda q: retrieve_bm25(bm25, corpus, q, TOP_K), gen_base),
        ("baseline2", lambda q: retrieve_faiss(faiss_index, embedder, corpus, q, TOP_K), gen_base),
        ("improved1", lambda q: retrieve_fusion(bm25, faiss_index, embedder, corpus, q, TOP_K, reranker), gen_base),
        ("improved2", lambda q: retrieve_fusion(bm25, faiss_index, embedder, corpus, q, TOP_K, reranker), gen_instr),
    ]

    for name, retriever, generator in systems:
        print(f"Evaluating {name}...")
        # Build all prompts
        prompts = [
            "Context: \n" + "\n".join(retriever(q)) + f"\nQuestion: {q}\nAnswer:"
            for q in questions
        ]
        # Batch generation with progress bar
        preds = []
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{name} generation", unit="batch"):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            batch_outputs = generator(batch_prompts, batch_size=BATCH_SIZE)
            # Handle variable pipeline output structures
            for out in batch_outputs:
                if isinstance(out, dict) and "generated_text" in out:
                    text = out["generated_text"]
                elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
                    text = out[0]["generated_text"]
                else:
                    text = str(out)
                preds.append(text.strip())

        # Compute metrics
        f1_scores = [compute_f1(gold, pred) for gold, pred in zip(answers, preds)]
        rouge_res = rouge.compute(predictions=preds, references=list(answers), rouge_types=["rougeL"])
        results[name] = {"F1": float(np.mean(f1_scores)), "ROUGE-L": rouge_res["rougeL"]}

    # Print summary
    print("\n=== Evaluation Results ===")
    for name, mets in results.items():
        print(f"{name}: F1 = {mets['F1']:.4f}, ROUGE-L = {mets['ROUGE-L']:.4f}")

if __name__ == "__main__":
    main()