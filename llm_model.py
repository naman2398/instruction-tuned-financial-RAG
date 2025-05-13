"""
Financial RAG Pipeline - Base Language Model Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Implements the base language model for generating answers.
  Loads a seq2seq model (default: google/flan-t5-small) and wraps
  it in a LangChain pipeline. Implements a generate_answer method
  that takes a retriever and query, constructs a prompt with
  retrieved context, and generates an answer.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Processes syntactic structures in prompts and generated text
  II. Semantics | Probabilistic Models: 
     - Uses probabilities for token generation
  III. Language Modeling | Transformers: 
     - T5 transformer architecture for sequence-to-sequence generation
     - Attention mechanisms to focus on relevant context
     - Encoder-decoder architecture for text generation
  IV. Applications | Custom Statistical or Symbolic: 
     - Application of language models to financial question answering
     - Prompt engineering for task-specific generation

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import os
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "google/flan-t5-small")
TOP_K = int(os.environ.get("TOP_K", 4))


class LLModel:
    """RAG‐style QA using an instruction‐tuned seq2seq model on GPU if available."""

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        self.llm = None

        try:
            # 1) Load seq2seq LM + tokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 2) Move to GPU if possible
            device = 0 if torch.cuda.is_available() else -1
            if device >= 0:
                model.to(f"cuda:{device}")
                print(f"Loading seq2seq LM on GPU: cuda:{device}")
            else:
                print("Loading seq2seq LM on CPU")

            # 3) Build HF text2text pipeline
            hf_pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=768,
                do_sample=True,
                temperature = 0.3 
            )

            # 4) Monkey‐patch for LangChain compatibility
            hf_pipe.prefix = ""
            hf_pipe.suffix = ""

            # 5) Wrap in LangChain
            self.llm = HuggingFacePipeline(pipeline=hf_pipe)
            print(f"Initialized text2text‐generation pipeline for: {model_name}")

        except Exception as e:
            print(f"Error initializing LLM: {e}")

    def generate_answer(self, retriever: Any, query: str, method: str = 'default') -> Dict[str, Any]:
        """
        Generate answer for `query` by:
          1) Retrieving documents via the given retriever
          2) Prompting Flan‐T5 to answer succinctly
        """
        if not self.llm:
            return {"answer": "LLM not initialized", "method": method}

        prompt_template = """
                    You are a helpful assistant. Answer the question based on the given context. If you don't know the answer, say "I don't know".Do not make up answers.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        try:
            result = chain({"query": query})
            answer = result.get("result", "").strip()
            return {
                "answer": answer,
                "source_documents": result.get("source_documents", []),
                "method": method
            }
        except Exception as e:
            print(f"Error during QA chain execution: {e}")
            return {"answer": f"Error: {e}", "method": method}