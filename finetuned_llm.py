"""
Financial RAG Pipeline - Finetuned Language Model Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Similar to llm_model.py but uses a finetuned model.
  Loads a finetuned seq2seq model from a specified path and
  implements the same generate_answer interface as the base model.
  Uses a prompt template specifically designed for the finetuned model.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Processes syntactic structures in prompts and generated text
  II. Semantics | Probabilistic Models: 
     - Uses probabilities for token generation
     - Domain-adapted token distributions
  III. Language Modeling | Transformers: 
     - Finetuned transformer model
     - Adapting pretrained model to domain-specific tasks
     - Transfer learning from general language model to financial domain
  IV. Applications | Custom Statistical or Symbolic: 
     - Domain adaptation for financial text
     - Task-specific finetuning for QA in finance domain

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import os
import torch
from typing import Any, Dict, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Constants from environment
FINETUNED_MODEL_NAME = os.environ.get("FINETUNED_MODEL_NAME", "./Finetuned_model")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TOP_K = int(os.environ.get("TOP_K", "3"))


class FinetunedLLModel:
    """RAG‐style QA using a finetuned seq2seq model on GPU if available."""

    def __init__(self, model_path: str = FINETUNED_MODEL_NAME):
        self.model_path = model_path
        self.llm = None

        try:
            # 1) Load finetuned seq2seq LM + tokenizer
            print(f"Loading finetuned model from: {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 2) Move to GPU if possible
            device = 0 if torch.cuda.is_available() and DEVICE.startswith("cuda") else -1
            if device >= 0:
                model.to(f"cuda:{device}")
                print(f"Loading finetuned seq2seq LM on GPU: cuda:{device}")
            else:
                print("Loading finetuned seq2seq LM on CPU")

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
            print(f"Initialized finetuned text2text‐generation pipeline from: {model_path}")

        except Exception as e:
            print(f"Error initializing finetuned LLM: {e}")

    def generate_answer(self, retriever: Any, query: str, method: str = 'finetuned') -> Dict[str, Any]:
        """
        Generate answer for `query` using the finetuned model:
          1) Retrieving documents via the given retriever
          2) Prompting the finetuned model to answer
        """
        if not self.llm:
            return {"answer": "Finetuned LLM not initialized", "method": method}

        # You may need to adjust this prompt template based on how your model was finetuned
        prompt_template = """
                        You are a helpful financial assistant. Answer the question based on the provided context. If you don't know the answer, say "I don't know".Do not make up answers.

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
            print(f"Error during finetuned QA chain execution: {e}")
            return {"answer": f"Error: {e}", "method": method}