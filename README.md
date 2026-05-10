


# AI RAG Chatbot with Fine-Tuned Data — Llama 3.1 + BGE-M3 + FAISS

**RAG-first customer-support chatbot using [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), FAISS retrieval, and LoRA/QLoRA fine-tuning on multilingual IT support tickets.**



<!-- Optional: add a banner/diagram here -->
<!-- 
<img width="900" alt="banner" src="https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/blob/main/assets/banner.png">
<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.svg">
-->

<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/RAG-finetune-Llama-3.1-8B-Instruct/blob/main/assets/model.png">


This repository contains an end-to-end **Retrieval-Augmented Generation (RAG)** project for customer-support automation. The goal is to build a multilingual IT-support chatbot that retrieves relevant solved tickets, uses them as grounding evidence, and then generates a professional support response using a fine-tuned Llama model.

Unlike a simple fine-tuning project, this project intentionally makes **RAG the core component**. The model is not trained only to memorize support answers. Instead, each supervised fine-tuning example is built around retrieved context from a FAISS-based knowledge base, and the validation pipeline also follows the same retrieve → rerank → generate workflow.

> ⚠️ Disclaimer: This project is only for research, education, and portfolio demonstration.
---



## Project Summary

- **Dataset:** Customer IT Support / Multilingual Customer Support Tickets (from [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?resource=download)) 
- **Retriever:** [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)  
- **Reranker:** [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)  
- **Generator:** [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  
- **Fine-tuning:** LoRA / QLoRA supervised fine-tuning  
- **Vector DB:** FAISS  
- **Task:** Generate grounded multilingual support replies from retrieved ticket evidence  

---

## Why This Project?

Customer-support chatbots need more than fluent generation. They need access to relevant historical issues, known fixes, ticket metadata, product context, and language-specific patterns.

This project combines:

- **retrieval** for grounding,
- **reranking** for better evidence selection,
- **fine-tuning** for support-style responses,
- and **validation** through the full retrieve → rerank → generate pipeline.

The final system is designed to answer support tickets using retrieved solved-ticket evidence instead of relying only on model memory.

---

## Repository Structure

```text
.
├── codes/
│   ├── 00_token_login.py                    # Hugging Face token login
│   ├── 01_build_rag_training_data.py         # RAG index + SFT data construction
│   ├── 02_train_llama_lora_or_qlora.py       # Llama LoRA/QLoRA fine-tuning
│   └── 03_validate_rag_lora_or_qlora.py      # RAG validation pipeline
│
├── assets/
│   ├── technical_details.md                  # detailed implementation notes
│   ├── requirements.txt                      # dependencies
│   └── rag_architecture.png                  # optional diagram
│
├── dataset/
│   └── Customer IT Support -Ticket Dataset/
│
├── outputs/
│   ├── processed/
│   ├── rag_store/
│   ├── sft_data/
│   ├── trainer_runs_lora_or_qlora/
│   ├── llama31_8b_rag_lora_or_qlora_adapter/
│   └── validation_results_lora_or_qlora/
│
└── README.md
```

For detailed pipeline design, configuration, artifacts, and troubleshooting, see [`assets/technical_details.md`](assets/technical_details.md).

---

## Pipeline

```text
Dataset
  ↓
Clean + deduplicate answered tickets
  ↓
Train / validation split
  ↓
BGE-M3 embeddings
  ↓
Train-only FAISS knowledge base
  ↓
RAG-grounded SFT examples
  ↓
Llama 3.1 LoRA/QLoRA fine-tuning
  ↓
Retrieve → rerank → generate validation
```

The key idea is that each training and validation prompt contains retrieved solved-ticket evidence.

---

## Scripts

### 1. Hugging Face Token Login

```bash
python codes/00_token_login.py
```

This authenticates access to the gated Llama model.

---

### 2. Build RAG Training Data

```bash
python codes/01_build_rag_training_data.py
```

This script builds:

```text
outputs/processed/train.csv
outputs/processed/validation.csv
outputs/rag_store/faiss_train.index
outputs/rag_store/corpus.jsonl
outputs/rag_store/train_vectors.npy
outputs/sft_data/train_sft_prompt_completion.jsonl
outputs/sft_data/validation_sft_prompt_completion.jsonl
```

---

### 3. Fine-Tune Llama

```bash
python codes/02_train_llama_lora_or_qlora.py
```

The script automatically chooses:

```text
CUDA available   → QLoRA
No CUDA          → CPU LoRA fallback
```

The same Llama 3.1 8B Instruct model is used in both cases.

---

### 4. Validate the RAG Chatbot

```bash
python codes/03_validate_rag_lora_or_qlora.py
```

This runs:

```text
validation ticket → retrieve → rerank → generate → evaluate
```

Outputs are saved to:

```text
outputs/validation_results_lora_or_qlora/
```

---

## Requirements

Install dependencies:

```bash
pip install -U pandas numpy scikit-learn tqdm ftfy faiss-cpu FlagEmbedding transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf safetensors huggingface_hub
```

A more detailed dependency list is available in [`assets/technical_details.md`](assets/technical_details.md).

---

## Example System Behavior

A customer reports:

```text
Our AWS deployment failed after a recent configuration update. The service is down and we need urgent help.
```

The system retrieves similar solved tickets, reranks them, and generates a grounded support response such as:

```text
Thank you for reporting the AWS deployment issue. We understand the urgency and the impact on your operations. Please provide deployment logs, recent configuration changes, affected services, and any CloudTrail or CloudFormation error messages. We recommend checking IAM permissions, service limits, and recent infrastructure updates while our cloud support team investigates further.
```

---

## Evaluation

The validation script reports both generation and retrieval metrics, including:

```text
lexical_f1
rouge_l_f1
top1_same_queue
top1_same_language
topk_same_queue
topk_same_language
topk_tag_overlap
best_faiss_score
best_rerank_score
```

Metrics are saved in:

```text
outputs/validation_results_lora_or_qlora/metrics_summary.json
```

---

## Notes

- Llama 3.1 access requires Hugging Face approval and a valid token.
- CPU training is supported but very slow for Llama 3.1 8B.
- The FAISS index is built only from the training split to reduce validation leakage.
- The project is intended for research, learning, and portfolio demonstration.

---

## Author’s Note

This project was designed to demonstrate a practical RAG-centered workflow for real-world support automation. The emphasis is on building the full system: retrieval, evidence construction, fine-tuning, reranking, generation, and validation.

For the full technical breakdown, see [`assets/technical_details.md`](assets/technical_details.md).
