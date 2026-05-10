


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



