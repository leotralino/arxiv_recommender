# 📚 ArXiv Recommender

<p align="center">
  <img src="assets/logo.png" width="200" alt="ArXiv Rec Logo">
</p>

<p align="center">
  <b>Automated research discovery using Bi-Encoder filtering and LLM-as-a-Judge ranking.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/managed_by-uv-purple.svg" alt="Managed by uv">
  <img src="https://img.shields.io/badge/linting-ruff-6340ac.svg" alt="Linting by Ruff">
  <img src="https://img.shields.io/badge/inference-Ollama-white.svg?logo=ollama&logoColor=black" alt="Ollama">
  <img src="https://img.shields.io/github/license/leotralino/arxiv_recommender" alt="License">
</p>

---

## 🚀 Overview

`arxiv_recommender` is a lightweight, local-first tool for staying ahead of the ArXiv curve. It implements a **two-stage paper ranking pipeline**:

1. **Fetch:** Retrieve the latest daily papers from ArXiv via API.
2. **Coarse Filtering:** High-speed semantic search using `sentence-transformers` (Bi-Encoders) to prune hundreds of papers based on your embedding profile.
3. **Fine Ranking:** LLM-as-a-Judge (local via Ollama) to rank the top candidates with qualitative reasoning.

## 📦 Installation

Ensure you have [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/) installed.

### 1. Setup the Project
```bash
# Clone and enter the repository
git clone [https://github.com/leotralino/arxiv_recommender.git](https://github.com/leotralino/arxiv_recommender.git)
cd arxiv_recommender

uv sync
```

### 2. Prepare the local LLM
```bash
# Pull the recommended model for ranking
ollama pull llama3.2:3b
```

### 3. Run pipeline
```bash
uv run arxiv-rec
```
