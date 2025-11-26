# 🩺 RAG Medical AI Assistant  
AI-powered medical question-answering system using Retrieval-Augmented Generation (RAG)

## 📌 Overview
The **RAG Medical AI Assistant** is an end-to-end system designed to provide safe, grounded, and explainable answers to medical questions by combining:

- **Semantic search over clinical reference documents**
- **Large Language Models (LLMs) for reasoning**
- **FastAPI backend for structured API access**
- **Vector storage (Pinecone) for efficient retrieval**

This project follows a modular architecture inspired by state-of-the-art AI engineering pipelines.

---

## 🚀 Features

### 🔎 Retrieval-Augmented Generation (RAG)
- Document chunking with recursive text splitting  
- Embedding generation using OpenAI or other models  
- Vector search using Pinecone  

### 🧠 AI Reasoning
- Response generation with LangChain chains  
- Context-injected prompts grounded in retrieved documents  
- Source citation and retrieval transparency  

### 🌐 Backend API
- Built with **FastAPI**
- Endpoints for:
  - `/chat` – conversational interface  
  - `/embed` – embedding and indexing new documents  
  - `/health` – health/status checks  

### 🐳 Containerized Deployment
- Dockerfile included  
- One-command spin-up using Docker Compose  

---

