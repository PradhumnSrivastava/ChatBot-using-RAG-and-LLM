# RAG-Based Document Question Answering System

## 1. Project Overview

This project is a **Retrieval-Augmented Generation (RAG) based chatbot** that allows users to upload documents (PDF or text) and ask questions based on their content.

The system improves the accuracy of Large Language Models (LLMs) by retrieving relevant context from the document before generating responses. This significantly reduces hallucinations and ensures reliable answers.

---

## 2. System Architecture

The system is divided into two main phases:

### A. Ingestion Phase (Preprocessing)
- Document loading  
- Text splitting  
- Embedding generation  
- Storage in vector database (FAISS)  

### B. Retrieval and Generation Phase (Runtime)
- User query input  
- Retrieval of relevant chunks  
- Context construction  
- LLM-based response generation  

---

## 3. Workflow

### Step 1: Document Loading

Documents are loaded using:
- `TextLoader` (for text files)
- `PyPDFLoader` (for PDF files)

---

### Step 2: Text Splitting

```python
RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)
