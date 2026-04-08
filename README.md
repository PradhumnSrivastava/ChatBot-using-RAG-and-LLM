# Project Working (Background Explanation)

This project follows a **Retrieval-Augmented Generation (RAG)** approach and works in two main phases:

1. **Ingestion Phase (Data Preparation)**
2. **Runtime Phase (Question Answering)**

---

## 1. Ingestion Phase (Data Preparation)

This phase prepares the document so that it can be efficiently searched later.

---

### Step 1: Document Loading

- The system loads data from a text file or PDF.
- It extracts raw textual content from the document.

---

### Step 2: Text Splitting

- The document is divided into smaller chunks.
- This is required because LLMs cannot process very large text at once.

**Example of chunks:**
- Chunk 1 → Introduction to Machine Learning  
- Chunk 2 → Supervised Learning Concepts  
- Chunk 3 → Neural Networks Overview  

- Overlapping is used to preserve context between chunks.

---

### Step 3: Embedding Generation

- Each chunk is converted into a numerical vector (embedding).
- These embeddings capture the semantic meaning of the text.

---

### Step 4: Storage in Vector Database (FAISS)

- All embeddings are stored in FAISS.
- FAISS allows fast similarity-based search.

---

### Result of Ingestion Phase

A searchable knowledge base is created where:
- Each chunk is represented as a vector  
- All vectors are stored in FAISS  

---

## 2. Runtime Phase (Question Answering)

This phase is executed when the user asks a question.

---

### Step 1: User Query

The user provides a question.

**Example:**
- What is deep learning?

---

### Step 2: Query Embedding

- The user query is converted into a vector.
- The same embedding model is used as in the ingestion phase.

---

### Step 3: Similarity Search (Core Step)

- The system searches the FAISS database.
- It finds the most relevant chunks by comparing vector similarity.

**Output:**
- Top 3–5 most relevant chunks

---

### Step 4: Context Creation

- The retrieved chunks are combined into a single context.

**Example:**
- Context = Chunk1 + Chunk2 + Chunk3  

---

### Step 5: LLM Input

- The system sends both context and question to the LLM.

**Format:**
- Context + Question  

---

### Step 6: Response Generation

- The LLM (Llama 3 via Groq) generates the answer.
- The model is instructed to answer only from the provided context.

If the answer is not present in the context:
- The model responds with: "I don't know"

---

## Complete Flow

Document → Split → Embeddings → FAISS  
Query → Embedding → Similar Chunks → Context → LLM → Answer  

---

## Core Idea

Instead of relying only on the LLM's internal knowledge:

- The system first retrieves relevant information  
- Then generates an answer based on that information  

---

## Analogy

This system works like a human reading a book:

1. A question is asked  
2. Relevant pages are found  
3. The answer is derived from those pages  

---

## With vs Without RAG

### Without RAG
- LLM answers from memory  
- Higher chance of incorrect answers  

### With RAG
- LLM answers from retrieved context  
- More accurate and reliable  

---

## Summary

- Documents are converted into embeddings  
- Stored in a vector database  
- Relevant information is retrieved for a query  
- LLM generates context-based answers  

This ensures accuracy, scalability, and reliability in document-based question answering systems.
