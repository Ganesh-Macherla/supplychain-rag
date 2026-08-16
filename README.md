# Meridian Supply Chain RAG Assistant

A document-grounded Retrieval-Augmented Generation (RAG) assistant for querying supply chain, procurement, supplier performance, and policy documents using natural language.

## Project Overview

The system allows users to ask questions about supplier performance, procurement policies, delivery metrics, quality issues, penalties, and supply-chain risks.

It retrieves relevant information from the source documents and generates answers grounded only in the retrieved context. Policy conditions are also evaluated deterministically to reduce incorrect clause application.

## 🎥 Demo

[![Watch the Demo](https://img.youtube.com/vi/_Mz2-YFcmbE/maxresdefault.jpg)]([https://youtu.be/_Mz2-YFcmbE](https://www.youtube.com/watch?v=_Mz2-YFcmbE))

▶️ [Watch the full demo on YouTube]([https://youtu.be/_Mz2-YFcmbE](https://www.youtube.com/watch?v=_Mz2-YFcmbE))

## Key Features

- PDF-based document processing
- Recursive text chunking with contextual overlap
- `nomic-embed-text` embeddings through Ollama
- Persistent ChromaDB vector storage
- Top-K semantic similarity retrieval
- Local LLM-based question answering
- Cross-document question answering
- Deterministic supplier policy evaluation
- Document and page-level source references
- Hallucination control through document-grounded responses
- Streamlit-based interactive interface
- Fully local and cost-free AI pipeline

## Workflow

```text
PDF Documents
      |
      v
PDF Text Extraction
      |
      v
Recursive Chunking
      |
      v
Ollama Embeddings
      |
      v
ChromaDB
      |
      v
User Question
      |
      v
Semantic Retrieval
      |
      +------> Deterministic Policy Evaluation
      |
      v
Local LLM
      |
      v
Grounded Answer + Sources
```

## Project Statistics

- 2 source documents
- 24 document chunks
- Top-6 semantic retrieval
- Persistent ChromaDB vector database
- Local Ollama embedding model
- Local Ollama language model

 ## Tech Stack
- Python
- Streamlit
- ChromaDB
- Ollama
- PyPDF
- Requests

## Project Structure

```text
supplychain-reg/
├── app.py
├── ingest.py
├── rag.py
├── requirements.txt
├── data/
│   ├── Meridian_Procurement_Policy_Handbook_v4.2.pdf
│   └── Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf
├── .env.example
└── .gitignore
```

## Setup

### Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Install the required Ollama models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Document Ingestion

Run:
```bash
python ingest.py
```
The ingestion pipeline extracts text from the PDFs, creates overlapping chunks, generates embeddings, and stores them in ChromaDB.

### Run the Application
Start the Streamlit application:
```bash
streamlit run app.py
```
The application will be available at:
```bash
http://localhost:8501
```

## Example Query
Question:
```text
Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?
```
The system retrieves the relevant supplier-performance and procurement-policy information, evaluates the applicable policy clauses, generates a grounded response, and displays the supporting document pages.

## Data Sources
The project uses Meridian Components documents covering:

- Supplier performance
- Procurement policies
- Delivery performance
- Defect rates
- Supplier penalties
- Supply-chain risks
- Procurement and escalation rules

## Hallucination Control

The system restricts generated answers to the retrieved document context. Policy clauses with conditional requirements, such as consecutive-quarter thresholds, are evaluated separately before the final answer is generated.

If the required information cannot be established from the documents, the system reports that the information is unavailable rather than inventing a value or policy condition.
