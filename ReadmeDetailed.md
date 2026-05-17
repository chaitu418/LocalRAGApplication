# local RAG application with Fastaapi + Langchain + Ollama + chromaDB

## Overview

This porject builds a local RAG(Retrival Augmented Generation ) system using:

- FastAPI
- Langchain
- ChromaDB
- Ollama
- HuggingFace embeddings

The system:
1.Read PDF documents
2.split them in to chunks
3.converts chunks in to embeddings
4.stores embeddings in chromaDB
5.Accepts user questions through FastAPI
6.retrives relavant chunks
7.sends context to llama3
8.returns intelligent answers

--

# architecure

```text

PDF
|
pypdf Loader
|
Text splitter
|
Embeddings
|
chromaDB
|
retriver
|
prompt
|
llama3
|
FastPAI response
```

--

# final project structure
```text
sam/
│
├── myenv/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── rag/
│   │   ├── ingest.py
│   │   └── chroma_db/
│   │
│   └── data/
│       └── docs/
│           └── pp.pdf
│
└── requirements.txt
```

# Step 1 — Create Virtual Environment

Go to project folder:

```bash
cd ~/projects/sam
```

Create venv:

```bash
python -m venv myenv
```

Activate venv.

## Git Bash

```bash
source myenv/Scripts/activate
```

## CMD

```cmd
myenv\Scripts\activate
```

Expected:

```text
(myenv)
```

---

# Step 2 — Install Dependencies

Upgrade pip:

```bash
python -m pip install -U pip
```

Install packages:

```bash
python -m pip install fastapi uvicorn
python -m pip install langchain
python -m pip install langchain-community
python -m pip install langchain-text-splitters
python -m pip install langchain-huggingface
python -m pip install langchain-ollama
python -m pip install chromadb
python -m pip install sentence-transformers
python -m pip install pypdf
```

---

# Step 3 — Add PDF

Place your PDF here:

```text
sam/app/data/docs/pp.pdf
```

---

# Step 4 — Create ingest.py

Create file:

```text
sam/app/rag/ingest.py
```

Code:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("../data/docs/pp.pdf")

documents = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
db = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="./chroma_db"
)

db.persist()

print("Documents indexed successfully.")
```

---

# Step 5 — Run Ingest

Go to rag folder:

```bash
cd app/rag
```

Run:

```bash
python ingest.py
```

Expected output:

```text
Documents indexed successfully.
```

This creates:

```text
app/rag/chroma_db
```

---
# What Happens During Ingest

```text
PDF
 ↓
Loaded by PyPDFLoader
 ↓
Split into chunks
 ↓
Embeddings created
 ↓
Stored in ChromaDB
```

---

# Step 6 — Install Ollama

Install Ollama from:

https://ollama.com

---

# Step 7 — Start Ollama

Open new terminal.

Run:

```bash
ollama serve
```

Keep terminal open.

---

# Step 8 — Download Llama3 Model

Open another terminal:

```bash
ollama pull llama3
```

Verify:

```bash
ollama list
```

Expected:

```text
llama3
```

---

# Step 9 — Create main.py

Create:

```text
sam/app/main.py
```

Code:

```python
from fastapi import FastAPI

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

app = FastAPI()

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma DB
db = Chroma(
    persist_directory="./app/rag/chroma_db",
    embedding_function=embedding
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM
llm = OllamaLLM(model="llama3")

@app.get("/")
def root():
    return {"message": "RAG API Running"}

@app.get("/ask")
def ask(q: str):

    # Retrieve relevant documents
    docs = retriever.invoke(q)

    # Combine context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{q}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return {
        "question": q,
        "answer": response
    }
```

---

# Step 10 — Run FastAPI

Go to project root:

```bash
cd ~/projects/sam
```

Run:

```bash
uvicorn app.main:app --reload
```

Expected:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

# Step 11 — Open Swagger UI

Open browser:

```text
http://localhost:8000/docs
```

---

# Step 12 — Test API

Expand:

```text
GET /ask
```

Click:

```text
Try it out
```

Enter:

```text
what is decorator?
```

Click:

```text
Execute
```

---

# Expected Response

```json
{
  "question": "what is decorator?",
  "answer": "A decorator in Python is..."
}
```

---

# API Endpoints

## Root Endpoint

```text
GET /
```

Response:

```json
{
  "message": "RAG API Running"
}
```

---

## Ask Endpoint

```text
GET /ask?q=your_question
```

Example:

```text
http://localhost:8000/ask?q=what is python decorator
```

---

# How Retrieval Works

## User Question

```text
what is decorator?
```

↓

## Embedding Generated

Question converted into vector.

↓

## ChromaDB Search

Finds semantically similar chunks.

↓

## Relevant Context Retrieved

Example:

```text
Decorators modify functions...
```

↓

## Prompt Built

```text
Context + Question
```

↓

## Llama3 Generates Answer

↓

## FastAPI Returns JSON

---

# Important Concepts

## Embeddings

Convert text into vectors.

Example:

```text
"Python decorator"
```

↓

```python
[0.21, -0.88, ...]
```

---

## ChromaDB

Vector database storing embeddings.

Used for:
- semantic search
- retrieval
- memory

---

## Retriever

Searches vector database using similarity.

---

## RAG

Retrieval-Augmented Generation.

Instead of LLM answering blindly:

```text
Retrieve context
+
Generate answer
```
---
## Add Multiple PDFs

Load all PDFs from folder.

---

## Add UI

Use:
- Streamlit
- React
- Next.js

---

## Add Authentication

JWT auth for APIs.

---

## Add Better Prompting

Use system prompts.

---

# Final Flow Summary

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Question
 ↓
Retriever
 ↓
Relevant Context
 ↓
Llama3
 ↓
Answer
```

---

# Technologies Used

- FastAPI
- LangChain
- ChromaDB
- Ollama
- Llama3
- HuggingFace Embeddings
- PyPDF

---

# What You Built

A complete local AI RAG backend capable of:

- chatting with PDFs
- semantic retrieval
- local LLM inference
- REST API serving
- vector search

This is foundational architecture for:
- enterprise AI assistants
- AI agents
- document chatbots
- autonomous QA systems
- internal knowledge systems