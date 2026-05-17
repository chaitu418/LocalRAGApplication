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

# LLM
llm = OllamaLLM(model="llama3")


@app.get("/")
def root():
    return {"message": "RAG API Running"}


@app.get("/ask")
def ask(q: str):

    # Retrieve docs
    docs = retriever.invoke(q)

    # Combine text
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

    # LLM response
    response = llm.invoke(prompt)

    return {
        "question": q,
        "answer": response
    }