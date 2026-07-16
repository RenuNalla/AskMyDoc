"""
Groundwork — Night 2: Retrieval + generation. FREE STACK VERSION.

What this does:
1. Takes a question
2. Embeds it with the same local model used in ingest.py
3. Runs MongoDB Atlas Vector Search to find the most relevant chunks
4. Stuffs those chunks into a prompt and asks Groq's free LLM API to answer
5. Returns the answer plus which chunks/sources were used

Setup:
    Same requirements.txt as ingest.py (pymongo, sentence-transformers, groq, python-dotenv)
    Get a free Groq API key at: https://console.groq.com/keys

Add to your .env file:
    GROQ_API_KEY=gsk_...
    (MONGODB_URI, DB_NAME, COLLECTION_NAME already there from ingest.py)

Before running this: create a Vector Search index in MongoDB Atlas on the
'embedding' field (Atlas UI > Search > Create Search Index > Vector Search),
named "vector_index", dimensions 384, similarity "cosine".

Usage:
    python query.py "What does this document say about X?"
"""

import os
import sys

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from groq import Groq

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("DB_NAME", "rag_demo")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "chunks")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "vector_index")
TOP_K = 4  # how many chunks to retrieve per question

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match ingest.py

print("Loading local embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

mongo = MongoClient(MONGODB_URI)
collection = mongo[DB_NAME][COLLECTION_NAME]

groq_client = Groq(api_key=GROQ_API_KEY)


def embed(text: str) -> list[float]:
    vector = embedder.encode(text, normalize_embeddings=True)
    return vector.tolist()


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Vector search MongoDB Atlas for the most relevant chunks."""
    query_vector = embed(question)
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "text": 1,
                "source": 1,
                "chunk_index": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )
    return f"""Answer the question using ONLY the context below. If the context
doesn't contain the answer, say you don't have enough information — do not
make anything up.

Context:
{context}

Question: {question}

Answer:"""


def ask(question: str) -> dict:
    chunks = retrieve(question)
    if not chunks:
        return {"answer": "No relevant information found in the knowledge base.", "sources": []}

    prompt = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    sources = [{"source": c["source"], "chunk_index": c["chunk_index"], "score": round(c["score"], 3)} for c in chunks]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = ask(question)

    print("\nAnswer:")
    print(result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, score {s['score']})")