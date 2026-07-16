"""
Groundwork — Night 1: Ingestion pipeline. FREE STACK VERSION.

Groundwork is a RAG app that answers questions strictly from your own
documents and always cites which chunk/source each answer came from —
grounded answers, not model guesses.

What this does:
1. Loads a document (PDF or .txt)
2. Splits it into overlapping chunks
3. Generates an embedding for each chunk using a LOCAL model
   (sentence-transformers — no API key, no cost, runs on your machine)
4. Stores {text, embedding, metadata} into a MongoDB Atlas collection

Run once per document you want to add to your knowledge base.

Setup:
    pip install pymongo sentence-transformers pypdf python-dotenv

Create a .env file next to this script with:
    MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/
    DB_NAME=rag_demo
    COLLECTION_NAME=chunks

Note: the first run will download the embedding model (~90MB), which
takes a minute. After that it's cached locally and runs instantly.

Usage:
    python ingest.py path/to/document.pdf
    python ingest.py path/to/notes.txt
"""

import os
import sys
import uuid

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.environ.get("DB_NAME", "rag_demo")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "chunks")

# Small, fast, free local embedding model. 384 dimensions.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Rough size limits for chunking (in characters, not tokens, to keep this simple).
CHUNK_SIZE = 1800       # ~roughly 400-500 tokens per chunk
CHUNK_OVERLAP = 200     # keeps context continuous across chunk boundaries

print("Loading local embedding model (first run downloads it, ~90MB)...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

mongo = MongoClient(MONGODB_URI)
collection = mongo[DB_NAME][COLLECTION_NAME]


def load_text(path: str) -> str:
    """Load raw text from a .pdf or .txt file."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph/sentence
    boundaries where possible so chunks don't cut mid-sentence."""
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # try to break at the last period/newline before the hard cutoff
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # step forward, keeping some overlap
    return chunks


def embed(text: str) -> list[float]:
    vector = embedder.encode(text, normalize_embeddings=True)
    return vector.tolist()


def ingest(path: str):
    print(f"Loading {path} ...")
    text = load_text(path)
    print(f"Loaded {len(text)} characters")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")

    docs = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}")
        vector = embed(chunk)
        docs.append({
            "_id": str(uuid.uuid4()),
            "text": chunk,
            "embedding": vector,
            "source": os.path.basename(path),
            "chunk_index": i,
        })

    if docs:
        collection.insert_many(docs)
        print(f"Inserted {len(docs)} chunks into {DB_NAME}.{COLLECTION_NAME}")

    print("\nNext step: create a Vector Search index on the 'embedding' field")
    print("in the MongoDB Atlas UI (Search > Create Search Index > Vector Search).")
    print(f"Dimensions: {len(docs[0]['embedding']) if docs else 'unknown'}, similarity: cosine")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py path/to/document.pdf")
        sys.exit(1)
    ingest(sys.argv[1])