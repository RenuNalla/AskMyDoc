"""
Groundwork — Night 3: FastAPI wrapper.

Wraps the ingest/query pipeline in a real web API instead of scripts.

Setup:
    pip install fastapi uvicorn
    (everything else already in requirements.txt from Night 1/2)

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs (FastAPI
generates this automatically — a nice thing to show in interviews too).

Endpoints:
    POST /ask        {"question": "..."}  -> {"answer": ..., "sources": [...]}
    POST /ingest      multipart file upload -> ingests a new document
    GET  /health      basic health check
"""

import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse the logic already built in query.py and ingest.py
from query import ask as run_query
from ingest import ingest as run_ingest

load_dotenv()

app = FastAPI(title="Groundwork", description="Grounded Q&A over your own documents")

# Allow a local frontend (or any origin, for demo purposes) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = run_query(payload.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {e}")
    return result


@app.post("/ingest")
def ingest_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    # Save the upload to a temp file, since ingest() expects a file path
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        run_ingest(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest: {e}")
    finally:
        os.remove(tmp_path)

    return {"status": "ingested", "filename": file.filename}