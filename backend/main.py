"""
main.py
FastAPI backend for ResearchLens.

Endpoints:
  POST /upload      — Upload PDF/TXT source documents
  POST /ask         — Ask a question (uses uploaded docs + optional web search)
  GET  /sources     — List currently loaded documents
  DELETE /sources   — Clear all loaded documents
  GET  /health      — Health check
"""

import os
import json
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from retriever import load_file
from agent import run_agent

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


load_dotenv()


#BASE_DIR = Path(__file__).resolve().parent
#FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(
    title="ResearchLens API",
    description="Evidence-grounded research agent with citations",
    version="1.0.0"
)
'''
app.mount(
#  "/frontend",
   StaticFiles(directory=FRONTEND_DIR),
   name="frontend"
)

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")
'''

# Allow frontend (same origin or localhost dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for parsed file chunks
# Key: filename, Value: list of chunk dicts
LOADED_DOCUMENTS: dict = {}

# Directory for temp file storage during upload processing
UPLOAD_DIR = tempfile.mkdtemp()

# Output log file for saving results
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "sample_outputs", "results.json")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    use_web: bool = True
    top_k: int = 6


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list
    chunks_used: int
    documents_loaded: int
    web_used: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": len(LOADED_DOCUMENTS)}


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF or TXT files.
    Parses them into chunks and stores in memory.
    """
    uploaded = []
    errors = []

    for file in files:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()

        if ext not in (".pdf", ".txt", ".md"):
            errors.append(f"{filename}: unsupported type (use PDF, TXT, or MD)")
            continue

        # Save temp file to disk for parsing
        temp_path = os.path.join(UPLOAD_DIR, filename)
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            chunks = load_file(temp_path, filename)
            LOADED_DOCUMENTS[filename] = chunks
            uploaded.append({"filename": filename, "chunks": len(chunks)})
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    return {
        "uploaded": uploaded,
        "errors": errors,
        "total_documents": len(LOADED_DOCUMENTS)
    }


@app.get("/sources")
def list_sources():
    """List all currently loaded documents and their chunk counts."""
    return {
        "documents": [
            {"filename": name, "chunks": len(chunks)}
            for name, chunks in LOADED_DOCUMENTS.items()
        ],
        "total": len(LOADED_DOCUMENTS)
    }


@app.delete("/sources")
def clear_sources():
    """Clear all loaded documents from memory."""
    LOADED_DOCUMENTS.clear()
    return {"message": "All documents cleared.", "total_documents": 0}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Main agent endpoint.
    Combines all loaded file chunks + optional web search,
    then returns a cited answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Flatten all file chunks into one list
    all_file_chunks = []
    for chunks in LOADED_DOCUMENTS.values():
        all_file_chunks.extend(chunks)

    # Run the agent
    result = run_agent(
        question=request.question,
        file_chunks=all_file_chunks,
        use_web=request.use_web,
        top_k=request.top_k
    )

    result["documents_loaded"] = len(LOADED_DOCUMENTS)

    # Save result to results.json for submission
    _save_result(result)

    return result


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

# Directory containing index.html / style.css. Adjust if your frontend
# files live elsewhere relative to this file.
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if not os.path.isdir(FRONTEND_DIR):
    # Fallback for setups where frontend files sit next to main.py
    FRONTEND_DIR = os.path.dirname(__file__)

# Mounted LAST (after all API routes above) so /upload, /ask, /sources,
# /health keep priority and only unmatched paths fall through to static
# files. html=True makes "/" resolve to index.html automatically, and
# every other file in FRONTEND_DIR (style.css, images, etc.) is now
# actually served instead of 404ing.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_result(result: dict):
    """Append each Q&A result to results.json for the submission sample outputs."""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    existing = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append({
        "timestamp": datetime.utcnow().isoformat(),
        **result
    })

    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)