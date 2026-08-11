"""
retriever.py
Handles two source types:
  1. Local files (PDF / TXT) — parsed into chunks
  2. Web search via DuckDuckGo — returns snippets with URLs
Then ranks all chunks by TF-IDF similarity to the query.
"""

import os
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_pdf(filepath: str) -> str:
    """Extract raw text from a PDF file."""
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def parse_txt(filepath: str) -> str:
    """Read a plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split text into overlapping word-level chunks.
    chunk_size=400 words keeps chunks within LLM context comfortably.
    overlap=80 words prevents cutting off context at boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_file(filepath: str, filename: str) -> List[Dict]:
    """
    Parse a file and return a list of source dicts.
    Each dict: { "content": str, "source": str, "type": "file" }
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        raw = parse_pdf(filepath)
    elif ext in (".txt", ".md"):
        raw = parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")

    chunks = chunk_text(raw)
    return [
        {"content": chunk, "source": filename, "type": "file"}
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search DuckDuckGo and return result snippets.
    Each dict: { "content": str, "source": str, "url": str, "type": "web" }
    """
    if not DDG_AVAILABLE:
        return []

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                content = r.get("body", "").strip()
                if content:
                    results.append({
                        "content": content,
                        "source": r.get("title", "Web Result"),
                        "url": r.get("href", ""),
                        "type": "web"
                    })
    except Exception as e:
        # DuckDuckGo can occasionally rate-limit; degrade gracefully
        print(f"[retriever] Web search error: {e}")

    return results


# ---------------------------------------------------------------------------
# TF-IDF Ranker
# ---------------------------------------------------------------------------

def rank_chunks(query: str, chunks: List[Dict], top_k: int = 6) -> List[Dict]:
    """
    Rank source chunks by TF-IDF cosine similarity to the query.
    Returns top_k most relevant chunks, each with a 'score' field added.

    Why TF-IDF and not embeddings?
    - Zero additional model downloads or API calls
    - Fast enough for <50 chunks
    - Deterministic and explainable
    Trade-off: misses semantic synonyms (e.g. "car" vs "automobile")
    """
    if not chunks:
        return []

    texts = [c["content"] for c in chunks]
    corpus = [query] + texts  # query is index 0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary edge case
        return chunks[:top_k]

    query_vec = tfidf_matrix[0]
    chunk_vecs = tfidf_matrix[1:]
    scores = cosine_similarity(query_vec, chunk_vecs).flatten()

    # Attach scores and sort
    ranked = []
    for i, chunk in enumerate(chunks):
        ranked.append({**chunk, "score": float(scores[i])})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    file_chunks: List[Dict],
    use_web: bool = True,
    top_k: int = 6
) -> List[Dict]:
    """
    Combine file chunks + web results, rank by relevance to query.
    Returns top_k chunks with numbered source labels attached.
    """
    all_chunks = list(file_chunks)  # copy

    if use_web:
        web_results = web_search(query, max_results=5)
        all_chunks.extend(web_results)

    if not all_chunks:
        return []

    ranked = rank_chunks(query, all_chunks, top_k=top_k)

    # Attach numbered source label for citation tracking
    for i, chunk in enumerate(ranked):
        chunk["source_id"] = i + 1  # Source 1, Source 2, ...

    return ranked