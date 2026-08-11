"""
citation.py
Formats retrieved chunks into a context block for the LLM prompt,
and parses the LLM's response to extract clean citations for the frontend.
"""

from typing import List, Dict


def build_context_block(chunks: List[Dict]) -> str:
    """
    Convert ranked chunks into a numbered context string injected into the prompt.

    Format:
        [Source 1] (filename.pdf)
        <chunk text>

        [Source 2] (Web: Article Title — https://...)
        <chunk text>
    """
    lines = []
    for chunk in chunks:
        sid = chunk["source_id"]
        if chunk["type"] == "web":
            label = f"Web: {chunk['source']} — {chunk.get('url', '')}"
        else:
            label = chunk["source"]

        lines.append(f"[Source {sid}] ({label})")
        lines.append(chunk["content"].strip())
        lines.append("")  # blank line between sources

    return "\n".join(lines)


def format_sources_for_response(chunks: List[Dict]) -> List[Dict]:
    """
    Build the clean source list returned to the frontend alongside the answer.
    Each item: { source_id, label, url (if web), type }
    """
    sources = []
    for chunk in chunks:
        sid = chunk["source_id"]
        entry = {
            "source_id": sid,
            "type": chunk["type"],
            "label": chunk["source"],
            "score": round(chunk.get("score", 0.0), 4),
        }
        if chunk["type"] == "web":
            entry["url"] = chunk.get("url", "")
        sources.append(entry)
    return sources