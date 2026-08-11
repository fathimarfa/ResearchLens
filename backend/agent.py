"""
agent.py
Core ResearchLens agent.
Takes a question + retrieved context, calls Groq LLM,
returns a cited answer and flags unsupported claims.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict

from retriever import retrieve
from citation import build_context_block, format_sources_for_response

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are ResearchLens, an evidence-grounded research assistant.

Your job:
1. Answer the user's question using ONLY the numbered sources provided in the context.
2. After every factual claim, add an inline citation like [Source 1] or [Source 2, Source 3].
3. If a part of the question CANNOT be answered from the provided sources, explicitly say:
   "⚠️ Not found in sources: <what is missing>"
4. Do NOT make up facts. Do NOT use knowledge outside the provided sources.
5. Keep your answer clear, structured, and concise.

Format your response as:
- A direct answer with inline [Source N] citations after each claim.
- A final "⚠️ Gaps" section listing anything the sources did not cover (if any).
"""


def run_agent(
    question: str,
    file_chunks: List[Dict],
    use_web: bool = True,
    top_k: int = 6,
    model: str = "llama-3.3-70b-versatile"
) -> Dict:
    """
    Full agent pipeline:
      1. Retrieve relevant chunks (files + optional web)
      2. Build context block with source labels
      3. Call Groq LLM with system prompt + context + question
      4. Return answer + sources list

    Returns:
        {
            "answer": str,
            "sources": List[Dict],
            "question": str,
            "chunks_used": int
        }
    """
    # Step 1: Retrieve
    ranked_chunks = retrieve(question, file_chunks, use_web=use_web, top_k=top_k)

    if not ranked_chunks:
        return {
            "question": question,
            "answer": "⚠️ No sources available. Please upload documents or enable web search.",
            "sources": [],
            "chunks_used": 0
        }

    # Step 2: Build context
    context_block = build_context_block(ranked_chunks)

    # Step 3: Call LLM
    user_message = f"""Context (use only these sources):
---
{context_block}
---

Question: {question}

Answer with inline citations [Source N] for every claim."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,   # Low temp = more faithful to sources, less hallucination
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    # Step 4: Format output
    sources = format_sources_for_response(ranked_chunks)
    web_used = any(chunk.get("type") == "web" for chunk in ranked_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(ranked_chunks),
        "web_used": web_used
    }