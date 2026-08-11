# ResearchLens

Evidence-grounded research agent. Upload PDF/TXT/MD documents (and optionally
pull in live web results), ask a question, and get back an answer where every
claim is tied to a numbered source — with an explicit callout for anything
the sources don't cover.

## How it works

1. **Retrieve** (`retriever.py`) — uploaded files are chunked into ~400-word
   overlapping windows; if web search is enabled, DuckDuckGo results are
   pulled in too. Everything is ranked against the question using TF-IDF +
   cosine similarity (no embeddings, no extra API calls, fully deterministic).
2. **Cite** (`citation.py`) — the top-ranked chunks are formatted into a
   numbered `[Source N]` context block for the prompt, and later reshaped
   into a clean source list for the UI.
3. **Answer** (`agent.py`) — the question + context block go to Groq
   (`llama-3.3-70b-versatile` by default) with a system prompt that requires
   an inline `[Source N]` citation after every claim and a `⚠️ Gaps` section
   for anything unsupported.
4. **Serve** (`main.py`) — a FastAPI backend exposes upload/ask/sources
   endpoints and serves the static frontend (`index.html` + `style.css`).

## Project structure

```
ResearchLens/
│
├── .env.example              ← copy to .env, add your GROQ_API_KEY              
├── README.md               
│
├── backend/
│   ├── main.py               ← FastAPI app (endpoints: /upload /ask /sources)
│   ├── agent.py              ← Groq LLM call + citation system prompt
│   ├── retriever.py          ← PDF/TXT parser + DuckDuckGo search + TF-IDF ranker
│   ├── citation.py           ← Context block builder + source formatter
│   └── requirements.txt      ← All pinned dependencies
│
├── frontend/
│   ├── index.html            ← Single-file UI (upload, ask, view cited answer)
│   ├── logo.svg
│   └── style.css          
│
│
└── sample_outputs/
     └── results.json          ← Auto-generated when you ask questions (submission evidence)-gitignored

```

## Setup

## 1. Clone the Repository

```bash
git clone https://github.com/fathimarfa/ResearchLens
cd ResearchLens

```

2. Create and Activate a Virtual Environment

Windows
```bash
python -m venv venv
venv\Scripts\activate

```

macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate

```
After activation, your terminal prompt should show (.venv) indicating the virtual environment is active.

3. Install dependencies

Make sure the virtual environment is activated, after activation, your terminal prompt should show (venv) indicating the virtual environment is active.then run:

pip install -r requirements.txt

4. Create your environment file

Copy the example environment file:

Windows CMD
```bash
cp ../.env.example ../.env
```
macOS / Linux
```bash
cp .env.example .env
```

Then open .env and add your Groq API key:

GROQ_API_KEY=your_groq_api_key_here

Never commit .env to GitHub. Keep your actual API key private.

▶️ Run the application

Start the FastAPI server with:

```bash
cd backend #because main.py is inside backend
python -m uvicorn main:app --reload --port 8000
```
You should see something similar to:

INFO:     Uvicorn running on http://127.0.0.1:8000

Open the application in your browser:

http://127.0.0.1:8000
OR
Open `http://localhost:8000` — the UI is served directly from the API.

## API

| Method | Path       | Description                                      |
|--------|------------|---------------------------------------------------|
| GET    | `/health`  | Health check + count of loaded documents          |
| POST   | `/upload`  | Upload one or more PDF/TXT/MD files (multipart)    |
| GET    | `/sources` | List currently loaded documents                    |
| DELETE | `/sources` | Clear all loaded documents from memory              |
| POST   | `/ask`     | Ask a question. Body: `{question, use_web, top_k}` |

`POST /ask` response:

```json
{
  "question": "...",
  "answer": "... [Source 1] ... [Source 2] ...",
  "sources": [
    {"source_id": 1, "type": "file", "label": "paper.pdf", "score": 0.83}
  ],
  "chunks_used": 6,
  "documents_loaded": 2,
  "web_used": false
}
```

Every call to `/ask` is also appended to `sample_outputs/results.json` for
easy review of past Q&A.

## Notes & limitations

- **Storage is in-memory.** Uploaded documents are lost on server restart;
  there's no persistence layer.
- **TF-IDF, not embeddings.** Fast and dependency-light, but it matches on
  literal terms — it won't connect "car" with "automobile". Swap in an
  embedding model in `retriever.py` if you need semantic recall.
- **Web search degrades gracefully.** If `duckduckgo_search`(now ddgs) isn't installed
  or DuckDuckGo(ddgs) rate-limits the request, `retrieve()` just falls back to
  file-only results instead of failing the request.
- **CORS is wide open** (`allow_origins=["*"]`) for local development —
  tighten this before deploying anywhere public.