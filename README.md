# RAG System — Retrieval-Augmented Generation

A full-stack RAG system that lets you ingest PDF, Markdown, and text documents
and ask questions answered with citations — grounded entirely in your own documents.

## Highlights

- Full-stack RAG app with FastAPI, React, Weaviate, hybrid search, re-ranking, and citation-grounded answers.
- **Streaming responses** via Server-Sent Events — tokens appear in real time as the LLM generates.
- **Streaming ingest progress** via SSE — `POST /api/ingest` returns a `job_id` immediately; connect to `GET /api/ingest/{job_id}/progress` to receive live stage events (loading → chunking → embedding → storing).
- Supports PDF, Markdown, and TXT ingestion with OCR fallback for scanned or corrupted PDFs.
- Structured logging throughout with `LOG_LEVEL` env var control (DEBUG/INFO/WARNING/ERROR).
- Includes a Retrieval Trace view to inspect hybrid-search candidates, re-ranked chunks, and source evidence.
- Supports both Gemini API and local Gemma inference via LiteRT-LM.
- Token-gated admin endpoints (`/api/settings`, `/api/clear`) using `Authorization: Bearer` header with timing-safe comparison (optional `ADMIN_TOKEN`).
- 45-test suite covering chunking, loading, API endpoints, auth, and config.

## Stack

| Component | Tool |
|---|---|
| Document Parsing | `pymupdf` (with OCR fallback via EasyOCR or Gemini Vision) |
| Chunking | Semantic chunking (`nltk` + cosine similarity) |
| Embeddings | `google/embeddinggemma-300m` via `sentence-transformers` (in-process) |
| Vector DB | Weaviate (Docker) — hybrid BM25 + vector search |
| Re-ranking | `BAAI/bge-reranker-v2-m3` cross-encoder |
| LLM (cloud) | Google Gemini API (`gemini-2.5-flash`) |
| LLM (local) | `gemma-4-E2B-it` via `litert-lm` (KV cache multi-turn) |
| Backend API | FastAPI |
| Frontend | React + Vite |

---
## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Ask a question — returns grounded answer with citations. |
| `POST` | `/api/chat/stream` | Stream answer tokens via SSE (`sources`, `token`, `done` events). |
| `POST` | `/api/ingest` | Upload a document — starts background ingestion, returns `job_id`. |
| `GET` | `/api/ingest/{job_id}/progress` | Stream ingest progress via SSE (`progress`, `done`, `error` events). |
| `GET` | `/api/status` | Current chunk count, ingested sources, and runtime settings. |
| `POST` | `/api/settings` | Update retrieval settings — requires `Authorization: Bearer` when `ADMIN_TOKEN` is set. |
| `POST` | `/api/clear` | Wipe the knowledge base — requires `Authorization: Bearer` when `ADMIN_TOKEN` is set. |
| `POST` | `/api/retrieve` | Return hybrid + reranked retrieval stages without calling the LLM. |
| `POST` | `/api/reset-session` | Reset local KV-cache chat session (no-op in Gemini mode). |

---

## Setup (New Machine)

### Prerequisites

- **Python 3.10+**
- **Docker** (for Weaviate)
- **Node.js 18+** (for the frontend)
- **HuggingFace account** with access to `google/embeddinggemma-300m` (free, but gated — you must visit [the model page](https://huggingface.co/google/embeddinggemma-300m) and click **"Agree and access repository"** before your `HF_TOKEN` will allow the download)

### 1. Clone & configure environment

```bash
git clone https://github.com/nafeesrakib22/rag-who-dis.git
cd rag-who-dis

cp .env.example .env
```

Open `.env` and fill in:
- **`GOOGLE_API_KEY`** — from [Google AI Studio](https://aistudio.google.com/app/apikey) (required for Gemini mode and Gemini Vision OCR)
- **`HF_TOKEN`** — from [HuggingFace settings](https://huggingface.co/settings/tokens) (required for the embedding model download — see Prerequisites above)
- **`LLM_PROVIDER`** — `gemini` (default) or `local` (see Local LLM section below)
- **`ADMIN_TOKEN`** — (optional) protects the `/api/settings` and `/api/clear` endpoints; see [Security](#security) below
- **`LOG_LEVEL`** — (optional) controls log verbosity: `DEBUG`, `INFO` (default), `WARNING`, or `ERROR`

### 2. Start Weaviate (vector database)

```bash
docker compose up weaviate -d
```

### 3. Install Python dependencies

```bash
python -m venv venv

# Activate the virtual environment:
source venv/bin/activate          # Linux / macOS / Git Bash on Windows
.\venv\Scripts\Activate.ps1       # Windows PowerShell
venv\Scripts\activate.bat         # Windows cmd

pip install -r requirements.txt
```

For reproducible installs with exact pinned versions:

```bash
pip install -r requirements.lock
```

> **First run:** `sentence-transformers` will download `google/embeddinggemma-300m`
> (~1.2 GB) and `BAAI/bge-reranker-v2-m3` (~2.3 GB) on first backend start.
> They are cached in `~/.cache/huggingface/` and reused on subsequent runs.

### 4. Start the backend

Make sure your virtual environment is activated (see step 3), then:

```bash
uvicorn backend.api:app --reload --reload-exclude weaviate_db
```

> **Note (Linux/WSL2):** The `weaviate_db` folder is owned by root (written by Docker).
> The `--reload-exclude weaviate_db` flag prevents a permission error in the file watcher.
> Alternatively, drop `--reload` entirely if you don't need auto-reload.

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at **http://localhost:5173**.

---

## Local LLM Setup (Optional)

To run inference entirely on-device without any API key, use the `gemma-4-E2B-it`
model via `litert-lm`.

### 1. Install litert-lm

`litert-lm` is included in `requirements.txt` but commented out. Uncomment it:

```
# litert-lm  ← remove the leading #
```

Then re-run (with your virtual environment activated):

```bash
pip install -r requirements.txt
```

### 2. Download the model

The model is gated — visit [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) on HuggingFace and accept the terms before running this command:

```bash
litert-lm import --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm gemma-e2b
```

This downloads and registers the model locally under the name `gemma-e2b`.
By default it is stored at `~/.litert-lm/models/gemma-e2b`.

### 3. Enable local mode in `.env`

```ini
LLM_PROVIDER=local
LOCAL_MODEL_PATH=/home/<your-username>/.litert-lm/models/gemma-e2b/model.litertlm
```

Example:

```ini
LOCAL_MODEL_PATH=/home/yourname/.litert-lm/models/gemma-e2b/model.litertlm
```

### How local mode works

- **History** is managed by the model's KV cache. Each conversation session is kept
  alive for the lifetime of the backend process — subsequent turns only process new
  tokens, making multi-turn responses progressively faster.
- **New Chat** (sidebar button) or a page refresh resets the session and starts fresh.
- **Query condensation** (`_condense_query`) is skipped — the model already has full
  context via its KV cache, so follow-up questions resolve correctly without reformulation.

> **Linux/WSL2 dependency:** `litert-lm` requires `libGLESv2` even on CPU-only mode.
> Install it with: `sudo apt-get install -y libgles2`

### Docker with local mode

The model directory is bind-mounted into the container via the `LOCAL_MODELS_DIR`
environment variable. Add it to your `.env`:

```ini
LOCAL_MODELS_DIR=/home/yourname/.litert-lm/models
```

`docker-compose.yml` reads this and mounts it as `/models` inside the container.
If `LOCAL_MODELS_DIR` is not set, the compose file falls back to
`/home/<your-username>/.litert-lm/models` — edit that default in `docker-compose.yml`
if you prefer not to use the env var.

---

## Docker (Full Stack)

The build bakes the embedding + reranker models into the image layer so there's no
download at runtime. `HF_TOKEN` must be passed as a build argument:

```bash
docker compose build --build-arg HF_TOKEN=your_hf_token
docker compose up -d
```

> The models are baked into the image at build time. Subsequent `docker compose up -d`
> runs (without `--build`) reuse the cached layer and start instantly.

---

## CLI Usage

```bash
# Ingest a document
python -m backend.main add data/myfile.pdf
python -m backend.main add data/notes.md

# Ask a question
python -m backend.main ask "What is the main argument of the document?"

# Check how many chunks are stored
python -m backend.main status

# Clear the entire knowledge base
python -m backend.main clear
```

---

## How It Works

### Ingestion Pipeline
```
POST /api/ingest  →  duplicate check  →  save temp file  →  return job_id
                                                                    ↓
GET /api/ingest/{job_id}/progress  ←←←  SSE progress stream  ←←←←←←

Background thread:
    Document (PDF / MD / TXT)
        → Load    [progress: loading]    Extract text (with OCR fallback)
        → Chunk   [progress: chunking]   Semantic chunking via sentence embeddings
        → Embed   [progress: embedding]  Dense vectors via embeddinggemma-300m
        → Store   [progress: storing]    Weaviate: vectors + BM25 index
        → Done    [event: done]
```

### Query Pipeline (Gemini mode)
```
User Question
    → Condense      Reformulate follow-up questions using conversation history
    → Embed         Same model as ingestion
    → Hybrid Search Weaviate: BM25 + vector (top 20 candidates)
    → Re-rank       bge-reranker-v2-m3 cross-encoder → top 5
    → Prompt        Context + history injected as text
    → Generate      Gemini API → grounded answer
```

### Query Pipeline (Local mode)
```
User Question
    → Embed         Same model as ingestion
    → Hybrid Search Weaviate: BM25 + vector (top 20 candidates)
    → Re-rank       bge-reranker-v2-m3 cross-encoder → top 5
    → Chat          Retrieved context injected into KV cache session
    → Generate      gemma-4-E2B-it (litert-lm) → grounded answer
                    (Prior turns already cached — only new tokens processed)
```

### OCR Strategy

Controlled by `OCR_STRATEGY` in `.env`:
- **`local`** — Uses EasyOCR on your machine (free, supports Bangla + English, slower)
- **`llm`** — Uses Gemini Vision (higher quality, consumes `GOOGLE_API_KEY` quota)

---

## Security

The `/api/settings` and `/api/clear` endpoints are admin-only — they modify
retrieval parameters and wipe the knowledge base respectively.
To protect them, set `ADMIN_TOKEN` in `.env`.

### Admin token authentication

```ini
ADMIN_TOKEN=my-secret-token-here
```

When set:
- Both `/api/settings` and `/api/clear` require the token in the standard HTTP auth header:
  ```
  Authorization: Bearer my-secret-token-here
  ```
- Requests without a valid token receive a `401 Unauthorized` response.
- Token comparison uses `secrets.compare_digest()` (constant-time) to prevent timing attacks.
- The frontend stores the token in `localStorage` and attaches the header automatically.

When `ADMIN_TOKEN` is blank or absent (the default), both endpoints are open —
convenient for local development with no friction.

### Input validation

Regardless of authentication, `/api/settings` enforces:
- `hybrid_alpha` must be between `0.0` and `1.0` (422 otherwise)
- `use_reranker` must be a boolean

---

## Testing

The project includes a 45-test suite that runs without requiring a GPU, API keys,
or a live Weaviate instance (API tests use a mocked pipeline).
For end-to-end manual checks against a real vector DB, start Weaviate separately.

```bash

# Run the full suite
pytest -v
```

Test coverage:
- **Chunker** — fixed-size splitting, semantic splitting, metadata passthrough, chunk size limits
- **Loader** — Bangla corruption detection, text/markdown loading, file dispatch, error handling
- **API** — chat, streaming SSE, ingest, status, clear, settings auth + validation
- **Config** — attribute presence, value ranges, LLM provider validation
- **Weaviate store** — deterministic UUID generation

---

## Utility Scripts

```bash
# Inspect all stored chunks in Weaviate
python inspect_chunks.py

# Preview text extraction from a PDF (without ingesting)
python peek_text.py data/myfile.pdf
```
