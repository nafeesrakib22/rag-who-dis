# RAG System — Retrieval-Augmented Generation

A full-stack RAG system that lets you ingest PDF, Markdown, JSON, and text documents
and ask questions answered with citations — grounded entirely in your own documents.

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

## Setup (New Machine)

### Prerequisites

- **Python 3.10+**
- **Docker** (for Weaviate)
- **Node.js 18+** (for the frontend)
- **HuggingFace account** with access to `google/embeddinggemma-300m` (free, gated)

### 1. Clone & configure environment

```bash
git clone <your-repo-url>
cd rag-who-dis

cp .env.example .env
```

Open `.env` and fill in:
- **`GOOGLE_API_KEY`** — from [Google AI Studio](https://aistudio.google.com/app/apikey) (required for Gemini mode and Gemini Vision OCR)
- **`HF_TOKEN`** — from [HuggingFace settings](https://huggingface.co/settings/tokens) (required for the embedding model download)
- **`LLM_PROVIDER`** — `gemini` (default) or `local` (see Local LLM section below)

### 2. Start Weaviate (vector database)

```bash
docker compose up weaviate -d
```

### 3. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **First run:** `sentence-transformers` will download `google/embeddinggemma-300m`
> (~600 MB) and `BAAI/bge-reranker-v2-m3` (~550 MB) on first backend start.
> They are cached in `~/.cache/huggingface/` and reused on subsequent runs.

### 4. Start the backend

```bash
uvicorn backend.api:app --reload
```

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

```bash
pip install litert-lm
```

### 2. Download the model

```bash
litert-lm import --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm gemma-e2b
```

This downloads and registers the model locally under the name `gemma-e2b`.
By default it is stored at `~/.litert-lm/models/gemma-e2b`.

### 3. Enable local mode in `.env`

```ini
LLM_PROVIDER=local
LOCAL_MODEL_PATH=/home/<your-username>/.litert-lm/models/gemma-e2b
```

### How local mode works

- **History** is managed by the model's KV cache. Each conversation session is kept
  alive for the lifetime of the backend process — subsequent turns only process new
  tokens, making multi-turn responses progressively faster.
- **New Chat** (sidebar button) or a page refresh resets the session and starts fresh.
- **Query condensation** (`_condense_query`) is skipped — the model already has full
  context via its KV cache, so follow-up questions resolve correctly without reformulation.

### Docker with local mode

The model directory is bind-mounted into the container automatically.
Update `docker-compose.yml` if your model path differs from the default:

```yaml
volumes:
  - /home/<your-username>/.litert-lm/models:/models:ro
```

---

## Docker (Full Stack)

```bash
docker compose up --build -d
```

The build bakes the embedding + reranker models into the image layer (no download at runtime).
Pass your HuggingFace token as a build argument:

```bash
docker compose build --build-arg HF_TOKEN=your_hf_token
docker compose up -d
```

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
Document (PDF / MD / TXT / JSON)
    → Load          Extract text (with OCR fallback for scanned PDFs)
    → Chunk         Semantic chunking using sentence embeddings
    → Embed         Dense vectors via embeddinggemma-300m (in-process)
    → Store         Weaviate: vectors + BM25 index
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

## Utility Scripts

```bash
# Inspect all stored chunks in Weaviate
python inspect_chunks.py

# Preview text extraction from a PDF (without ingesting)
python peek_text.py data/myfile.pdf
```
