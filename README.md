# RAG System — Retrieval-Augmented Generation

A full-stack RAG system that lets you ingest PDF, Markdown, JSON, and text documents
and ask questions answered with citations — grounded entirely in your own documents.

## Stack

| Component | Tool |
|---|---|
| Document Parsing | `pymupdf` (with OCR fallback via EasyOCR or Gemini Vision) |
| Chunking | Semantic chunking (`nltk` + cosine similarity) |
| Embeddings | Ollama (local) — `embeddinggemma` model |
| Vector DB | Weaviate (Docker) — hybrid BM25 + vector search |
| Re-ranking | `BAAI/bge-reranker-v2-m3` cross-encoder |
| LLM | Google Gemini API (`gemini-2.5-flash`) |
| Backend API | FastAPI |
| Frontend | React + Vite |

---

## Setup (New Machine)

### Prerequisites

Make sure the following are installed on your machine:
- **Python 3.10+**
- **Docker** (for Weaviate)
- **Ollama** — [install here](https://ollama.com/download)
- **Node.js 18+** (for the frontend)

### 1. Clone & configure environment

```bash
git clone <your-repo-url>
cd RAG

# Create your .env from the template
cp .env.example .env
```

Open `.env` and fill in your values:
- **`GOOGLE_API_KEY`** — get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- All other values can be left as their defaults for a local setup.

### 2. Start Weaviate (vector database)

```bash
docker compose up -d
```

### 3. Pull the embedding model via Ollama

```bash
ollama pull embeddinggemma
```

Make sure Ollama is running in the background (`ollama serve`).

### 4. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Start the backend

```bash
uvicorn backend.api:app --reload
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at **http://localhost:5173**.

---

## CLI Usage

You can also use the system entirely from the command line (no frontend needed):

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
    → Embed         Dense vectors via Ollama (embeddinggemma)
    → Store         Weaviate: vectors + BM25 index
```

### Query Pipeline
```
User Question
    → Embed         Same model as ingestion
    → Hybrid Search Weaviate: BM25 + vector (top 20 candidates)
    → Re-rank       BAAI/bge-reranker-v2-m3 cross-encoder → top 5
    → Prompt        Context-injected prompt with citations
    → Generate      Gemini API → grounded answer
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
