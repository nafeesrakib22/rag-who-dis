# Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs)
by giving them access to external, up-to-date knowledge at query time.

A standard LLM is trained on a fixed dataset with a knowledge cutoff date. It cannot know about:
- Your private documents
- Events after its training cutoff
- Proprietary internal data

RAG solves this by retrieving relevant information from a knowledge base and injecting it
into the LLM's prompt as context. The model then generates an answer grounded in that context.

## How RAG Works — Step by Step

### Ingestion Pipeline (offline / one-time)

```
Document (PDF/Markdown)
    → Load          Extract raw text page by page
    → Chunk         Split into small overlapping windows (~500 characters)
    → Embed         Convert each chunk to a dense vector (384 floats)
    → Store         Save vectors + metadata in ChromaDB
```

### Query Pipeline (real-time)

```
User Question
    → Embed         Same model as ingestion — must match!
    → Retrieve      ChromaDB: find top-5 most similar chunks (cosine similarity)
    → Prompt        Build context-injected prompt with retrieved chunks
    → Generate      Gemini API: answer grounded in retrieved context
    → Cite          Return answer + source citations
```

## Key Concepts

### Chunking
Long documents are split into small text windows with overlap. Smaller chunks produce
more precise embeddings and better retrieval. Overlap prevents information loss at boundaries.

### Embeddings
Text is converted to high-dimensional vectors (384 floats with `all-MiniLM-L6-v2`).
Semantically similar text produces geometrically close vectors. This enables semantic search —
finding relevant content even when the exact words don't match the query.

### Vector Database (ChromaDB)
Stores chunk embeddings. At query time, computes cosine similarity between the query vector
and all stored vectors. Returns the top-k most similar chunks. Much faster than naive
brute-force search thanks to HNSW indexing.

### Grounded Generation
The LLM is instructed to answer ONLY from the provided context. This reduces hallucination
and makes every answer verifiable through citations.

---

## Technology Stack

| Component | Tool |
|---|---|
| PDF Parsing | `pypdf` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | `chromadb` (local persistent storage) |
| LLM | `google-generativeai` — Gemini 1.5 Flash |
| Interface | Python CLI (`argparse`) |

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
export GEMINI_API_KEY=your_key_here
```

---

## Usage

```bash
# Ingest a PDF
python main.py add data/myfile.pdf

# Ingest a Markdown file
python main.py add data/notes.md

# Ask a question
python main.py ask "What is the main argument of the document?"

# Check knowledge base size
python main.py status

# Clear everything
python main.py clear
```

---

## Learning Path — Test Components in Isolation

These commands let you observe each component individually to understand what it does.

### 1. Understand chunking
```bash
python -c "
from src.chunker import chunk_text
text = 'This is a test sentence. ' * 60
chunks = chunk_text(text, {'source': 'test.txt', 'page': 1})
print(f'Chunks produced: {len(chunks)}')
print(f'Chunk 0 text: {chunks[0][\"text\"][:80]}')
print(f'Chunk 1 text: {chunks[1][\"text\"][:80]}')
print(f'Overlap visible: {chunks[0][\"text\"][-30:]} | {chunks[1][\"text\"][:30]}')
"
```
> You'll see that chunk 1 starts with some of the same text chunk 0 ended with — that's the overlap.

### 2. Understand embeddings
```bash
python -c "
from src.embedder import Embedder
e = Embedder()
vecs = e.embed(['The cat sat on the mat.', 'A kitten rested on a rug.', 'Stock markets fell.'])
print(f'Each text → a vector of {len(vecs[0])} floats')
print(f'First 5 values of vec 0: {[round(x,3) for x in vecs[0][:5]]}')
print(f'First 5 values of vec 1: {[round(x,3) for x in vecs[1][:5]]}')
print('(vec 0 and vec 1 should look very similar since the sentences mean the same thing)')
"
```

### 3. Understand ChromaDB insert + retrieval
```bash
python -c "
import os; os.makedirs('/tmp/chroma_test', exist_ok=True)
from src.embedder import Embedder
from src.vector_store import ChromaStore
e = Embedder()
store = ChromaStore('/tmp/chroma_test')
chunks = [
    {'text': 'Paris is the capital of France.', 'source': 'geo.txt', 'page': 1, 'chunk_index': 0},
    {'text': 'Berlin is the capital of Germany.', 'source': 'geo.txt', 'page': 1, 'chunk_index': 1},
    {'text': 'The Eiffel Tower is in Paris.', 'source': 'geo.txt', 'page': 1, 'chunk_index': 2},
]
embeddings = e.embed([c['text'] for c in chunks])
store.add_chunks(chunks, embeddings)
results = store.query(e.embed(['What is the capital of France?'])[0], n_results=2)
for r in results:
    print(f'  → \"{r[\"text\"]}\" (distance={r[\"distance\"]})')
"
```
