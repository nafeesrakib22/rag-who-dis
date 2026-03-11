"""
backend/api.py — FastAPI REST backend for the RAG system

Exposes the RAGPipeline over HTTP so the React frontend can call it.

Endpoints:
  POST /api/chat    — ask a question, get answer + sources
  POST /api/ingest  — upload a file and ingest it into the vector store
  GET  /api/status  — get chunk count
  POST /api/clear   — wipe the collection
"""

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core import config
from backend.core.rag import RAGPipeline


# ---------------------------------------------------------------------------
# App lifecycle — load the heavy models once at startup
# ---------------------------------------------------------------------------

pipeline: RAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline (models + DB connection) once at server startup."""
    global pipeline
    print("[api] Starting up — loading RAG pipeline...")
    pipeline = RAGPipeline()
    print("[api] RAG pipeline ready.")
    yield
    # Shutdown
    if pipeline and hasattr(pipeline.store, "close"):
        pipeline.store.close()
    print("[api] Shutdown complete.")


app = FastAPI(title="RAG API", lifespan=lifespan)

# Allow the Vite dev server (port 5173) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str

class SourceModel(BaseModel):
    n: int
    source: str
    page: int
    chunk_index: int
    hybrid_score: float | None = None
    rerank_score: float | None = None
    preview: str
    text: str | None = None

class ChatResponse(BaseModel):
    message_id: str
    answer: str
    sources: list[SourceModel]
    stages: dict[str, list[SourceModel]] | None = None

class StatusResponse(BaseModel):
    chunk_count: int
    hybrid_alpha: float

class SettingsRequest(BaseModel):
    hybrid_alpha: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question. Returns the grounded answer and source citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = pipeline.ask(req.question)
        return ChatResponse(
            message_id=str(uuid.uuid4()),
            answer=result["answer"],
            sources=[SourceModel(**s) for s in result["sources"]],
            stages={
                stage: [SourceModel(**s) for s in chunks]
                for stage, chunks in result.get("stages", {}).items()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    allowed = {".pdf", ".md", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed)}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pipeline.ingest(tmp_path)
        return {
            "message": f"'{file.filename}' ingested successfully.",
            "chunk_count": pipeline.store.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Return how many chunks are in the vector store."""
    return StatusResponse(
        chunk_count=pipeline.store.count(),
        hybrid_alpha=config.HYBRID_ALPHA
    )
@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    """Update system settings and persist them to .env file."""
    try:
        # Update in memory config
        from backend.core import config as cfg
        cfg.HYBRID_ALPHA = req.hybrid_alpha
        
        # Update .env file
        env_path = Path(".env")
        lines = []
        found = False
        
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("HYBRID_ALPHA="):
                        lines.append(f"HYBRID_ALPHA={req.hybrid_alpha}\n")
                        found = True
                    else:
                        lines.append(line)
        
        if not found:
            # Ensure newline if file isn't empty and doesn't end with one
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"HYBRID_ALPHA={req.hybrid_alpha}\n")
            
        with open(env_path, "w") as f:
            f.writelines(lines)
            
        print(f"[api] Updated HYBRID_ALPHA to {req.hybrid_alpha} persistently.")
        return {"message": "Settings updated.", "hybrid_alpha": cfg.HYBRID_ALPHA}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear():
    """Wipe all chunks from the vector store."""
    pipeline.store.clear()
    return {"message": "Knowledge base cleared.", "chunk_count": 0}


class RetrieveRequest(BaseModel):
    question: str

@app.post("/api/retrieve")
async def retrieve(req: RetrieveRequest):
    """Run hybrid search + reranking, return both stages without calling the LLM."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        from backend.core import config as cfg
        # Embed
        query_vector = pipeline.embedder.embed([req.question], mode="query")[0]
        # Stage 1 — hybrid search
        candidates = pipeline.store.query(
            query_vector,
            n_results=cfg.RERANK_CANDIDATES,
            query_text=req.question,
        )
        # Stage 2 — rerank
        reranked = pipeline.reranker.rerank(req.question, candidates, top_n=cfg.TOP_K_RESULTS)

        def fmt(chunks):
            return [
                {
                    "n": i + 1,
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                    "hybrid_score": c.get("hybrid_score"),
                    "rerank_score": c.get("rerank_score"),
                    "text": c["text"],
                    "preview": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"],
                }
                for i, c in enumerate(chunks)
            ]

        return {
            "stages": {
                "initial": fmt(candidates),
                "reranked": fmt(reranked),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
