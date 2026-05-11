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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class MessageModel(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[MessageModel] = []
    session_id: str | None = None  # required for local LLM mode; ignored by Gemini mode


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
    use_reranker: bool
    llm_provider: str  # 'gemini' or 'local'

class SettingsRequest(BaseModel):
    hybrid_alpha: float | None = None
    use_reranker: bool | None = None



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question. Returns the grounded answer and source citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
        result = pipeline.ask(
            req.question,
            history=history_dicts,
            session_id=req.session_id,
        )

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
        hybrid_alpha=config.HYBRID_ALPHA,
        use_reranker=config.USE_RERANKER,
        llm_provider=config.LLM_PROVIDER,
    )

@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    """Update system settings and persist them to .env file."""
    try:
        from backend.core import config as cfg
        
        # Update memory config
        if req.hybrid_alpha is not None:
            cfg.HYBRID_ALPHA = req.hybrid_alpha
        if req.use_reranker is not None:
            cfg.USE_RERANKER = req.use_reranker
            
        # Update .env file
        env_path = Path(".env")
        lines = []
        updates = {}
        if req.hybrid_alpha is not None:
            updates["HYBRID_ALPHA"] = str(req.hybrid_alpha)
        if req.use_reranker is not None:
            updates["USE_RERANKER"] = "True" if req.use_reranker else "False"

        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        lines.append(line)
                        continue
                    key, _, _ = stripped.partition("=")
                    if key.strip() in updates:
                        lines.append(f"{key.strip()}={updates.pop(key.strip())}\n")
                    else:
                        lines.append(line)
        
        for k, v in updates.items():
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{k}={v}\n")
            
        with open(env_path, "w") as f:
            f.writelines(lines)
            
        return {
            "message": "Settings updated.",
            "hybrid_alpha": cfg.HYBRID_ALPHA,
            "use_reranker": cfg.USE_RERANKER
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear():
    """Wipe all chunks from the vector store."""
    pipeline.store.clear()
    return {"message": "Knowledge base cleared.", "chunk_count": 0}


@app.post("/api/reset-session")
async def reset_session():
    """
    Reset the local LLM KV cache session.
    Call this when the user starts a new chat or refreshes.
    No-op when LLM_PROVIDER=gemini.
    """
    pipeline.reset_local_session()
    return {"message": "Session reset.", "llm_provider": config.LLM_PROVIDER}


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
        if cfg.USE_RERANKER:
            reranked = pipeline.reranker.rerank(req.question, candidates, top_n=cfg.TOP_K_RESULTS)
        else:
            reranked = []


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
