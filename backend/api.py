"""
backend/api.py — FastAPI REST backend for the RAG system

Exposes the RAGPipeline over HTTP so the React frontend can call it.

Endpoints:
  POST /api/chat    — ask a question, get answer + sources
  POST /api/ingest  — upload a file and ingest it into the vector store
  GET  /api/status  — get chunk count
  POST /api/clear   — wipe the collection
"""

import asyncio
import json
import os
import uuid
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    auth_required: bool  # whether ADMIN_TOKEN is configured

class SettingsRequest(BaseModel):
    hybrid_alpha: float | None = None
    use_reranker: bool | None = None
    admin_token: str | None = None  # sent by the frontend for authentication



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
        result = await asyncio.to_thread(
            pipeline.ask,
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


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Event types:
      sources  — JSON array of source citations (sent once before tokens)
      token    — a chunk of the generated answer text
      done     — signals the stream is complete
      error    — an error occurred
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_stream():
        """
        Async SSE generator.

        The pipeline (prepare_context + stream_answer) is entirely synchronous
        and CPU/IO-bound.  We run it in a background thread and bridge tokens
        back to this async generator via an asyncio.Queue, so the event loop is
        never blocked and tokens are forwarded to the client as they arrive.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

        def _run_pipeline():
            try:
                context = pipeline.prepare_context(
                    req.question,
                    history=history_dicts,
                    session_id=req.session_id,
                )
                if context is None:
                    loop.call_soon_threadsafe(queue.put_nowait, ("empty", None))
                    return

                loop.call_soon_threadsafe(queue.put_nowait, ("sources", context))

                for token in pipeline.stream_answer(context):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))

                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

        # Kick off the blocking pipeline work in the default thread-pool executor
        future = loop.run_in_executor(None, _run_pipeline)

        try:
            while True:
                kind, payload = await queue.get()
                if kind == "empty":
                    yield _sse("token", "The knowledge base is empty. Please ingest a document first.")
                    yield _sse("sources", "[]")
                    yield _sse("done", "")
                    break
                elif kind == "sources":
                    yield _sse("sources", json.dumps(payload["sources"], ensure_ascii=False))
                    yield _sse("stages", json.dumps(payload["stages"], ensure_ascii=False))
                elif kind == "token":
                    yield _sse("token", payload)
                elif kind == "done":
                    yield _sse("done", "")
                    break
                elif kind == "error":
                    yield _sse("error", payload)
                    break
        finally:
            await future  # ensure the thread has finished before the response closes

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


def _sse(event: str, data: str) -> str:
    """Format a single SSE message."""
    # SSE data lines must not contain bare newlines; encode them
    escaped = data.replace("\n", "\\n")
    return f"event: {event}\ndata: {escaped}\n\n"


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
        await asyncio.to_thread(pipeline.ingest, tmp_path)
        chunk_count = await asyncio.to_thread(pipeline.store.count)
        return {
            "message": f"'{file.filename}' ingested successfully.",
            "chunk_count": chunk_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Return how many chunks are in the vector store."""
    chunk_count = await asyncio.to_thread(pipeline.store.count)
    return StatusResponse(
        chunk_count=chunk_count,
        hybrid_alpha=config.HYBRID_ALPHA,
        use_reranker=config.USE_RERANKER,
        llm_provider=config.LLM_PROVIDER,
        auth_required=bool(config.ADMIN_TOKEN),
    )

@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    """
    Update system settings and persist them to .env file.

    When ADMIN_TOKEN is set in .env, the request must include a matching
    admin_token field. When ADMIN_TOKEN is blank/unset, access is open
    (convenient for local dev).
    """
    # ── Authentication ──────────────────────────────────────────
    if config.ADMIN_TOKEN and req.admin_token != config.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")

    # ── Input validation ───────────────────────────────────────
    if req.hybrid_alpha is not None:
        if not (0.0 <= req.hybrid_alpha <= 1.0):
            raise HTTPException(
                status_code=422,
                detail="hybrid_alpha must be between 0.0 and 1.0.",
            )

    try:
        from backend.core import config as cfg

        # Update in-memory config
        if req.hybrid_alpha is not None:
            cfg.HYBRID_ALPHA = req.hybrid_alpha
        if req.use_reranker is not None:
            cfg.USE_RERANKER = req.use_reranker

        # Persist to .env file
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
            "use_reranker": cfg.USE_RERANKER,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear():
    """Wipe all chunks from the vector store."""
    await asyncio.to_thread(pipeline.store.clear)
    return {"message": "Knowledge base cleared.", "chunk_count": 0}


@app.post("/api/reset-session")
async def reset_session():
    """
    Reset the local LLM KV cache session.
    Call this when the user starts a new chat or refreshes.
    No-op when LLM_PROVIDER=gemini.
    """
    await asyncio.to_thread(pipeline.reset_local_session)
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
        vectors = await asyncio.to_thread(pipeline.embedder.embed, [req.question], mode="query")
        query_vector = vectors[0]
        # Stage 1 — hybrid search
        candidates = await asyncio.to_thread(
            pipeline.store.query,
            query_vector,
            cfg.RERANK_CANDIDATES,
            req.question,
        )
        # Stage 2 — rerank
        if cfg.USE_RERANKER:
            reranked = await asyncio.to_thread(
                pipeline.reranker.rerank,
                req.question,
                candidates,
                cfg.TOP_K_RESULTS,
            )
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
