"""
Tests for backend.api — FastAPI endpoints with a mocked RAGPipeline.

These tests do NOT require Weaviate, GPU, or any API keys.
"""

import pytest
from unittest.mock import MagicMock, patch
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Create a test client with a fully mocked pipeline.

    We replace the app's lifespan so it never instantiates the real
    RAGPipeline (which would try to connect to Weaviate and load models).
    """
    mock_pipeline = MagicMock()
    mock_pipeline.store.count.return_value = 42

    # Replace the lifespan to avoid loading real models / connecting to Weaviate
    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    import backend.api as api_module
    original_lifespan = api_module.app.router.lifespan_context
    api_module.app.router.lifespan_context = _noop_lifespan
    api_module.pipeline = mock_pipeline

    try:
        with TestClient(api_module.app) as c:
            yield c, mock_pipeline
    finally:
        api_module.app.router.lifespan_context = original_lifespan


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------

class TestStatus:

    def test_returns_chunk_count(self, client):
        c, mock = client
        res = c.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["chunk_count"] == 42
        assert "hybrid_alpha" in data
        assert "use_reranker" in data
        assert "llm_provider" in data


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

class TestChat:

    def test_empty_question_returns_400(self, client):
        c, _ = client
        res = c.post("/api/chat", json={"question": "", "history": []})
        assert res.status_code == 400

    def test_valid_question_returns_answer(self, client):
        c, mock = client
        mock.ask.return_value = {
            "answer": "The answer is 42.",
            "sources": [
                {
                    "n": 1, "source": "doc.pdf", "page": 1, "chunk_index": 0,
                    "hybrid_score": 0.9, "rerank_score": 8.5,
                    "preview": "some text...", "text": "some text content",
                }
            ],
            "stages": {
                "initial": [],
                "reranked": [],
            },
        }
        res = c.post("/api/chat", json={"question": "What is the answer?"})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "The answer is 42."
        assert len(data["sources"]) == 1
        assert "message_id" in data

    def test_pipeline_error_returns_500(self, client):
        c, mock = client
        mock.ask.side_effect = RuntimeError("LLM crashed")
        res = c.post("/api/chat", json={"question": "Will this fail?"})
        assert res.status_code == 500


# ---------------------------------------------------------------------------
# POST /api/chat/stream
# ---------------------------------------------------------------------------

class TestChatStream:

    def test_empty_question_returns_400(self, client):
        c, _ = client
        res = c.post("/api/chat/stream", json={"question": "  "})
        assert res.status_code == 400

    def test_streaming_returns_sse_events(self, client):
        c, mock = client
        mock.prepare_context.return_value = {
            "question": "test",
            "session_id": None,
            "retrieved_chunks": [{"text": "ctx", "source": "f.pdf", "page": 1, "chunk_index": 0}],
            "candidates": [],
            "summary": "",
            "selected_past": [],
            "recent_history": [],
            "sources": [{"n": 1, "source": "f.pdf", "page": 1, "chunk_index": 0,
                         "hybrid_score": 0.9, "preview": "ctx", "text": "ctx"}],
            "stages": {"initial": [], "reranked": []},
        }
        mock.stream_answer.return_value = iter(["Hello ", "world!"])

        res = c.post("/api/chat/stream", json={"question": "test"})
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        body = res.text
        assert "event: sources" in body
        assert "event: token" in body
        assert "event: done" in body

    def test_empty_kb_streams_fallback(self, client):
        c, mock = client
        mock.prepare_context.return_value = None

        res = c.post("/api/chat/stream", json={"question": "test"})
        assert res.status_code == 200
        assert "knowledge base is empty" in res.text.lower()


# ---------------------------------------------------------------------------
# POST /api/settings — auth + validation
# ---------------------------------------------------------------------------

class TestSettings:

    def test_open_access_when_no_token_configured(self, client):
        """When ADMIN_TOKEN is empty, settings should be accessible without a token."""
        c, _ = client
        import backend.core.config as cfg
        original = cfg.ADMIN_TOKEN
        cfg.ADMIN_TOKEN = ""  # no token required
        try:
            res = c.post("/api/settings", json={"hybrid_alpha": 0.5})
            assert res.status_code == 200
        finally:
            cfg.ADMIN_TOKEN = original

    def test_rejects_without_token_when_required(self, client):
        """When ADMIN_TOKEN is set, requests without a matching token get 401."""
        c, _ = client
        import backend.core.config as cfg
        original = cfg.ADMIN_TOKEN
        cfg.ADMIN_TOKEN = "secret123"
        try:
            res = c.post("/api/settings", json={"hybrid_alpha": 0.5})
            assert res.status_code == 401
        finally:
            cfg.ADMIN_TOKEN = original

    def test_accepts_correct_token(self, client):
        c, _ = client
        import backend.core.config as cfg
        original = cfg.ADMIN_TOKEN
        cfg.ADMIN_TOKEN = "secret123"
        try:
            res = c.post("/api/settings", json={
                "hybrid_alpha": 0.5,
                "admin_token": "secret123",
            })
            assert res.status_code == 200
        finally:
            cfg.ADMIN_TOKEN = original

    def test_rejects_invalid_alpha(self, client):
        c, _ = client
        import backend.core.config as cfg
        original = cfg.ADMIN_TOKEN
        cfg.ADMIN_TOKEN = ""
        try:
            res = c.post("/api/settings", json={"hybrid_alpha": 1.5})
            assert res.status_code == 422
            res = c.post("/api/settings", json={"hybrid_alpha": -0.1})
            assert res.status_code == 422
        finally:
            cfg.ADMIN_TOKEN = original

    def test_valid_alpha_range_accepted(self, client):
        c, _ = client
        import backend.core.config as cfg
        original = cfg.ADMIN_TOKEN
        cfg.ADMIN_TOKEN = ""
        try:
            for val in [0.0, 0.5, 1.0]:
                res = c.post("/api/settings", json={"hybrid_alpha": val})
                assert res.status_code == 200
        finally:
            cfg.ADMIN_TOKEN = original


# ---------------------------------------------------------------------------
# POST /api/clear
# ---------------------------------------------------------------------------

class TestClear:

    def test_clear_returns_zero(self, client):
        c, mock = client
        res = c.post("/api/clear")
        assert res.status_code == 200
        data = res.json()
        assert data["chunk_count"] == 0
        mock.store.clear.assert_called_once()
