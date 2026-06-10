"""
Tests for backend.core.weaviate_store (UUID generation) and backend.core.config.
"""

from backend.core.weaviate_store import WeaviateStore


# ---------------------------------------------------------------------------
# Deterministic UUID generation
# ---------------------------------------------------------------------------

class TestMakeUuid:

    def test_deterministic(self):
        """Same inputs should always produce the same UUID."""
        a = WeaviateStore._make_uuid("doc.pdf", 1, 0)
        b = WeaviateStore._make_uuid("doc.pdf", 1, 0)
        assert a == b

    def test_different_source_different_uuid(self):
        a = WeaviateStore._make_uuid("doc.pdf", 1, 0)
        b = WeaviateStore._make_uuid("other.pdf", 1, 0)
        assert a != b

    def test_different_page_different_uuid(self):
        a = WeaviateStore._make_uuid("doc.pdf", 1, 0)
        b = WeaviateStore._make_uuid("doc.pdf", 2, 0)
        assert a != b

    def test_different_chunk_different_uuid(self):
        a = WeaviateStore._make_uuid("doc.pdf", 1, 0)
        b = WeaviateStore._make_uuid("doc.pdf", 1, 1)
        assert a != b

    def test_valid_uuid_format(self):
        import uuid
        result = WeaviateStore._make_uuid("test.txt", 1, 0)
        parsed = uuid.UUID(result)  # raises if invalid
        assert parsed.version == 5


# ---------------------------------------------------------------------------
# Config basics (no external dependencies needed)
# ---------------------------------------------------------------------------

class TestConfig:

    def test_config_imports(self):
        """Config module should be importable and expose expected attributes."""
        from backend.core import config
        assert hasattr(config, "GEMINI_MODEL")
        assert hasattr(config, "CHUNK_SIZE")
        assert hasattr(config, "HYBRID_ALPHA")
        assert hasattr(config, "WEAVIATE_HOST")
        assert hasattr(config, "LLM_PROVIDER")

    def test_chunk_size_positive(self):
        from backend.core import config
        assert config.CHUNK_SIZE > 0

    def test_hybrid_alpha_range(self):
        from backend.core import config
        assert 0.0 <= config.HYBRID_ALPHA <= 1.0

    def test_llm_provider_valid(self):
        from backend.core import config
        assert config.LLM_PROVIDER in ("gemini", "local")
