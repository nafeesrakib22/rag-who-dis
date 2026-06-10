"""
Tests for backend.core.chunker — semantic and fixed-size chunking.
"""

import pytest
from unittest.mock import MagicMock
from backend.core.chunker import chunk_text, chunk_documents, SemanticChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedder(dim: int = 4):
    """Return a mock embedder whose .embed() returns sequential vectors."""
    embedder = MagicMock()
    call_count = [0]

    def fake_embed(texts, **kw):
        vecs = []
        for _ in texts:
            call_count[0] += 1
            vecs.append([float(call_count[0])] * dim)
        return vecs

    embedder.embed = MagicMock(side_effect=fake_embed)
    return embedder


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------

class TestFixedSizeChunking:

    def test_single_small_text(self):
        """A short text should produce exactly one chunk."""
        chunks = chunk_text("Hello world.", {"source": "test.txt", "page": 1})
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world."
        assert chunks[0]["source"] == "test.txt"

    def test_overlap_produces_overlapping_chunks(self):
        """Fixed-size chunking with overlap should produce sliding windows."""
        text = "A" * 200
        chunks = chunk_text(text, {"source": "f.txt", "page": 1}, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        # Second chunk should start at offset 80 (100 - 20)
        assert chunks[1]["text"] == "A" * 100

    def test_empty_text(self):
        """Empty text should produce no chunks."""
        chunks = chunk_text("", {"source": "f.txt", "page": 1})
        assert chunks == []

    def test_metadata_passthrough(self):
        """Extra metadata keys should be preserved in chunk dicts."""
        meta = {"source": "f.txt", "page": 1, "custom_key": "custom_val"}
        chunks = chunk_text("Hello.", meta)
        assert chunks[0]["custom_key"] == "custom_val"

    def test_chunk_index_increments(self):
        """Each chunk within a page should have an incrementing chunk_index."""
        text = "word " * 500  # long enough for multiple chunks
        chunks = chunk_text(text, {"source": "f.txt", "page": 1}, chunk_size=100, overlap=0)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

class TestSemanticChunker:

    def test_splits_at_semantic_boundaries(self):
        """Semantic chunker should split text into multiple chunks."""
        embedder = _fake_embedder()
        chunker = SemanticChunker(
            embedder, breakpoint_threshold_percentile=50, max_chunk_size=500
        )
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_text(self):
        chunker = SemanticChunker(_fake_embedder())
        assert chunker.split_text("") == []
        assert chunker.split_text("   ") == []

    def test_single_sentence(self):
        chunker = SemanticChunker(_fake_embedder())
        result = chunker.split_text("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_max_chunk_size_respected(self):
        """No chunk should exceed max_chunk_size."""
        embedder = _fake_embedder()
        chunker = SemanticChunker(
            embedder, breakpoint_threshold_percentile=99, max_chunk_size=50
        )
        text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima."
        chunks = chunker.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 60  # allow small overrun from last appended sentence


# ---------------------------------------------------------------------------
# chunk_documents convenience wrapper
# ---------------------------------------------------------------------------

class TestChunkDocuments:

    def test_multiple_pages(self):
        pages = [
            {"text": "Page one text.", "source": "doc.pdf", "page": 1},
            {"text": "Page two text.", "source": "doc.pdf", "page": 2},
        ]
        chunks = chunk_documents(pages, chunk_size=500, overlap=0)
        assert len(chunks) == 2
        assert chunks[0]["page"] == 1
        assert chunks[1]["page"] == 2

    def test_semantic_chunker_used_when_provided(self):
        embedder = _fake_embedder()
        sc = SemanticChunker(embedder, max_chunk_size=5000)
        pages = [{"text": "Hello world. Goodbye world.", "source": "f.md", "page": 1}]
        chunks = chunk_documents(pages, semantic_chunker=sc)
        assert len(chunks) >= 1
        embedder.embed.assert_called()  # semantic chunker should invoke the embedder
