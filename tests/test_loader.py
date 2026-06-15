"""
Tests for backend.core.loader — document loading and corruption detection.
"""

import os
import tempfile
import pytest
from backend.core.loader import is_text_corrupted, load_text, load_document


# ---------------------------------------------------------------------------
# is_text_corrupted
# ---------------------------------------------------------------------------

class TestIsTextCorrupted:

    def test_english_text_not_corrupted(self):
        assert is_text_corrupted("This is a normal English paragraph.") is False

    def test_empty_text(self):
        assert is_text_corrupted("") is False
        assert is_text_corrupted("   ") is False

    def test_short_text(self):
        assert is_text_corrupted("abc") is False

    def test_valid_bangla_text(self):
        """Valid Bangla with common anchor words should not be flagged."""
        text = "এটি একটি পরীক্ষা। এবং এই ডকুমেন্ট থেকে তথ্য সংগ্রহ করে।"
        assert is_text_corrupted(text) is False

    def test_corrupted_non_unicode(self):
        """Text with mostly non-Bangla-range characters (corruption) should be flagged."""
        # Simulate garbled extraction: lots of Latin chars pretending to be Bangla
        text = "ÿÿÿÿ" * 30 + "ক" * 5  # mostly gibberish, few Bangla chars
        # Should be detected as corrupted since non-Bangla dominates
        result = is_text_corrupted(text)
        assert result is True

    def test_mixed_english_bangla(self):
        """Primarily English text should not be flagged, even with some Bangla."""
        text = "This is mostly English text with ক mixed in."
        assert is_text_corrupted(text) is False

    def test_numbers_and_symbols_only(self):
        """Pure numbers/symbols with no alpha chars should not be flagged."""
        assert is_text_corrupted("1234567890 !@#$%") is False


# ---------------------------------------------------------------------------
# load_text (markdown / txt)
# ---------------------------------------------------------------------------

class TestLoadText:

    def test_loads_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello, this is test content.")
            path = f.name
        try:
            pages = load_text(path)
            assert len(pages) == 1
            assert pages[0]["text"] == "Hello, this is test content."
            assert pages[0]["page"] == 1
            assert pages[0]["source"] == os.path.basename(path)
        finally:
            os.unlink(path)

    def test_loads_markdown_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Heading\n\nSome markdown content.")
            path = f.name
        try:
            pages = load_text(path)
            assert "# Heading" in pages[0]["text"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# load_document dispatch
# ---------------------------------------------------------------------------

class TestLoadDocument:

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document(path)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/file.pdf")

    def test_txt_dispatch(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Dispatch test.")
            path = f.name
        try:
            pages = load_document(path)
            assert len(pages) == 1
            assert pages[0]["text"] == "Dispatch test."
        finally:
            os.unlink(path)
