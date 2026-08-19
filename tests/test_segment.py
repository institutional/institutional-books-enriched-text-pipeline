"""
Tests for library/segment/ segmenters.

Institutional Books - Enriched Text - 2026
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from library.segment.nupunkt_segmenter import (
    locate_base_model_for_language,
    segment_book_nupunkt,
)
from library.segment.sat_segmenter import (
    segment_book_sat,
    segment_text_with_sat,
)

# Check for nupunkt availability
try:
    import nupunkt  # noqa: F401

    _nupunkt_available = True
except ImportError:
    _nupunkt_available = False

# Check for SAT/wtpsplit and GPU availability
try:
    import torch
    from wtpsplit import SaT  # noqa: F401

    _sat_available = torch.cuda.is_available() or torch.backends.mps.is_available()
except ImportError:
    _sat_available = False

# Default model directory
_default_model_dir = Path("./DATA/pretrain/models")

# Skip markers
requires_nupunkt = pytest.mark.skipif(not _nupunkt_available, reason="nupunkt not installed")
requires_nupunkt_model = pytest.mark.skipif(
    not _nupunkt_available or not (_default_model_dir / "eng_nupunkt.bin").exists(),
    reason="nupunkt not installed or English model not available",
)
requires_sat = pytest.mark.skipif(
    not _sat_available, reason="wtpsplit not installed or no GPU available"
)


class TestSegmentBookNupunkt:
    """Tests for nupunkt segmentation."""

    def test_missing_language_raises(self):
        """Test that missing language_gen raises ValueError."""
        book = {"barcode_src": "test", "middlematter": ["Some text."]}

        with pytest.raises(ValueError, match="No 'language_gen' found"):
            segment_book_nupunkt(book, config=None)

    def test_missing_middlematter_raises(self):
        """Test that missing middlematter raises ValueError."""
        book = {"barcode_src": "test", "language_gen": "fra"}

        with pytest.raises(ValueError, match="missing middlematter"):
            segment_book_nupunkt(book, config=None)

    def test_empty_middlematter_raises(self):
        """Test that empty middlematter raises ValueError."""
        book = {"barcode_src": "test", "language_gen": "fra", "middlematter": []}

        with pytest.raises(ValueError, match="missing middlematter"):
            segment_book_nupunkt(book, config=None)

    @patch("library.segment.nupunkt_segmenter.locate_base_model_for_language")
    @patch("library.segment.nupunkt_segmenter.adapt_model_to_book_inmemory")
    def test_segments_middlematter(self, mock_adapt, mock_locate):
        """Test that segmentation creates middlematter_sentences."""
        mock_locate.return_value = "/fake/model.bin"
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["Sentence one.", "Sentence two."]
        mock_adapt.return_value = mock_tokenizer

        book = {
            "barcode_src": "test",
            "language_gen": "fra",
            "middlematter": ["Sentence one. Sentence two."],
        }

        result = segment_book_nupunkt(book, config=None)

        assert "middlematter_sentences" in result
        assert result["middlematter_sentences"] == ["Sentence one.", "Sentence two."]
        assert "middlematter" not in result

    @patch("library.segment.nupunkt_segmenter.locate_base_model_for_language")
    @patch("library.segment.nupunkt_segmenter.adapt_model_to_book_inmemory")
    def test_preserves_other_fields(self, mock_adapt, mock_locate):
        """Test that other book fields are preserved."""
        mock_locate.return_value = "/fake/model.bin"
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["Sentence."]
        mock_adapt.return_value = mock_tokenizer

        book = {
            "barcode_src": "test123",
            "language_gen": "fra",
            "middlematter": ["Sentence."],
            "title": "Test Book",
        }

        result = segment_book_nupunkt(book, config=None)

        assert result["barcode_src"] == "test123"
        assert result["title"] == "Test Book"


class TestLocateBaseModel:
    """Tests for model location."""

    def test_missing_model_raises(self, tmp_path):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Base model for language"):
            locate_base_model_for_language("xyz", tmp_path)

    def test_finds_existing_model(self, tmp_path):
        """Test that existing model file is found."""
        model_file = tmp_path / "fra_nupunkt.bin"
        model_file.write_text("fake model")

        result = locate_base_model_for_language("fra", tmp_path)

        assert result == model_file


class TestSegmentBookSat:
    """Tests for SAT segmentation."""

    def test_missing_middlematter_raises(self):
        """Test that missing middlematter raises ValueError."""
        book = {"barcode_src": "test", "language_gen": "zho"}

        with pytest.raises(ValueError, match="missing middlematter"):
            segment_book_sat(book, config=None)

    def test_empty_middlematter_raises(self):
        """Test that empty middlematter raises ValueError."""
        book = {"barcode_src": "test", "language_gen": "zho", "middlematter": []}

        with pytest.raises(ValueError, match="missing middlematter"):
            segment_book_sat(book, config=None)

    def test_segments_with_mock_model(self):
        """Test that segmentation works with a mock SAT model."""
        mock_sat = MagicMock()
        mock_sat.split.return_value = ["Sentence one.", "Sentence two."]

        book = {
            "barcode_src": "test",
            "language_gen": "zho",
            "middlematter": ["Sentence one. Sentence two."],
        }

        result = segment_book_sat(book, config=None, sat_model=mock_sat)

        assert "middlematter_sentences" in result
        assert result["middlematter_sentences"] == ["Sentence one.", "Sentence two."]
        assert "middlematter" not in result

    def test_preserves_other_fields(self):
        """Test that other book fields are preserved."""
        mock_sat = MagicMock()
        mock_sat.split.return_value = ["Sentence."]

        book = {
            "barcode_src": "test123",
            "language_gen": "zho",
            "middlematter": ["Sentence."],
            "title": "Test Book",
        }

        result = segment_book_sat(book, config=None, sat_model=mock_sat)

        assert result["barcode_src"] == "test123"
        assert result["title"] == "Test Book"


class TestSegmentTextWithSat:
    """Tests for low-level SAT segmentation."""

    def test_calls_model_split(self):
        """Test that segment_text_with_sat calls model.split."""
        mock_sat = MagicMock()
        mock_sat.split.return_value = ["Hello.", "World."]

        result = segment_text_with_sat(mock_sat, "Hello. World.")

        mock_sat.split.assert_called_once_with("Hello. World.")
        assert result == ["Hello.", "World."]


# =============================================================================
# Integration tests - require actual models
# =============================================================================


@requires_nupunkt
class TestNupunktIntegration:
    """Integration tests for nupunkt segmentation with real models."""

    def test_nupunkt_import(self):
        """Test that nupunkt can be imported and used."""
        import nupunkt

        # Basic test with built-in English model
        text = "Hello world. How are you? I am fine."
        sentences = nupunkt.sent_tokenize(text)

        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine."

    def test_nupunkt_handles_abbreviations(self):
        """Test that nupunkt handles common abbreviations."""
        import nupunkt

        text = "Dr. Smith went to Washington. He met Mr. Jones there."
        sentences = nupunkt.sent_tokenize(text)

        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
        assert "Mr. Jones" in sentences[1]

    def test_nupunkt_multiline(self):
        """Test that nupunkt handles multiline text."""
        import nupunkt

        text = "First sentence.\nSecond sentence.\nThird sentence."
        sentences = nupunkt.sent_tokenize(text)

        assert len(sentences) == 3


@requires_nupunkt_model
class TestNupunktModelIntegration:
    """Integration tests requiring nupunkt model files."""

    def test_segment_book_with_real_model(self):
        """Test full book segmentation with real nupunkt model."""
        book = {
            "barcode_src": "test",
            "language_gen": "eng",
            "middlematter": [
                "This is the first page of the book with interesting content. The second sentence on this page discusses important topics.",
                "The third sentence appears on page two of the document. Here is the fourth sentence with more information. Finally, the fifth sentence concludes the text.",
            ],
        }

        result = segment_book_nupunkt(book, config=None, model_dir=_default_model_dir)

        assert "middlematter_sentences" in result
        assert "middlematter" not in result
        # Should have split into 5 sentences
        assert len(result["middlematter_sentences"]) >= 5

    def test_segment_preserves_content(self):
        """Test that segmentation preserves all text content."""
        original_text = "Hello world. How are you today? I am doing well."
        book = {
            "barcode_src": "test",
            "language_gen": "eng",
            "middlematter": [original_text],
        }

        result = segment_book_nupunkt(book, config=None, model_dir=_default_model_dir)

        # Rejoined sentences should contain all original words
        rejoined = " ".join(result["middlematter_sentences"])
        for word in ["Hello", "world", "How", "are", "you", "today", "well"]:
            assert word in rejoined


@requires_sat
class TestSatIntegration:
    """Integration tests for SAT segmentation with real models."""

    def test_sat_model_loads(self):
        """Test that SAT model can be loaded."""
        from library.segment.sat_segmenter import load_sat_model

        model = load_sat_model()
        assert model is not None

    def test_sat_segments_text(self):
        """Test that SAT segments text correctly."""
        from library.segment.sat_segmenter import load_sat_model, segment_text_with_sat

        model = load_sat_model()
        text = "Hello world. How are you? I am fine."
        sentences = segment_text_with_sat(model, text)

        assert len(sentences) >= 2  # Should split into multiple sentences
        # All original content should be preserved
        rejoined = "".join(sentences)
        assert "Hello" in rejoined
        assert "fine" in rejoined

    def test_segment_book_sat_with_real_model(self):
        """Test full book segmentation with real SAT model."""
        from library.segment.sat_segmenter import load_sat_model

        model = load_sat_model()
        book = {
            "barcode_src": "test",
            "language_gen": "zho",
            "middlematter": ["First sentence. Second sentence. Third one here."],
        }

        result = segment_book_sat(book, config=None, sat_model=model)

        assert "middlematter_sentences" in result
        assert "middlematter" not in result
        assert len(result["middlematter_sentences"]) >= 2
