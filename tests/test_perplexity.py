"""Tests for library/perplexity/compute_perplexity.py."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from library.perplexity.compute_perplexity import (
    compute_perplexities_in_book,
    compute_perplexity,
)


class TestComputePerplexity:
    def test_returns_negative_one_for_short_text(self):
        """Test that compute_perplexity returns -1 for text < 5 chars."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = compute_perplexity("Hi", model, tokenizer, "cpu")

        assert result == -1.0
        tokenizer.assert_not_called()

    def test_returns_negative_one_for_empty_text(self):
        """Test that compute_perplexity returns -1 for empty text."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = compute_perplexity("", model, tokenizer, "cpu")

        assert result == -1.0

    def test_returns_float_for_valid_text(self):
        """Test that compute_perplexity returns float > 0 for valid text."""
        model = MagicMock()
        tokenizer = MagicMock()

        # Mock tokenizer output
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        tokenizer.return_value = mock_inputs

        # Mock model output with loss
        mock_output = MagicMock()
        mock_output.loss.item.return_value = 2.0  # log perplexity
        model.return_value = mock_output

        result = compute_perplexity("This is a valid text for testing.", model, tokenizer, "cpu")

        assert isinstance(result, float)
        assert result > 0
        # exp(2.0) ≈ 7.389
        assert abs(result - 7.389) < 0.01

    def test_handles_truncation_warning(self):
        """Test that truncation is handled gracefully."""
        model = MagicMock()
        tokenizer = MagicMock()

        # Mock tokenizer output with overflowing tokens
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "overflowing_tokens": torch.tensor([[4, 5, 6]]),
        }
        tokenizer.return_value = mock_inputs

        mock_output = MagicMock()
        mock_output.loss.item.return_value = 1.5
        model.return_value = mock_output

        result = compute_perplexity("This is a valid text.", model, tokenizer, "cpu")

        assert isinstance(result, float)
        assert result > 0


class TestComputePerplexitiesInBook:
    def test_raises_for_unknown_book(self):
        """Test that ValueError is raised for book without barcode."""
        model = MagicMock()
        tokenizer = MagicMock()
        book = {"middlematter_sentences": ["Test."]}

        with pytest.raises(ValueError, match="Unknown book"):
            compute_perplexities_in_book(book, model, tokenizer, "cpu")

    def test_raises_for_missing_sentences(self):
        """Test that ValueError is raised for book without sentences."""
        model = MagicMock()
        tokenizer = MagicMock()
        book = {"barcode_src": "book1"}

        with pytest.raises(ValueError, match="No sentences found"):
            compute_perplexities_in_book(book, model, tokenizer, "cpu")

    def test_raises_for_missing_paragraph_indices(self):
        """Test that ValueError is raised for book without paragraph indices."""
        model = MagicMock()
        tokenizer = MagicMock()
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test sentence."],
        }

        with pytest.raises(ValueError, match="No paragraph indices"):
            compute_perplexities_in_book(book, model, tokenizer, "cpu")

    @patch("library.perplexity.compute_perplexity.compute_perplexity")
    def test_returns_correct_count(self, mock_compute):
        """Test that perplexities count matches paragraph count."""
        mock_compute.return_value = 10.0
        model = MagicMock()
        tokenizer = MagicMock()

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
                "Fourth sentence.",
            ],
            "subtopic_paragraph_start_indices": [0, 2],  # 2 paragraphs
        }

        result = compute_perplexities_in_book(book, model, tokenizer, "cpu")

        assert result["book_id"] == "book1"
        assert len(result["perplexities"]) == 2
        assert all(p == 10.0 for p in result["perplexities"])

    @patch("library.perplexity.compute_perplexity.compute_perplexity")
    def test_paragraphs_built_correctly(self, mock_compute):
        """Test that paragraphs are built from sentences correctly."""
        captured_texts = []

        def capture_text(text, model, tokenizer, device):
            captured_texts.append(text)
            return 5.0

        mock_compute.side_effect = capture_text
        model = MagicMock()
        tokenizer = MagicMock()

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["A.", "B.", "C.", "D."],
            "subtopic_paragraph_start_indices": [0, 2],
        }

        compute_perplexities_in_book(book, model, tokenizer, "cpu")

        # First paragraph: sentences 0-1, Second paragraph: sentences 2-3
        assert captured_texts == ["A. B.", "C. D."]
