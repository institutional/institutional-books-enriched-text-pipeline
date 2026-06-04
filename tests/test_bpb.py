"""Tests for library/bpb/compute_bpb.py."""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from library.bpb.compute_bpb import (
    compute_bpb,
    compute_bpb_in_book,
)


class TestComputeBPB:
    def test_returns_negative_one_for_short_text(self):
        """Test that compute_bpb returns -1 for text < 5 chars."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = compute_bpb("Hi", model, tokenizer, "cpu")

        assert result == -1.0
        tokenizer.assert_not_called()

    def test_returns_negative_one_for_empty_text(self):
        """Test that compute_bpb returns -1 for empty text."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = compute_bpb("", model, tokenizer, "cpu")

        assert result == -1.0

    def test_returns_float_for_valid_text(self):
        """Test that compute_bpb returns float > 0 for valid text."""
        model = MagicMock()
        tokenizer = MagicMock()

        text = "This is a valid text for testing."
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        tokenizer.return_value = mock_inputs

        mock_output = MagicMock()
        mock_output.loss.item.return_value = 2.0  # mean loss in nats
        model.return_value = mock_output

        result = compute_bpb(text, model, tokenizer, "cpu")

        assert isinstance(result, float)
        assert result > 0
        # n_predicted = 5 - 1 = 4, total_loss = 2.0 * 4 = 8.0
        # n_bytes = len("This is a valid text for testing.".encode()) = 33
        # bpb = 8.0 / (33 * log(2))
        expected = (2.0 * 4) / (len(text.encode("utf-8")) * math.log(2))
        assert abs(result - expected) < 0.001

    def test_handles_truncation_warning(self):
        """Test that truncation is handled gracefully."""
        model = MagicMock()
        tokenizer = MagicMock()

        text = "This is a valid text."
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "overflowing_tokens": torch.tensor([[4, 5, 6]]),
        }
        tokenizer.return_value = mock_inputs

        mock_output = MagicMock()
        mock_output.loss.item.return_value = 1.5
        model.return_value = mock_output

        result = compute_bpb(text, model, tokenizer, "cpu")

        assert isinstance(result, float)
        assert result > 0


class TestComputeBPBInBook:
    def test_raises_for_unknown_book(self):
        """Test that ValueError is raised for book without barcode."""
        model = MagicMock()
        tokenizer = MagicMock()
        book = {"middlematter_sentences": ["Test."]}

        with pytest.raises(ValueError, match="Unknown book"):
            compute_bpb_in_book(book, model, tokenizer, "cpu")

    def test_raises_for_missing_sentences(self):
        """Test that ValueError is raised for book without sentences."""
        model = MagicMock()
        tokenizer = MagicMock()
        book = {"barcode_src": "book1"}

        with pytest.raises(ValueError, match="No sentences found"):
            compute_bpb_in_book(book, model, tokenizer, "cpu")

    def test_raises_for_missing_paragraph_indices(self):
        """Test that ValueError is raised for book without paragraph indices."""
        model = MagicMock()
        tokenizer = MagicMock()
        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test sentence."],
        }

        with pytest.raises(ValueError, match="No paragraph indices"):
            compute_bpb_in_book(book, model, tokenizer, "cpu")

    @patch("library.bpb.compute_bpb.compute_bpb_batched")
    def test_returns_correct_count(self, mock_batched):
        """Test that BPB values count matches paragraph count."""
        mock_batched.return_value = [0.85, 0.92]
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
            "subtopic_paragraph_start_indices": [0, 2],
        }

        result = compute_bpb_in_book(book, model, tokenizer, "cpu")

        assert result["book_id"] == "book1"
        assert len(result["bpb_values"]) == 2
        assert all(v == pytest.approx(v) for v in result["bpb_values"])

    @patch("library.bpb.compute_bpb.compute_bpb_batched")
    def test_paragraphs_built_correctly(self, mock_batched):
        """Test that paragraphs are built from sentences correctly."""
        mock_batched.return_value = [0.7, 0.8]
        model = MagicMock()
        tokenizer = MagicMock()

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["A.", "B.", "C.", "D."],
            "subtopic_paragraph_start_indices": [0, 2],
        }

        compute_bpb_in_book(book, model, tokenizer, "cpu")

        call_args = mock_batched.call_args
        paragraphs_arg = call_args[0][0]
        assert paragraphs_arg == ["A. B.", "C. D."]
