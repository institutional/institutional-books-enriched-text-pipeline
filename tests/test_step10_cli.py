"""Tests for commands/step10_chunk.py CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from commands.steps.step10_chunk import main as step10_main


class TestStep10CLI:
    @patch("commands.steps.step10_chunk.get_embedding_model")
    @patch("commands.steps.step10_chunk.chunk_book_texttiling")
    def test_processes_jsonl_file(self, mock_chunk, mock_get_model, tmp_path: Path):
        """Test that step10 CLI reads input JSONL and writes output JSONL."""
        mock_get_model.return_value = MagicMock()
        mock_chunk.return_value = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Sentence one.", "Sentence two."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter_sentences": ["Sentence one.", "Sentence two."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step10_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 1
        assert "subtopic_paragraph_start_indices" in output_books[0]
        assert "subtopic_section_start_indices" in output_books[0]

    @patch("commands.steps.step10_chunk.get_embedding_model")
    @patch("commands.steps.step10_chunk.chunk_book_texttiling")
    def test_uses_texttiling_by_default(self, mock_texttiling, mock_get_model, tmp_path: Path):
        """Test that texttiling is used by default."""
        mock_get_model.return_value = MagicMock()
        mock_texttiling.return_value = {
            "barcode_src": "test",
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {"barcode_src": "test", "middlematter_sentences": ["Content."]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step10_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        mock_texttiling.assert_called_once()

    @patch("commands.steps.step10_chunk.get_embedding_model")
    @patch("commands.steps.step10_chunk.chunk_book_c99")
    def test_algorithm_override_c99(self, mock_c99, mock_get_model, tmp_path: Path):
        """Test that --algorithm c99 uses C99 chunking."""
        mock_get_model.return_value = MagicMock()
        mock_c99.return_value = {
            "barcode_src": "test",
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {"barcode_src": "test", "middlematter_sentences": ["Content."]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step10_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--algorithm", "c99",
            ],
        )

        assert result.exit_code == 0
        mock_c99.assert_called_once()

    @patch("commands.steps.step10_chunk.get_embedding_model")
    @patch("commands.steps.step10_chunk.chunk_book_texttiling")
    def test_processes_multiple_books(self, mock_chunk, mock_get_model, tmp_path: Path):
        """Test that multiple books are processed."""
        mock_get_model.return_value = MagicMock()
        mock_chunk.side_effect = [
            {
                "barcode_src": "book1",
                "subtopic_paragraph_start_indices": [0],
                "subtopic_section_start_indices": [0],
            },
            {
                "barcode_src": "book2",
                "subtopic_paragraph_start_indices": [0, 2],
                "subtopic_section_start_indices": [0],
            },
        ]

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter_sentences": ["A.", "B."]},
            {"barcode_src": "book2", "middlematter_sentences": ["C.", "D.", "E."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step10_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert output_books[0]["barcode_src"] == "book1"
        assert output_books[1]["barcode_src"] == "book2"

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            step10_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code != 0
