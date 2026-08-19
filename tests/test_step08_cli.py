"""
Tests for commands/step08_segment.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from commands.steps.step08_segment import main as step08_main


class TestStep08CLI:
    def test_requires_segmenter_option(self, tmp_path: Path):
        """Test that --segmenter option is required."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {"barcode_src": "test", "language_gen": "fra", "middlematter": ["Content."]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step08_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code != 0

    @patch("commands.steps.step08_segment.segment_book_nupunkt")
    def test_processes_nupunkt(self, mock_segment, tmp_path: Path):
        """Test that nupunkt segmenter is called for nupunkt option."""
        mock_segment.return_value = {
            "barcode_src": "test",
            "language_gen": "fra",
            "middlematter_sentences": ["Sentence one.", "Sentence two."],
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {"barcode_src": "test", "language_gen": "fra", "middlematter": ["Content."]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step08_main,
            [
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
                "--segmenter",
                "nupunkt",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        mock_segment.assert_called_once()

        output_book = json.loads(output_file.read_text().strip())
        assert "middlematter_sentences" in output_book

    @patch("commands.steps.step08_segment.get_sat_model")
    @patch("commands.steps.step08_segment.segment_book_sat")
    def test_processes_sat(self, mock_segment, mock_get_model, tmp_path: Path):
        """Test that sat segmenter is called for sat option."""
        mock_get_model.return_value = MagicMock()
        mock_segment.return_value = {
            "barcode_src": "test",
            "language_gen": "zho",
            "middlematter_sentences": ["Sentence."],
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {"barcode_src": "test", "language_gen": "zho", "middlematter": ["Content."]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step08_main,
            [
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
                "--segmenter",
                "sat",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        mock_segment.assert_called_once()

    @patch("commands.steps.step08_segment.segment_book_nupunkt")
    def test_processes_multiple_books(self, mock_segment, tmp_path: Path):
        """Test that multiple books are processed."""
        mock_segment.side_effect = [
            {"barcode_src": "book1", "middlematter_sentences": ["S1."]},
            {"barcode_src": "book2", "middlematter_sentences": ["S2."]},
        ]

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "language_gen": "fra", "middlematter": ["Text 1."]},
            {"barcode_src": "book2", "language_gen": "fra", "middlematter": ["Text 2."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step08_main,
            [
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
                "--segmenter",
                "nupunkt",
            ],
        )

        assert result.exit_code == 0
        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert output_books[0]["barcode_src"] == "book1"
        assert output_books[1]["barcode_src"] == "book2"
