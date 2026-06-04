"""Tests for commands/step04_dehyphenate.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from commands.steps.step04_dehyphenate import main as step04_main
from library.denoise.ngrams import NGramScorer, build_ngram_stats


class TestStep04CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step04 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter": ["Page one.", "Page two."], "language_gen": "en"},
            {"barcode_src": "book2", "middlematter": ["Single page."], "language_gen": "en"},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step04_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert "middlematter" in output_books[0]
        assert output_books[0]["barcode_src"] == "book1"

    def test_no_hyphens_unchanged(self, tmp_path: Path):
        """Test that books without hyphens pass through unchanged."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter": ["Normal text here.", "More normal text."],
            "language_gen": "en",
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step04_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())
        assert output_book["middlematter"] == book["middlematter"]

    def test_dehyphenation_with_mocked_scorer(self, tmp_path: Path):
        """Test dehyphenation processing with mocked n-gram scorer."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter": ["Text with hyphen-\nated word."],
            "language_gen": "en",
        }
        input_file.write_text(json.dumps(book))

        stats = build_ngram_stats("text with hyphenated word")
        mock_scorer = NGramScorer(stats)

        runner = CliRunner()
        with patch(
            "library.denoise.dehyphenate.load_ngram_scorer",
            return_value=mock_scorer,
        ):
            result = runner.invoke(
                step04_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
            )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())
        assert "middlematter" in output_book
