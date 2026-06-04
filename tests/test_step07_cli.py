"""Tests for commands/step07_validate_segmenter.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.steps.step07_validate_segmenter import main as step07_main


class TestStep07CLI:
    def test_processes_jsonl_file_nupunkt(self, tmp_path: Path):
        """Test that step07 CLI processes nupunkt language books correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # "fra" (French) is a nupunkt language
        books = [
            {"barcode_src": "book1", "language_gen": "fra", "middlematter": ["Content."]},
            {"barcode_src": "book2", "language_gen": "ita", "middlematter": ["Content."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step07_main,
            ["--input-file", str(input_file), "--output-file", str(output_file), "--segmenter", "nupunkt"],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2

    def test_processes_jsonl_file_sat(self, tmp_path: Path):
        """Test that step07 CLI processes sat language books correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # "zho" (Chinese) is a sat language
        books = [
            {"barcode_src": "book1", "language_gen": "zho", "middlematter": ["Content."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step07_main,
            ["--input-file", str(input_file), "--output-file", str(output_file), "--segmenter", "sat"],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_fails_on_segmenter_mismatch(self, tmp_path: Path):
        """Test that mismatched segmenter raises error."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        # "fra" is nupunkt but we're using sat segmenter
        book = {"barcode_src": "test", "language_gen": "fra", "middlematter": ["Content."]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step07_main,
            ["--input-file", str(input_file), "--output-file", str(output_file), "--segmenter", "sat"],
        )

        assert result.exit_code != 0
        assert "should be in nupunkt shard" in str(result.exception)

    def test_requires_segmenter_option(self, tmp_path: Path):
        """Test that --segmenter option is required."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {"barcode_src": "test", "language_gen": "fra", "middlematter": ["Content."]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step07_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code != 0
