"""Tests for commands/step01_uniformize_unicode.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.step01_uniformize_unicode import main as step01_main


class TestStep01CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step01 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "text_by_page_src": ["Page one.", "Page two."]},
            {"barcode_src": "book2", "text_by_page_src": ["Hello world."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step01_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert "uniformized_text" in output_books[0]
        assert "text_by_page_src" not in output_books[0]
        assert output_books[0]["barcode_src"] == "book1"

    def test_unicode_normalization_applied(self, tmp_path: Path):
        """Test that unicode normalization is actually applied."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {"barcode_src": "test", "text_by_page_src": ["Hello\u2014world"]}
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step01_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())
        assert output_book["uniformized_text"] == ["Hello-world"]
