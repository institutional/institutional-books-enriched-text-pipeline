"""Tests for commands/step02_remove_duplicate_pages.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.step02_remove_duplicate_pages import main as step02_main


class TestStep02CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step02 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "uniformized_text": ["Page one.", "Page two."]},
            {"barcode_src": "book2", "uniformized_text": ["Single page."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step02_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert "uniformized_text" in output_books[0]
        assert output_books[0]["barcode_src"] == "book1"

    def test_removes_duplicate_pages(self, tmp_path: Path):
        """Test that duplicate pages are actually removed."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        duplicate_page = "This is a page with enough content to be considered for deduplication."
        book = {
            "barcode_src": "test",
            "uniformized_text": [duplicate_page, duplicate_page, "Different content here."],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step02_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())
        assert len(output_book["uniformized_text"]) == 2
        assert output_book.get("_duplicate_pages_removed") == 1
