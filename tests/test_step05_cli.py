"""Tests for commands/step05_headerfooter_removal.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.steps.step05_headerfooter_removal import main as step05_main


class TestStep05CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step05 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter": ["Page one content.", "Page two content."]},
            {"barcode_src": "book2", "middlematter": ["Single page content."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step05_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert "middlematter" in output_books[0]
        assert output_books[0]["barcode_src"] == "book1"

    def test_removes_repeated_headers(self, tmp_path: Path):
        """Test that repeated headers are removed."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter": [
                "CHAPTER HEADER LINE\nContent of page 1",
                "CHAPTER HEADER LINE\nContent of page 2",
                "CHAPTER HEADER LINE\nContent of page 3",
                "CHAPTER HEADER LINE\nContent of page 4",
            ],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step05_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())

        for page in output_book["middlematter"]:
            assert "CHAPTER HEADER LINE" not in page

    def test_preserves_unique_content(self, tmp_path: Path):
        """Test that unique content is preserved."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter": [
                "Unique line abc 1\nContent A",
                "Unique line def 2\nContent B",
                "Unique line ghi 3\nContent C",
            ],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step05_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())

        assert len(output_book["middlematter"]) == 3
        assert "Unique line abc 1" in output_book["middlematter"][0]
