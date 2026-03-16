"""Tests for commands/step06_pagenumber_removal.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.step06_pagenumber_removal import main as step06_main


class TestStep06CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step06 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter": ["Page one content.", "Page two content."]},
            {"barcode_src": "book2", "middlematter": ["Single page content."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step06_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert "middlematter" in output_books[0]
        assert output_books[0]["barcode_src"] == "book1"

    def test_removes_page_numbers(self, tmp_path: Path):
        """Test that standalone page numbers are removed."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter": [
                "42\nActual content here.\nMore content.",
                "43\nAnother page of content.\n44",
            ],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step06_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())

        assert "42" not in output_book["middlematter"][0]
        assert "Actual content here" in output_book["middlematter"][0]
        assert "43" not in output_book["middlematter"][1]
        assert "44" not in output_book["middlematter"][1]

    def test_preserves_content_with_numbers(self, tmp_path: Path):
        """Test that content containing numbers is preserved."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter": [
                "Chapter 1: Introduction\nIn 1984, the event occurred.",
                "Section 2.3: Analysis\nThe year 2020 was significant.",
            ],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step06_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())

        assert "Chapter 1: Introduction" in output_book["middlematter"][0]
        assert "In 1984" in output_book["middlematter"][0]
        assert "Section 2.3: Analysis" in output_book["middlematter"][1]
