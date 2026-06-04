"""Tests for commands/step09_remove_stray_numbers.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.steps.step09_remove_stray_numbers import main as step09_main


class TestStep09CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step09 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter_sentences": ["Content.", "123", "More."]},
            {"barcode_src": "book2", "middlematter_sentences": ["Text.", "456", "End."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step09_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert "middlematter_sentences" in output_books[0]
        assert "middlematter_sentences" in output_books[1]

    def test_removes_stray_numbers(self, tmp_path: Path):
        """Test that stray number sentences are removed."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter_sentences": [
                "This is a real sentence.",
                "4200",
                "Another sentence here.",
                "9999",
                "Final sentence.",
            ],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step09_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        sentences = output_book["middlematter_sentences"]
        assert "4200" not in sentences
        assert "9999" not in sentences
        assert "This is a real sentence." in sentences
        assert "Another sentence here." in sentences
        assert "Final sentence." in sentences

    def test_preserves_sentences_with_embedded_numbers(self, tmp_path: Path):
        """Test that sentences containing numbers as part of text are preserved."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "middlematter_sentences": [
                "Chapter 1 begins here.",
                "There are 42 items in the list.",
                "The year was 1984.",
            ],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step09_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        sentences = output_book["middlematter_sentences"]
        assert len(sentences) == 3
        assert "Chapter 1 begins here." in sentences
        assert "There are 42 items in the list." in sentences
        assert "The year was 1984." in sentences

    def test_processes_multiple_books(self, tmp_path: Path):
        """Test that multiple books are processed correctly."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter_sentences": ["Sentence A.", "111"]},
            {"barcode_src": "book2", "middlematter_sentences": ["Sentence B.", "222"]},
            {"barcode_src": "book3", "middlematter_sentences": ["Sentence C.", "333"]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step09_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 3
        assert output_books[0]["barcode_src"] == "book1"
        assert output_books[1]["barcode_src"] == "book2"
        assert output_books[2]["barcode_src"] == "book3"
        # Verify stray numbers removed from each
        assert "111" not in output_books[0]["middlematter_sentences"]
        assert "222" not in output_books[1]["middlematter_sentences"]
        assert "333" not in output_books[2]["middlematter_sentences"]

    def test_preserves_other_book_fields(self, tmp_path: Path):
        """Test that other book fields are preserved unchanged."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test123",
            "language_gen": "eng",
            "title": "Test Book",
            "middlematter_sentences": ["Content.", "456"],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step09_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert output_book["barcode_src"] == "test123"
        assert output_book["language_gen"] == "eng"
        assert output_book["title"] == "Test Book"

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            step09_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code != 0
