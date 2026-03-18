"""Tests for commands/step15_clean.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.step15_clean import main as step15_main


class TestStep15CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step15 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "language_gen": "eng",
                "annotated_frontmatter": "<idi-endmatter>...</idi-endmatter>",
                "annotated_middlematter": "<idi-section>...</idi-section>",
                "annotated_backmatter": "",
                "metadata": {"token_count": 100},
                "middlematter_sentences": ["A.", "B."],
                "subtopic_paragraph_start_indices": [0, 1],
                "subtopic_section_start_indices": [0],
                # These should be removed
                "frontmatter": ["Front"],
                "middlematter": ["Middle"],
                "backmatter": ["Back"],
                "duplicate_paragraphs": {},
                "representative_paragraphs": {},
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step15_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 1

        output_book = output_books[0]
        # Should keep essential fields
        assert output_book["barcode_src"] == "book1"
        assert output_book["language_gen"] == "eng"
        assert "annotated_frontmatter" in output_book
        assert "annotated_middlematter" in output_book
        assert "annotated_backmatter" in output_book
        assert "metadata" in output_book

        # Should remove intermediate fields (including sentences/indices by default now)
        assert "frontmatter" not in output_book
        assert "middlematter" not in output_book
        assert "backmatter" not in output_book
        assert "duplicate_paragraphs" not in output_book
        assert "representative_paragraphs" not in output_book
        assert "middlematter_sentences" not in output_book
        assert "subtopic_paragraph_start_indices" not in output_book
        assert "subtopic_section_start_indices" not in output_book

    def test_no_keep_sentences(self, tmp_path: Path):
        """Test --no-keep-sentences removes middlematter_sentences."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "language_gen": "eng",
            "annotated_middlematter": "...",
            "middlematter_sentences": ["A.", "B."],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step15_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--no-keep-sentences",
            ],
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert "middlematter_sentences" not in output_book

    def test_no_keep_indices(self, tmp_path: Path):
        """Test --no-keep-indices removes paragraph/section indices."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "language_gen": "eng",
            "annotated_middlematter": "...",
            "subtopic_paragraph_start_indices": [0, 1],
            "subtopic_section_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step15_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--no-keep-indices",
            ],
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert "subtopic_paragraph_start_indices" not in output_book
        assert "subtopic_section_start_indices" not in output_book

    def test_processes_multiple_books(self, tmp_path: Path):
        """Test that multiple books are processed."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "language_gen": "eng"},
            {"barcode_src": "book2", "language_gen": "deu"},
            {"barcode_src": "book3", "language_gen": "fra"},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step15_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 3
        assert output_books[0]["barcode_src"] == "book1"
        assert output_books[1]["barcode_src"] == "book2"
        assert output_books[2]["barcode_src"] == "book3"

    def test_removes_error_fields(self, tmp_path: Path):
        """Test that error fields are removed in cleanup."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "language_gen": "eng",
            "annotation_error": "Some error",
            "metadata_error": "Another error",
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step15_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert "annotation_error" not in output_book
        assert "metadata_error" not in output_book

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            step15_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code != 0
