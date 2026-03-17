"""Tests for commands/dedup_compute_simhashes.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.dedup_compute_simhashes import main as dedup_compute_main


class TestDedupComputeSimhashesCLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["Hello world.", "This is a test."],
                "subtopic_paragraph_start_indices": [0],
            },
            {
                "barcode_src": "book2",
                "middlematter_sentences": ["Another book.", "More content here."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            dedup_compute_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_records = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_records) == 2
        assert "book_id" in output_records[0]
        assert "simhashes" in output_records[0]

    def test_output_format_compact(self, tmp_path: Path):
        """Test that output uses compact format: one line per book with list of hashes."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "testbook",
            "middlematter_sentences": [
                "First paragraph sentence one.",
                "First paragraph sentence two.",
                "Second paragraph sentence.",
                "Third paragraph content.",
            ],
            "subtopic_paragraph_start_indices": [0, 2, 3],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            dedup_compute_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_record = json.loads(output_file.read_text().strip())
        assert output_record["book_id"] == "testbook"
        assert len(output_record["simhashes"]) == 3  # 3 paragraphs
        # Hashes are stored as 32-char hex strings (128-bit)
        assert all(isinstance(h, str) and len(h) == 32 for h in output_record["simhashes"])

    def test_custom_ngram_size(self, tmp_path: Path):
        """Test that custom ngram size is respected."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test sentence for hashing."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            dedup_compute_main,
            [
                "--input-file",
                str(input_file),
                "--output-file",
                str(output_file),
                "--ngram-size",
                "5",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_skips_books_without_paragraphs(self, tmp_path: Path):
        """Test that books without paragraph indices are skipped."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": "has_paras",
                "middlematter_sentences": ["Content."],
                "subtopic_paragraph_start_indices": [0],
            },
            {
                "barcode_src": "no_paras",
                "middlematter_sentences": ["Content."],
                # missing subtopic_paragraph_start_indices
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            dedup_compute_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_records = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_records) == 1
        assert output_records[0]["book_id"] == "has_paras"

    def test_creates_output_directory(self, tmp_path: Path):
        """Test that output directory is created if it doesn't exist."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "nested" / "dir" / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Content."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            dedup_compute_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            dedup_compute_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code != 0
