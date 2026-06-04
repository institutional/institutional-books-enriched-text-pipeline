"""Tests for commands/step14_add_metadata.py CLI."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from commands.step14_add_metadata import main as step14_main


class TestStep14CLI:
    @patch("commands.step14_add_metadata.compute_text_stats")
    def test_processes_jsonl_file(self, mock_compute_stats, tmp_path: Path):
        """Test that step14 CLI reads input JSONL and writes output JSONL."""
        mock_compute_stats.return_value = {
            "token_count": 100,
            "char_count": 500,
            "word_count": 50,
            "sentence_count": 5,
            "paragraph_count": 2,
            "section_count": 1,
            "bigram_count": 49,
            "bigram_count_unique": 40,
            "trigram_count": 48,
            "trigram_count_unique": 38,
            "tokenizability_o200k_base_ratio": 85.5,
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["Sentence one.", "Sentence two."],
                "subtopic_paragraph_start_indices": [0, 1],
                "subtopic_section_start_indices": [0],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step14_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 1
        assert "metadata" in output_books[0]
        assert output_books[0]["metadata"]["token_count"] == 100

    @patch("commands.step14_add_metadata.compute_text_stats")
    def test_includes_bpb_stats(self, mock_compute_stats, tmp_path: Path):
        """Test that BPB statistics are included when file is provided."""
        mock_compute_stats.return_value = {
            "token_count": 50,
            "char_count": 200,
            "word_count": 25,
            "sentence_count": 2,
            "paragraph_count": 1,
            "section_count": 1,
            "bigram_count": 24,
            "bigram_count_unique": 20,
            "trigram_count": 23,
            "trigram_count_unique": 18,
            "tokenizability_o200k_base_ratio": 90.0,
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        bpb_file = tmp_path / "bpb.jsonl"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Content."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))

        bpb_record = {"book_id": "book1", "bpb_values": [0.8, 1.2, 1.6]}
        bpb_file.write_text(json.dumps(bpb_record))

        runner = CliRunner()
        result = runner.invoke(
            step14_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--bpb-file", str(bpb_file),
            ],
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert "bpb_min" in output_book["metadata"]
        assert "bpb_max" in output_book["metadata"]
        assert output_book["metadata"]["bpb_min"] == 0.8
        assert output_book["metadata"]["bpb_max"] == 1.6

    @patch("commands.step14_add_metadata.compute_text_stats")
    def test_processes_multiple_books(self, mock_compute_stats, tmp_path: Path):
        """Test that multiple books are processed."""
        mock_compute_stats.side_effect = [
            {"token_count": 100, "char_count": 500},
            {"token_count": 200, "char_count": 1000},
        ]

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {"barcode_src": "book1", "middlematter_sentences": ["A."]},
            {"barcode_src": "book2", "middlematter_sentences": ["B."]},
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step14_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2
        assert output_books[0]["metadata"]["token_count"] == 100
        assert output_books[1]["metadata"]["token_count"] == 200

    @patch("commands.step14_add_metadata.compute_text_stats")
    def test_preserves_existing_fields(self, mock_compute_stats, tmp_path: Path):
        """Test that existing book fields are preserved."""
        mock_compute_stats.return_value = {"token_count": 100}

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "language_gen": "eng",
            "annotated_middlematter": "<section>...</section>",
            "middlematter_sentences": ["Content."],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step14_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert output_book["barcode_src"] == "book1"
        assert output_book["language_gen"] == "eng"
        assert output_book["annotated_middlematter"] == "<section>...</section>"

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            step14_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code != 0
