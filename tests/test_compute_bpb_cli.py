"""Tests for commands/compute_bpb.py CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from commands.compute_bpb import main as compute_bpb_main


class TestComputeBPBCLI:
    @patch("commands.compute_bpb.load_bpb_model")
    @patch("commands.compute_bpb.compute_bpb_in_book")
    def test_processes_books(self, mock_compute, mock_load_model, tmp_path: Path):
        """Test CLI processes books and outputs BPB records."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.return_value = {
            "book_id": "book1",
            "bpb_values": [0.85, 1.23],
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.bpb.jsonl"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Sentence one.", "Sentence two."],
            "subtopic_paragraph_start_indices": [0, 1],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            compute_bpb_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_records = [
            json.loads(line) for line in output_file.read_text().strip().split("\n")
        ]
        assert len(output_records) == 1
        assert output_records[0]["book_id"] == "book1"
        assert output_records[0]["bpb_values"] == [0.85, 1.23]

    @patch("commands.compute_bpb.load_bpb_model")
    @patch("commands.compute_bpb.compute_bpb_in_book")
    def test_output_format_matches_expected_structure(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test output JSONL format has {book_id, bpb_values} structure."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.side_effect = [
            {"book_id": "book1", "bpb_values": [0.82, 1.45, 0.67]},
            {"book_id": "book2", "bpb_values": [0.91]},
        ]

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.bpb.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["A.", "B.", "C."],
                "subtopic_paragraph_start_indices": [0, 1, 2],
            },
            {
                "barcode_src": "book2",
                "middlematter_sentences": ["X."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            compute_bpb_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_records = [
            json.loads(line) for line in output_file.read_text().strip().split("\n")
        ]
        assert len(output_records) == 2

        for record in output_records:
            assert "book_id" in record
            assert "bpb_values" in record
            assert isinstance(record["bpb_values"], list)
            assert "barcode_src" not in record
            assert "middlematter_sentences" not in record

        assert output_records[0]["book_id"] == "book1"
        assert output_records[0]["bpb_values"] == [0.82, 1.45, 0.67]
        assert output_records[1]["book_id"] == "book2"
        assert output_records[1]["bpb_values"] == [0.91]

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.bpb.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            compute_bpb_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code != 0

    @patch("commands.compute_bpb.load_bpb_model")
    @patch("commands.compute_bpb.compute_bpb_in_book")
    def test_skips_books_with_errors(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test that books with errors are skipped gracefully."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.side_effect = [
            ValueError("No sentences found"),
            {"book_id": "book2", "bpb_values": [0.91]},
        ]

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.bpb.jsonl"

        books = [
            {"barcode_src": "book1"},
            {
                "barcode_src": "book2",
                "middlematter_sentences": ["X."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            compute_bpb_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_records = [
            json.loads(line) for line in output_file.read_text().strip().split("\n")
        ]
        assert len(output_records) == 1
        assert output_records[0]["book_id"] == "book2"
        assert output_records[0]["bpb_values"] == [0.91]

    @patch("commands.compute_bpb.load_bpb_model")
    @patch("commands.compute_bpb.compute_bpb_in_book")
    def test_creates_output_directory(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test that output directory is created if it doesn't exist."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.return_value = {"book_id": "book1", "bpb_values": [0.8]}

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "nested" / "dir" / "output.bpb.jsonl"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            compute_bpb_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    @patch("commands.compute_bpb.load_bpb_model")
    def test_uses_custom_model_from_config(self, mock_load_model, tmp_path: Path):
        """Test that custom model name from config is used."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.bpb.jsonl"
        config_file = tmp_path / "config.yaml"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))
        config_file.write_text("bpb:\n  model_name: custom/model-name\n")

        with patch("commands.compute_bpb.compute_bpb_in_book") as mock_compute:
            mock_compute.return_value = {"book_id": "book1", "bpb_values": [0.8]}

            runner = CliRunner()
            runner.invoke(
                compute_bpb_main,
                [
                    "--input-file",
                    str(input_file),
                    "--output-file",
                    str(output_file),
                    "--config-file",
                    str(config_file),
                ],
            )

            mock_load_model.assert_called_once()
            call_args = mock_load_model.call_args
            assert call_args[0][0] == "custom/model-name"

    @patch("commands.compute_bpb.load_bpb_model")
    @patch("commands.compute_bpb.compute_bpb_in_book")
    def test_resume_skips_processed_books(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test that --resume skips books that were already processed."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.return_value = {"book_id": "book2", "bpb_values": [1.2]}

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.bpb.jsonl"
        progress_file = tmp_path / "output.bpb.progress.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["A."],
                "subtopic_paragraph_start_indices": [0],
            },
            {
                "barcode_src": "book2",
                "middlematter_sentences": ["B."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        progress_records = [{"book_id": "book1", "bpb_values": [0.9]}]
        progress_file.write_text("\n".join(json.dumps(r) for r in progress_records) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            compute_bpb_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--resume",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_records = [
            json.loads(line) for line in output_file.read_text().strip().split("\n")
        ]
        assert len(output_records) == 2

        book_ids = {r["book_id"] for r in output_records}
        assert book_ids == {"book1", "book2"}

        assert not progress_file.exists()
        assert mock_compute.call_count == 1

    @patch("commands.compute_bpb.load_bpb_model")
    @patch("commands.compute_bpb.compute_bpb_in_book")
    def test_progress_file_cleaned_up_on_success(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test that progress file is removed after successful completion."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.return_value = {"book_id": "book1", "bpb_values": [0.8]}

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.bpb.jsonl"
        progress_file = tmp_path / "output.bpb.progress.jsonl"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            compute_bpb_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert not progress_file.exists()
