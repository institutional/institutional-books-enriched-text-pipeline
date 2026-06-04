"""Tests for commands/process_shard.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from commands.process_shard import main as process_shard_main


class TestProcessShardCLI:
    @patch("commands.process_shard.import_step_function")
    def test_resume_skips_processed_books(self, mock_import_step, tmp_path: Path):
        """Test that --resume skips books that were already processed."""
        # Mock step function to just pass through
        mock_import_step.return_value = lambda book, config, seg: book

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create input with 3 books
        books = [
            {"barcode_src": f"book{i}", "text_by_page_src": [f"Page {i}"]}
            for i in range(3)
        ]
        input_file = input_dir / "shard0001_nupunkt.jsonl"
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        # Create progress files with first 2 books already processed
        complete_progress = output_dir / "shard0001.complete.progress.jsonl"
        incomplete_progress = output_dir / "shard0001.incomplete.progress.jsonl"

        progress_records = [
            {
                "barcode_src": "book0",
                "_processing_complete": True,
                "_last_completed_step": "step10_chunk",
            },
            {
                "barcode_src": "book1",
                "_processing_complete": True,
                "_last_completed_step": "step10_chunk",
            },
        ]
        complete_progress.write_text(
            "\n".join(json.dumps(r) for r in progress_records) + "\n"
        )
        incomplete_progress.write_text("")  # Empty incomplete file

        runner = CliRunner()
        result = runner.invoke(
            process_shard_main,
            [
                "--shard-id", "0001",
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--segmenter", "nupunkt",
                "--resume",
            ],
        )

        assert result.exit_code == 0

        # Check output file exists and has all 3 books
        complete_file = output_dir / "shard0001.complete.jsonl"
        assert complete_file.exists()

        output_books = [
            json.loads(line) for line in complete_file.read_text().strip().split("\n")
        ]
        assert len(output_books) == 3

        # Progress file should be cleaned up
        assert not complete_progress.exists()

    @patch("commands.process_shard.import_step_function")
    def test_progress_file_cleaned_up_on_success(self, mock_import_step, tmp_path: Path):
        """Test that progress file is removed after successful completion."""
        mock_import_step.return_value = lambda book, config, seg: book

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        book = {"barcode_src": "book1", "text_by_page_src": ["Page 1"]}
        input_file = input_dir / "shard0001_nupunkt.jsonl"
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            process_shard_main,
            [
                "--shard-id", "0001",
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--segmenter", "nupunkt",
            ],
        )

        assert result.exit_code == 0

        # Output should exist
        complete_file = output_dir / "shard0001.complete.jsonl"
        assert complete_file.exists()

        # Progress file should not exist after successful completion
        progress_file = output_dir / "shard0001.complete.progress.jsonl"
        assert not progress_file.exists()

    @patch("commands.process_shard.import_step_function")
    def test_fresh_start_removes_old_progress(self, mock_import_step, tmp_path: Path):
        """Test that without --resume, old progress files are removed."""
        mock_import_step.return_value = lambda book, config, seg: book

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create input with 1 book
        book = {"barcode_src": "book0", "text_by_page_src": ["Page 0"]}
        input_file = input_dir / "shard0001_nupunkt.jsonl"
        input_file.write_text(json.dumps(book))

        # Create old progress file with stale data
        complete_progress = output_dir / "shard0001.complete.progress.jsonl"
        complete_progress.write_text(
            json.dumps({"barcode_src": "old_book", "_processing_complete": True}) + "\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            process_shard_main,
            [
                "--shard-id", "0001",
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--segmenter", "nupunkt",
                # No --resume flag
            ],
        )

        assert result.exit_code == 0

        # Check output only has the new book, not the old one
        complete_file = output_dir / "shard0001.complete.jsonl"
        output_books = [
            json.loads(line) for line in complete_file.read_text().strip().split("\n")
        ]
        assert len(output_books) == 1
        assert output_books[0]["barcode_src"] == "book0"
