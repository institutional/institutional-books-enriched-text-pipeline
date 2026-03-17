"""Tests for commands/step11_compute_perplexity.py CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from commands.step11_compute_perplexity import main as step11_main


class TestStep11CLI:
    def test_exits_successfully_when_disabled(self, tmp_path: Path):
        """Test CLI exits with success when perplexity is disabled (default)."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step11_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        # Log output goes to stderr, not stdout
        assert not output_file.exists()

    @patch("commands.step11_compute_perplexity.load_perplexity_model")
    @patch("commands.step11_compute_perplexity.compute_perplexities_in_book")
    def test_processes_books_when_enabled(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test CLI processes books when perplexity is enabled."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.return_value = {
            "book_id": "book1",
            "perplexities": [10.5, 20.3],
        }

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        config_file = tmp_path / "config.yaml"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Sentence one.", "Sentence two."],
            "subtopic_paragraph_start_indices": [0, 1],
        }
        input_file.write_text(json.dumps(book))
        config_file.write_text("perplexity:\n  enabled: true\n")

        runner = CliRunner()
        result = runner.invoke(
            step11_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--config-file", str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_records = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_records) == 1
        assert output_records[0]["book_id"] == "book1"
        assert output_records[0]["perplexities"] == [10.5, 20.3]

    @patch("commands.step11_compute_perplexity.load_perplexity_model")
    @patch("commands.step11_compute_perplexity.compute_perplexities_in_book")
    def test_output_format_matches_expected_structure(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test output JSONL format matches expected structure."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.side_effect = [
            {"book_id": "book1", "perplexities": [12.5, 45.2, 8.7]},
            {"book_id": "book2", "perplexities": [5.0]},
        ]

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        config_file = tmp_path / "config.yaml"

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
        config_file.write_text("perplexity:\n  enabled: true\n")

        runner = CliRunner()
        result = runner.invoke(
            step11_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--config-file", str(config_file),
            ],
        )

        assert result.exit_code == 0

        output_records = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_records) == 2

        # Check structure
        for record in output_records:
            assert "book_id" in record
            assert "perplexities" in record
            assert isinstance(record["perplexities"], list)

        assert output_records[0]["book_id"] == "book1"
        assert output_records[0]["perplexities"] == [12.5, 45.2, 8.7]
        assert output_records[1]["book_id"] == "book2"
        assert output_records[1]["perplexities"] == [5.0]

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            step11_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code != 0

    @patch("commands.step11_compute_perplexity.load_perplexity_model")
    @patch("commands.step11_compute_perplexity.compute_perplexities_in_book")
    def test_skips_books_with_errors(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test that books with errors are skipped gracefully."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.side_effect = [
            ValueError("No sentences found"),
            {"book_id": "book2", "perplexities": [5.0]},
        ]

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        config_file = tmp_path / "config.yaml"

        books = [
            {"barcode_src": "book1"},  # Missing required fields
            {
                "barcode_src": "book2",
                "middlematter_sentences": ["X."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))
        config_file.write_text("perplexity:\n  enabled: true\n")

        runner = CliRunner()
        result = runner.invoke(
            step11_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--config-file", str(config_file),
            ],
        )

        assert result.exit_code == 0

        output_records = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_records) == 1
        assert output_records[0]["book_id"] == "book2"

    @patch("commands.step11_compute_perplexity.load_perplexity_model")
    @patch("commands.step11_compute_perplexity.compute_perplexities_in_book")
    def test_creates_output_directory(
        self, mock_compute, mock_load_model, tmp_path: Path
    ):
        """Test that output directory is created if it doesn't exist."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_compute.return_value = {"book_id": "book1", "perplexities": [1.0]}

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "nested" / "dir" / "output.jsonl"
        config_file = tmp_path / "config.yaml"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))
        config_file.write_text("perplexity:\n  enabled: true\n")

        runner = CliRunner()
        result = runner.invoke(
            step11_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--config-file", str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    @patch("commands.step11_compute_perplexity.load_perplexity_model")
    def test_uses_custom_model_from_config(self, mock_load_model, tmp_path: Path):
        """Test that custom model name from config is used."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        config_file = tmp_path / "config.yaml"

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Test."],
            "subtopic_paragraph_start_indices": [0],
        }
        input_file.write_text(json.dumps(book))
        config_file.write_text(
            "perplexity:\n  enabled: true\n  model_name: custom/model-name\n"
        )

        with patch("commands.step11_compute_perplexity.compute_perplexities_in_book") as mock_compute:
            mock_compute.return_value = {"book_id": "book1", "perplexities": [1.0]}

            runner = CliRunner()
            runner.invoke(
                step11_main,
                [
                    "--input-file", str(input_file),
                    "--output-file", str(output_file),
                    "--config-file", str(config_file),
                ],
            )

            mock_load_model.assert_called_once()
            call_args = mock_load_model.call_args
            assert call_args[0][0] == "custom/model-name"
