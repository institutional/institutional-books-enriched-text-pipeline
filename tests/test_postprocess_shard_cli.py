"""Tests for commands/postprocess_shard.py CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from commands.postprocess_shard import main as postprocess_main


class TestPostprocessShardCLI:
    @patch("commands.postprocess_shard.load_em_subclassifier")
    @patch("commands.postprocess_shard.compute_text_stats")
    def test_processes_shard(self, mock_text_stats, mock_load_classifier, tmp_path: Path):
        """Test that postprocess_shard processes books through all steps."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["TOC_INDEX"]
        mock_load_classifier.return_value = mock_classifier

        mock_text_stats.return_value = {
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
                "language_gen": "eng",
                "frontmatter": ["Front page"],
                "middlematter_sentences": ["Sentence one.", "Sentence two."],
                "subtopic_paragraph_start_indices": [0, 1],
                "subtopic_section_start_indices": [0],
                "backmatter": [],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            postprocess_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 1

        book = output_books[0]
        # Should have annotated fields from step13
        assert "annotated_frontmatter" in book
        assert "annotated_middlematter" in book
        assert "annotated_backmatter" in book
        # Should have metadata from step14
        assert "metadata" in book
        # step15 (clean) removes intermediate fields but keeps essential ones
        assert "barcode_src" in book
        assert "language_gen" in book
        # frontmatter is removed by step15 clean
        assert "frontmatter" not in book

    @patch("commands.postprocess_shard.load_em_subclassifier")
    def test_uses_perplexity_file(self, mock_load_classifier, tmp_path: Path):
        """Test that perplexity values are included in annotations when file is provided."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = []
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        perp_file = tmp_path / "perplexity.jsonl"

        book = {
            "barcode_src": "book1",
            "language_gen": "eng",
            "frontmatter": [],
            "middlematter_sentences": ["Content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "backmatter": [],
        }
        input_file.write_text(json.dumps(book))

        perp_record = {"book_id": "book1", "perplexities": [12.5]}
        perp_file.write_text(json.dumps(perp_record))

        runner = CliRunner()
        result = runner.invoke(
            postprocess_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--perplexity-file", str(perp_file),
            ],
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        # Perplexity should be in the annotated middlematter (step13)
        assert 'perplexity="12.5"' in output_book["annotated_middlematter"]

    @patch("commands.postprocess_shard.load_em_subclassifier")
    def test_step_range(self, mock_load_classifier, tmp_path: Path):
        """Test explicitly specifying step range."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = []
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "language_gen": "eng",
            "frontmatter": [],
            "middlematter_sentences": ["Content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "backmatter": [],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        # Explicitly run step13 only
        result = runner.invoke(
            postprocess_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--start-step", "step13_annotate",
                "--end-step", "step13_annotate",
            ],
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        # Should have annotated fields
        assert "annotated_middlematter" in output_book
        # Should still have intermediate fields (no cleaning step)
        assert "frontmatter" in output_book

    @patch("commands.postprocess_shard.load_em_subclassifier")
    def test_annotation_produces_tags(self, mock_load_classifier, tmp_path: Path):
        """Test that annotation produces proper XML-like tags."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["TOC_INDEX"]
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "language_gen": "eng",
            "frontmatter": ["Table of Contents page"],
            "middlematter_sentences": ["Content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "backmatter": [],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            postprocess_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        # Check frontmatter has endmatter tags (now a single string)
        assert '<idi-endmatter type="TOC_INDEX">' in output_book["annotated_frontmatter"]
        # Check middlematter has section/paragraph tags
        assert "<idi-section>" in output_book["annotated_middlematter"]
        assert "<idi-paragraph" in output_book["annotated_middlematter"]

    @patch("commands.postprocess_shard.load_em_subclassifier")
    @patch("commands.postprocess_shard.compute_text_stats")
    def test_processes_multiple_books(self, mock_text_stats, mock_load_classifier, tmp_path: Path):
        """Test that multiple books are processed."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = []
        mock_load_classifier.return_value = mock_classifier
        mock_text_stats.return_value = {"token_count": 50}

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": f"book{i}",
                "language_gen": "eng",
                "frontmatter": [],
                "middlematter_sentences": ["Content."],
                "subtopic_paragraph_start_indices": [0],
                "subtopic_section_start_indices": [0],
                "backmatter": [],
            }
            for i in range(3)
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            postprocess_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 3

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            postprocess_main,
            ["--input-file", str(input_file), "--output-file", str(output_file)],
        )

        assert result.exit_code != 0
