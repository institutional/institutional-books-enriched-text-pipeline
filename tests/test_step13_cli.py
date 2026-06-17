"""Tests for commands/step13_annotate.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from commands.steps.step13_annotate import main as step13_main


class TestStep13CLI:
    @patch("commands.steps.step13_annotate.load_em_subclassifier")
    def test_processes_jsonl_file(self, mock_load_classifier, tmp_path: Path):
        """Test that step13 CLI reads input JSONL and writes output JSONL."""
        # Mock classifier to return TOC_INDEX for all pages
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["TOC_INDEX"]
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "frontmatter": ["Front page content"],
                "middlematter_sentences": ["Sentence one.", "Sentence two."],
                "subtopic_paragraph_start_indices": [0],
                "subtopic_section_start_indices": [0],
                "backmatter": [],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step13_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 1
        assert "annotated_frontmatter" in output_books[0]
        assert "annotated_middlematter" in output_books[0]
        assert "annotated_backmatter" in output_books[0]

    @patch("commands.steps.step13_annotate.load_em_subclassifier")
    def test_annotates_frontmatter(self, mock_load_classifier, tmp_path: Path):
        """Test that frontmatter pages are annotated with endmatter tags."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["TOC_INDEX", "TOC_INDEX"]
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "frontmatter": ["Page 1", "Page 2"],
            "middlematter_sentences": ["Content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "backmatter": [],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step13_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        # annotated_frontmatter is now a single string with pages joined by newlines
        assert "<header>" in output_book["annotated_frontmatter"]
        assert output_book["annotated_frontmatter"].count('<div class="toc_index">') == 2

    @patch("commands.steps.step13_annotate.load_em_subclassifier")
    def test_annotates_backmatter(self, mock_load_classifier, tmp_path: Path):
        """Test that backmatter pages are wrapped in footer tags."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["BIBLIO", "OTHERENDMATTER"]
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "frontmatter": [],
            "middlematter_sentences": ["Content."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "backmatter": ["References page", "Appendix page"],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step13_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert "<footer>" in output_book["annotated_backmatter"]
        assert '<div class="biblio">' in output_book["annotated_backmatter"]
        assert '<div class="otherendmatter">' in output_book["annotated_backmatter"]

    @patch("commands.steps.step13_annotate.load_em_subclassifier")
    def test_annotates_middlematter(self, mock_load_classifier, tmp_path: Path):
        """Test that middlematter is annotated with section/paragraph tags."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = []
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "book1",
            "frontmatter": [],
            "middlematter_sentences": ["First paragraph.", "Second paragraph."],
            "subtopic_paragraph_start_indices": [0, 1],
            "subtopic_section_start_indices": [0],
            "backmatter": [],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        result = runner.invoke(
            step13_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert "<section>" in output_book["annotated_middlematter"]
        assert "<p" in output_book["annotated_middlematter"]

    @patch("commands.steps.step13_annotate.load_em_subclassifier")
    def test_uses_bpb_file(self, mock_load_classifier, tmp_path: Path):
        """Test that BPB values are included when file is provided."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = []
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        bpb_file = tmp_path / "bpb.jsonl"

        book = {
            "barcode_src": "book1",
            "frontmatter": [],
            "middlematter_sentences": ["Content here."],
            "subtopic_paragraph_start_indices": [0],
            "subtopic_section_start_indices": [0],
            "backmatter": [],
        }
        input_file.write_text(json.dumps(book))

        bpb_record = {"book_id": "book1", "bpb_values": [0.85]}
        bpb_file.write_text(json.dumps(bpb_record))

        runner = CliRunner()
        result = runner.invoke(
            step13_main,
            [
                "--input-file", str(input_file),
                "--output-file", str(output_file),
                "--bpb-file", str(bpb_file),
            ],
        )

        assert result.exit_code == 0

        output_book = json.loads(output_file.read_text().strip())
        assert 'data-bpb="0.8500"' in output_book["annotated_middlematter"]

    @patch("commands.steps.step13_annotate.load_em_subclassifier")
    def test_processes_multiple_books(self, mock_load_classifier, tmp_path: Path):
        """Test that multiple books are processed correctly."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = []
        mock_load_classifier.return_value = mock_classifier

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "frontmatter": [],
                "middlematter_sentences": ["A."],
                "subtopic_paragraph_start_indices": [0],
                "subtopic_section_start_indices": [0],
                "backmatter": [],
            },
            {
                "barcode_src": "book2",
                "frontmatter": [],
                "middlematter_sentences": ["B."],
                "subtopic_paragraph_start_indices": [0],
                "subtopic_section_start_indices": [0],
                "backmatter": [],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        result = runner.invoke(
            step13_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code == 0

        output_books = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(output_books) == 2

    def test_fails_on_missing_input_file(self, tmp_path: Path):
        """Test that CLI fails when input file doesn't exist."""
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = tmp_path / "output.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            step13_main, ["--input-file", str(input_file), "--output-file", str(output_file)]
        )

        assert result.exit_code != 0
