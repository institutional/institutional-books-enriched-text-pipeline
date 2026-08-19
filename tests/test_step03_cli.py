"""
Tests for commands/step03_separate_frontmatter_backmatter.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from commands.steps.step03_separate_frontmatter_backmatter import main as step03_main


class MockClassifier:
    """Mock classifier that labels pages based on content."""

    def predict(self, texts: list[str]) -> list[str]:
        results = []
        for text in texts:
            if "frontmatter" in text.lower() or "preface" in text.lower():
                results.append("ENDMATTER")
            elif "index" in text.lower() or "bibliography" in text.lower():
                results.append("ENDMATTER")
            else:
                results.append("MIDDLEMATTER")
        return results


class TestStep03CLI:
    def test_processes_jsonl_file(self, tmp_path: Path):
        """Test that step03 CLI reads input JSONL and writes output JSONL."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "uniformized_text": [
                    "Preface content here.",
                    "Chapter 1 main content.",
                    "Chapter 2 more content.",
                    "Index and bibliography.",
                ],
            },
        ]
        input_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()
        with patch(
            "commands.steps.step03_separate_frontmatter_backmatter.get_classifier",
            return_value=MockClassifier(),
        ):
            result = runner.invoke(
                step03_main,
                ["--input-file", str(input_file), "--output-file", str(output_file)],
            )

        assert result.exit_code == 0
        assert output_file.exists()

        output_book = json.loads(output_file.read_text().strip())
        assert "frontmatter" in output_book
        assert "middlematter" in output_book
        assert "backmatter" in output_book
        assert "uniformized_text" not in output_book

    def test_separates_content_correctly(self, tmp_path: Path):
        """Test that frontmatter/middlematter/backmatter are separated."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        book = {
            "barcode_src": "test",
            "uniformized_text": [
                "Preface page.",
                "Main chapter one.",
                "Main chapter two.",
                "Index page.",
            ],
        }
        input_file.write_text(json.dumps(book))

        runner = CliRunner()
        with patch(
            "commands.steps.step03_separate_frontmatter_backmatter.get_classifier",
            return_value=MockClassifier(),
        ):
            result = runner.invoke(
                step03_main,
                ["--input-file", str(input_file), "--output-file", str(output_file)],
            )

        assert result.exit_code == 0
        output_book = json.loads(output_file.read_text().strip())

        assert output_book["frontmatter"] == ["Preface page."]
        assert output_book["middlematter"] == ["Main chapter one.", "Main chapter two."]
        assert output_book["backmatter"] == ["Index page."]
