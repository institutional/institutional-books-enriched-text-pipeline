"""
Tests for commands/dedup_mark_removed.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.dedup_mark_removed import main as dedup_mark_removed_main


class TestDedupMarkRemovedCLI:
    def test_marks_duplicate_for_removal(self, tmp_path: Path):
        """Test that a duplicate paragraph referencing a removed cluster is marked."""
        shard_file = tmp_path / "shard.jsonl"
        removal_file = tmp_path / "removed.json"

        books = [
            {
                "barcode_src": "book1",
                "subtopic_paragraph_start_indices": [0, 1],
                "duplicate_paragraphs": {"0": "bookX:5"},
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        removal_file.write_text(json.dumps(["bookX:5"]))

        runner = CliRunner()
        result = runner.invoke(
            dedup_mark_removed_main,
            ["--shard-file", str(shard_file), "--removal-file", str(removal_file)],
        )

        assert result.exit_code == 0
        output_book = json.loads(shard_file.read_text().strip())
        assert "removed_paragraphs" in output_book
        assert output_book["removed_paragraphs"]["0"] == "removed_cluster"

    def test_marks_representative_for_removal(self, tmp_path: Path):
        """Test that a representative paragraph of a removed cluster is marked."""
        shard_file = tmp_path / "shard.jsonl"
        removal_file = tmp_path / "removed.json"

        books = [
            {
                "barcode_src": "bookX",
                "subtopic_paragraph_start_indices": [0, 1],
                "representative_paragraphs": {"0": True},
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        removal_file.write_text(json.dumps(["bookX:0"]))

        runner = CliRunner()
        result = runner.invoke(
            dedup_mark_removed_main,
            ["--shard-file", str(shard_file), "--removal-file", str(removal_file)],
        )

        assert result.exit_code == 0
        output_book = json.loads(shard_file.read_text().strip())
        assert "removed_paragraphs" in output_book
        assert output_book["removed_paragraphs"]["0"] == "removed_cluster"

    def test_no_removal_when_cluster_not_listed(self, tmp_path: Path):
        """Test that paragraphs are not marked when their cluster is not in the removal list."""
        shard_file = tmp_path / "shard.jsonl"
        removal_file = tmp_path / "removed.json"

        books = [
            {
                "barcode_src": "book1",
                "subtopic_paragraph_start_indices": [0],
                "duplicate_paragraphs": {"0": "bookY:3"},
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        removal_file.write_text(json.dumps(["bookX:5"]))

        runner = CliRunner()
        result = runner.invoke(
            dedup_mark_removed_main,
            ["--shard-file", str(shard_file), "--removal-file", str(removal_file)],
        )

        assert result.exit_code == 0
        output_book = json.loads(shard_file.read_text().strip())
        assert "removed_paragraphs" not in output_book

    def test_preserves_other_fields(self, tmp_path: Path):
        """Test that other book fields are preserved."""
        shard_file = tmp_path / "shard.jsonl"
        removal_file = tmp_path / "removed.json"

        books = [
            {
                "barcode_src": "book1",
                "title": "Test Book",
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        removal_file.write_text(json.dumps([]))

        runner = CliRunner()
        result = runner.invoke(
            dedup_mark_removed_main,
            ["--shard-file", str(shard_file), "--removal-file", str(removal_file)],
        )

        assert result.exit_code == 0
        output_book = json.loads(shard_file.read_text().strip())
        assert output_book["title"] == "Test Book"


class TestRemovedParagraphsInAnnotation:
    """Test that removed paragraphs are skipped during annotation."""

    def test_removed_paragraph_not_in_output(self):
        from library.annotate.middlematter import annotate_middlematter

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Keep this.", "Remove this.", "Keep this too."],
            "subtopic_paragraph_start_indices": [0, 1, 2],
            "subtopic_section_start_indices": [0],
            "removed_paragraphs": {"1": "removed_cluster"},
        }
        result, _ = annotate_middlematter(book)
        assert "Keep this." in result
        assert "Keep this too." in result
        assert "Remove this." not in result

    def test_removed_duplicate_not_in_output(self):
        from library.annotate.middlematter import annotate_middlematter

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Regular.", "Dup removed.", "Also regular."],
            "subtopic_paragraph_start_indices": [0, 1, 2],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"1": "bookX:5"},
            "removed_paragraphs": {"1": "removed_cluster"},
        }
        result, _ = annotate_middlematter(book)
        assert "Regular." in result
        assert "Also regular." in result
        assert "Dup removed." not in result
        assert "bookX:5" not in result

    def test_removed_representative_not_in_output(self):
        from library.annotate.middlematter import annotate_middlematter

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Regular.", "Rep removed.", "Also regular."],
            "subtopic_paragraph_start_indices": [0, 1, 2],
            "subtopic_section_start_indices": [0],
            "representative_paragraphs": {"1": True},
            "removed_paragraphs": {"1": "removed_cluster"},
        }
        result, _ = annotate_middlematter(book)
        assert "Regular." in result
        assert "Also regular." in result
        assert "Rep removed." not in result
        assert "data-representative" not in result

    def test_removed_breaks_consecutive_duplicate_merging(self):
        """If a removed paragraph sits between two duplicates of the same cluster,
        they should not be merged."""
        from library.annotate.middlematter import annotate_middlematter

        book = {
            "barcode_src": "book1",
            "middlematter_sentences": ["Dup1.", "Removed.", "Dup3."],
            "subtopic_paragraph_start_indices": [0, 1, 2],
            "subtopic_section_start_indices": [0],
            "duplicate_paragraphs": {"0": "bookX:5", "1": "bookX:6", "2": "bookX:7"},
            "removed_paragraphs": {"1": "removed_cluster"},
        }
        result, _ = annotate_middlematter(book)
        assert "Removed." not in result
        # Para 0 and 2 should be in separate aside tags since para 1 broke the sequence
        assert result.count("<aside") == 2
        assert 'data-cluster="bookX:5"' in result
        assert 'data-cluster="bookX:7"' in result
