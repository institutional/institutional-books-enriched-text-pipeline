"""
Tests for commands/dedup_annotate.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.dedup_annotate import main as dedup_annotate_main
from commands.dedup_build_lookup import main as dedup_build_lookup_main
from commands.dedup_compute_simhashes import main as dedup_compute_main
from commands.dedup_find_duplicates import main as dedup_find_main


def write_lookup(path: Path, entries: list[tuple[str, str]]) -> None:
    """Write a per-shard lookup sidecar (TSV of member -> representative)."""
    path.write_text("".join(f"{member}\t{rep}\n" for member, rep in entries))


class TestDedupAnnotateCLI:
    def test_annotates_books(self, tmp_path: Path):
        """Test that CLI reads shard and lookup, writes annotated books."""
        shard_file = tmp_path / "shard.complete.jsonl"
        lookup_file = tmp_path / "shard.lookup.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["Para 1.", "Para 2."],
                "subtopic_paragraph_start_indices": [0, 1],
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        write_lookup(lookup_file, [])

        runner = CliRunner()
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code == 0
        assert shard_file.exists()

        output_book = json.loads(shard_file.read_text().strip())
        assert output_book["barcode_src"] == "book1"

    def test_marks_duplicates(self, tmp_path: Path):
        """Test that duplicate paragraphs are marked correctly."""
        shard_file = tmp_path / "shard.complete.jsonl"
        lookup_file = tmp_path / "shard.lookup.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["Content."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))

        # book1.0 is a duplicate of book2.5 (representative)
        write_lookup(lookup_file, [("book1.0", "book2.5")])

        runner = CliRunner()
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code == 0

        output_book = json.loads(shard_file.read_text().strip())
        assert "duplicate_paragraphs" in output_book
        assert output_book["duplicate_paragraphs"]["0"] == "book2:5"

    def test_marks_representatives(self, tmp_path: Path):
        """Test that cluster representatives are marked correctly."""
        shard_file = tmp_path / "shard.complete.jsonl"
        lookup_file = tmp_path / "shard.lookup.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["Content."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))

        # book1.0 is the representative of its cluster
        write_lookup(lookup_file, [("book1.0", "book1.0")])

        runner = CliRunner()
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code == 0

        output_book = json.loads(shard_file.read_text().strip())
        assert "representative_paragraphs" in output_book
        assert output_book["representative_paragraphs"]["0"] is True

    def test_preserves_other_fields(self, tmp_path: Path):
        """Test that other book fields are preserved."""
        shard_file = tmp_path / "shard.complete.jsonl"
        lookup_file = tmp_path / "shard.lookup.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "title": "Test Book",
                "language_gen": "eng",
                "middlematter_sentences": ["Content."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        write_lookup(lookup_file, [])

        runner = CliRunner()
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code == 0

        output_book = json.loads(shard_file.read_text().strip())
        assert output_book["title"] == "Test Book"
        assert output_book["language_gen"] == "eng"

    def test_processes_multiple_books(self, tmp_path: Path):
        """Test that multiple books are processed correctly."""
        shard_file = tmp_path / "shard.complete.jsonl"
        lookup_file = tmp_path / "shard.lookup.jsonl"

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
            {
                "barcode_src": "book3",
                "middlematter_sentences": ["C."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        write_lookup(lookup_file, [])

        runner = CliRunner()
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code == 0

        output_books = [json.loads(line) for line in shard_file.read_text().strip().split("\n")]
        assert len(output_books) == 3
        assert output_books[0]["barcode_src"] == "book1"
        assert output_books[1]["barcode_src"] == "book2"
        assert output_books[2]["barcode_src"] == "book3"

    def test_overwrites_shard_file(self, tmp_path: Path):
        """Test that shard file is overwritten (not appended to)."""
        shard_file = tmp_path / "shard.complete.jsonl"
        lookup_file = tmp_path / "shard.lookup.jsonl"

        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": ["Content."],
                "subtopic_paragraph_start_indices": [0],
            },
        ]
        shard_file.write_text("\n".join(json.dumps(b) for b in books))
        write_lookup(lookup_file, [])

        runner = CliRunner()
        # Run twice
        runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code == 0

        # Should still have only 1 book (overwrite, not append)
        output_books = [json.loads(line) for line in shard_file.read_text().strip().split("\n")]
        assert len(output_books) == 1

    def test_fails_on_missing_shard_file(self, tmp_path: Path):
        """Test that CLI fails when shard file doesn't exist."""
        shard_file = tmp_path / "nonexistent.jsonl"
        lookup_file = tmp_path / "shard.lookup.jsonl"
        write_lookup(lookup_file, [])

        runner = CliRunner()
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code != 0

    def test_fails_on_missing_lookup_file(self, tmp_path: Path):
        """Test that CLI fails when lookup file doesn't exist."""
        shard_file = tmp_path / "shard.complete.jsonl"
        lookup_file = tmp_path / "nonexistent.lookup.jsonl"

        books = [{"barcode_src": "book1", "subtopic_paragraph_start_indices": [0]}]
        shard_file.write_text(json.dumps(books[0]))

        runner = CliRunner()
        result = runner.invoke(
            dedup_annotate_main,
            ["--shard-file", str(shard_file), "--lookup-file", str(lookup_file)],
        )

        assert result.exit_code != 0


class TestDedupEndToEnd:
    def test_preserves_paragraph_count_through_dedup_workflow(self, tmp_path: Path):
        """Test that N paragraph chunks in input results in N paragraph chunks in output."""
        # Setup directories
        shard_file = tmp_path / "shard.complete.jsonl"
        simhash_dir = tmp_path / "simhashes"
        simhash_dir.mkdir()
        simhash_file = simhash_dir / "shard.simhashes.jsonl"
        clusters_file = tmp_path / "clusters.json"
        lookup_dir = tmp_path / "lookups"

        # Create books with known paragraph counts
        books = [
            {
                "barcode_src": "book1",
                "middlematter_sentences": [
                    "First para sentence one.",
                    "First para sentence two.",
                    "Second para content.",
                    "Third para here.",
                    "Fourth paragraph text.",
                ],
                "subtopic_paragraph_start_indices": [0, 2, 3, 4],  # 4 paragraphs
            },
            {
                "barcode_src": "book2",
                "middlematter_sentences": [
                    "Book two para one.",
                    "Book two para two.",
                    "Book two para three.",
                ],
                "subtopic_paragraph_start_indices": [0, 1, 2],  # 3 paragraphs
            },
        ]
        original_para_counts = {
            "book1": len(books[0]["subtopic_paragraph_start_indices"]),
            "book2": len(books[1]["subtopic_paragraph_start_indices"]),
        }
        shard_file.write_text("\n".join(json.dumps(b) for b in books))

        runner = CliRunner()

        # Phase 1: Compute simhashes
        result = runner.invoke(
            dedup_compute_main,
            ["--input-file", str(shard_file), "--output-file", str(simhash_file)],
        )
        assert result.exit_code == 0

        # Phase 2: Find duplicates
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(simhash_dir), "--output-file", str(clusters_file)],
        )
        assert result.exit_code == 0

        # Phase 3: Build per-shard lookup sidecars
        result = runner.invoke(
            dedup_build_lookup_main,
            [
                "--shard-dir",
                str(tmp_path),
                "--clusters-file",
                str(clusters_file),
                "--output-dir",
                str(lookup_dir),
            ],
        )
        assert result.exit_code == 0

        # Phase 4: Annotate
        result = runner.invoke(
            dedup_annotate_main,
            [
                "--shard-file",
                str(shard_file),
                "--lookup-file",
                str(lookup_dir / "shard.lookup.jsonl"),
            ],
        )
        assert result.exit_code == 0

        # Verify paragraph counts are preserved
        output_books = [json.loads(line) for line in shard_file.read_text().strip().split("\n")]
        assert len(output_books) == 2

        for book in output_books:
            book_id = book["barcode_src"]
            output_para_count = len(book["subtopic_paragraph_start_indices"])
            assert output_para_count == original_para_counts[book_id], (
                f"{book_id}: expected {original_para_counts[book_id]} paragraphs, "
                f"got {output_para_count}"
            )

        # Also verify sentences are preserved
        assert len(output_books[0]["middlematter_sentences"]) == 5
        assert len(output_books[1]["middlematter_sentences"]) == 3
