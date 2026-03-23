"""Tests for commands/dedup_find_duplicates.py CLI."""

import json
from pathlib import Path

from click.testing import CliRunner

from commands.dedup_find_duplicates import main as dedup_find_main


class TestDedupFindDuplicatesCLI:
    def test_processes_simhash_files(self, tmp_path: Path):
        """Test that CLI reads simhash files and writes clusters JSON."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        # Create simhash files
        simhash1 = {"book_id": "book1", "simhashes": [12345, 67890]}
        simhash2 = {"book_id": "book2", "simhashes": [11111, 22222]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_data = json.loads(output_file.read_text())
        assert "clusters" in output_data
        assert "statistics" in output_data

    def test_output_format(self, tmp_path: Path):
        """Test that output has correct structure."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        simhash = {"book_id": "book1", "simhashes": [12345]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_data = json.loads(output_file.read_text())
        stats = output_data["statistics"]
        assert "total_records" in stats
        assert "duplicate_pairs" in stats
        assert "clusters" in stats

    def test_finds_exact_duplicates(self, tmp_path: Path):
        """Test that identical simhashes are detected as duplicates."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        # Same hash = exact duplicate
        identical_hash = 123456789012345678901234567890
        simhash1 = {"book_id": "book1", "simhashes": [identical_hash]}
        simhash2 = {"book_id": "book2", "simhashes": [identical_hash]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_data = json.loads(output_file.read_text())
        # Should have 1 cluster with 2 members
        assert output_data["statistics"]["clusters"] == 1
        assert output_data["statistics"]["duplicate_pairs"] == 1

    def test_custom_threshold(self, tmp_path: Path):
        """Test that threshold parameter affects duplicate detection."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        # Hashes that differ by a small number of bits
        simhash1 = {"book_id": "book1", "simhashes": [0b1111111111111111]}
        simhash2 = {"book_id": "book2", "simhashes": [0b1111111111111110]}  # 1 bit diff
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        # With threshold=0, they should NOT be duplicates
        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            [
                "--input-dir",
                str(input_dir),
                "--output-file",
                str(output_file),
                "--threshold",
                "0",
            ],
        )

        assert result.exit_code == 0
        output_data = json.loads(output_file.read_text())
        assert output_data["statistics"]["duplicate_pairs"] == 0

    def test_expands_doc_ids_correctly(self, tmp_path: Path):
        """Test that doc_ids are expanded as book_id.index."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        identical_hash = 999999999
        # Book with multiple paragraphs, some duplicated across books
        simhash1 = {"book_id": "bookA", "simhashes": [identical_hash, 111]}
        simhash2 = {"book_id": "bookB", "simhashes": [identical_hash, 222]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_data = json.loads(output_file.read_text())
        # Check doc_ids in clusters use book_id.index format
        clusters = output_data["clusters"]
        assert len(clusters) >= 1  # At least one cluster for the identical hashes
        for rep, members in clusters.items():
            assert "." in rep  # Rep should be book_id.index format
            assert all("." in m for m in members)

    def test_empty_input_directory(self, tmp_path: Path):
        """Test that CLI fails gracefully with empty input directory."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_file)],
        )

        assert result.exit_code != 0  # Should fail with no files

    def test_fails_on_missing_input_dir(self, tmp_path: Path):
        """Test that CLI fails when input directory doesn't exist."""
        input_dir = tmp_path / "nonexistent"
        output_file = tmp_path / "clusters.json"

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_file)],
        )

        assert result.exit_code != 0

    def test_streaming_mode_processes_files(self, tmp_path: Path):
        """Test that streaming mode reads simhash files and writes clusters JSON."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        simhash1 = {"book_id": "book1", "simhashes": [12345, 67890]}
        simhash2 = {"book_id": "book2", "simhashes": [11111, 22222]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            [
                "--input-dir", str(input_dir),
                "--output-file", str(output_file),
                "--streaming",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        output_data = json.loads(output_file.read_text())
        assert "clusters" in output_data
        assert "statistics" in output_data

    def test_streaming_mode_finds_duplicates(self, tmp_path: Path):
        """Test that streaming mode correctly identifies duplicate hashes."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        # Same hash = exact duplicate
        identical_hash = 123456789012345678901234567890
        simhash1 = {"book_id": "book1", "simhashes": [identical_hash]}
        simhash2 = {"book_id": "book2", "simhashes": [identical_hash]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            [
                "--input-dir", str(input_dir),
                "--output-file", str(output_file),
                "--streaming",
            ],
        )

        assert result.exit_code == 0

        output_data = json.loads(output_file.read_text())
        # Should have 1 cluster with 2 members
        assert output_data["statistics"]["clusters"] == 1
        assert output_data["statistics"]["duplicate_pairs"] == 1

    def test_streaming_mode_matches_default_mode(self, tmp_path: Path):
        """Test that streaming mode produces same results as default mode."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_default = tmp_path / "clusters_default.json"
        output_streaming = tmp_path / "clusters_streaming.json"

        # Create test data with some duplicates
        identical_hash = 999999999999
        simhash1 = {"book_id": "book1", "simhashes": [identical_hash, 111, 222]}
        simhash2 = {"book_id": "book2", "simhashes": [identical_hash, 333, 444]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        runner = CliRunner()

        # Run default mode
        result_default = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_default)],
        )
        assert result_default.exit_code == 0

        # Run streaming mode
        result_streaming = runner.invoke(
            dedup_find_main,
            [
                "--input-dir", str(input_dir),
                "--output-file", str(output_streaming),
                "--streaming",
            ],
        )
        assert result_streaming.exit_code == 0

        # Compare results
        data_default = json.loads(output_default.read_text())
        data_streaming = json.loads(output_streaming.read_text())

        assert data_default["statistics"] == data_streaming["statistics"]
        # Compare cluster contents (representatives may differ due to processing order)
        default_clusters = sorted(sorted(v) for v in data_default["clusters"].values())
        streaming_clusters = sorted(sorted(v) for v in data_streaming["clusters"].values())
        assert default_clusters == streaming_clusters
