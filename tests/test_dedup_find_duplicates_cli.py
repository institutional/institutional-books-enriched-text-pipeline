"""Tests for commands/dedup_find_duplicates.py CLI.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from commands.dedup_find_duplicates import (
    _buckets_from_sorted_keys,
    _extract_band_values,
)
from commands.dedup_find_duplicates import (
    main as dedup_find_main,
)
from utils.simhash import BAND_BITS, NUM_BANDS, extract_bands


class TestExtractBandValues:
    """Tests for vectorized numpy band extraction."""

    def test_matches_scalar_extract_bands(self):
        """Numpy vectorized extraction must match scalar pure-Python for random hashes."""
        rng = np.random.default_rng(42)
        n = 1000
        hashes_lo = rng.integers(0, 2**64, size=n, dtype=np.uint64)
        hashes_hi = rng.integers(0, 2**64, size=n, dtype=np.uint64)

        for band_idx in range(NUM_BANDS):
            band_values = _extract_band_values(hashes_lo, hashes_hi, band_idx)
            for i in range(n):
                full_hash = int(hashes_lo[i]) | (int(hashes_hi[i]) << 64)
                expected = extract_bands(full_hash)[band_idx]
                assert int(band_values[i]) == expected, (
                    f"Mismatch at record {i}, band {band_idx}"
                )

    def test_zero_hash(self):
        """All-zero hash produces zero for every band."""
        hashes_lo = np.zeros(1, dtype=np.uint64)
        hashes_hi = np.zeros(1, dtype=np.uint64)
        for band_idx in range(NUM_BANDS):
            result = _extract_band_values(hashes_lo, hashes_hi, band_idx)
            assert int(result[0]) == 0

    def test_max_hash(self):
        """All-ones hash produces max value for each band."""
        hashes_lo = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
        hashes_hi = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
        for band_idx in range(NUM_BANDS):
            result = _extract_band_values(hashes_lo, hashes_hi, band_idx)
            expected_max = (1 << BAND_BITS[band_idx]) - 1
            assert int(result[0]) == expected_max

    def test_lo_hi_boundary(self):
        """Bands 0-2 use hashes_lo only, bands 3-5 use hashes_hi only."""
        # Set lo=max, hi=0: bands from lo should be nonzero, bands from hi should be zero
        hashes_lo = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
        hashes_hi = np.array([0], dtype=np.uint64)

        full_hash = int(hashes_lo[0]) | (int(hashes_hi[0]) << 64)
        for band_idx in range(NUM_BANDS):
            result = int(_extract_band_values(hashes_lo, hashes_hi, band_idx)[0])
            expected = extract_bands(full_hash)[band_idx]
            assert result == expected, f"Band {band_idx}: got {result}, expected {expected}"

        # Verify: lo-only bands are nonzero, hi-only bands are zero
        assert int(_extract_band_values(hashes_lo, hashes_hi, 0)[0]) != 0
        assert int(_extract_band_values(hashes_lo, hashes_hi, 5)[0]) == 0

    def test_output_dtype(self):
        """Band values should be uint32."""
        hashes_lo = np.array([123456789], dtype=np.uint64)
        hashes_hi = np.array([987654321], dtype=np.uint64)
        for band_idx in range(NUM_BANDS):
            result = _extract_band_values(hashes_lo, hashes_hi, band_idx)
            assert result.dtype == np.uint32


class TestBucketsFromSortedKeys:
    """Tests for bucket extraction from sorted packed keys."""

    def test_single_bucket(self):
        """All records with same band_value form one bucket."""
        band_value = np.uint64(42)
        keys = np.array(
            [(band_value << np.uint64(32)) | np.uint64(i) for i in range(5)],
            dtype=np.uint64,
        )
        buckets = list(_buckets_from_sorted_keys(keys, len(keys)))
        assert len(buckets) == 1
        assert sorted(buckets[0]) == [0, 1, 2, 3, 4]

    def test_multiple_buckets(self):
        """Different band_values produce separate buckets."""
        keys = np.array(
            [
                (np.uint64(10) << np.uint64(32)) | np.uint64(0),
                (np.uint64(10) << np.uint64(32)) | np.uint64(1),
                (np.uint64(20) << np.uint64(32)) | np.uint64(2),
                (np.uint64(20) << np.uint64(32)) | np.uint64(3),
                (np.uint64(20) << np.uint64(32)) | np.uint64(4),
            ],
            dtype=np.uint64,
        )
        buckets = list(_buckets_from_sorted_keys(keys, len(keys)))
        assert len(buckets) == 2
        assert sorted(buckets[0]) == [0, 1]
        assert sorted(buckets[1]) == [2, 3, 4]

    def test_singletons_excluded(self):
        """Buckets with only 1 member are not yielded."""
        keys = np.array(
            [
                (np.uint64(1) << np.uint64(32)) | np.uint64(0),
                (np.uint64(2) << np.uint64(32)) | np.uint64(1),
                (np.uint64(3) << np.uint64(32)) | np.uint64(2),
            ],
            dtype=np.uint64,
        )
        buckets = list(_buckets_from_sorted_keys(keys, len(keys)))
        assert len(buckets) == 0

    def test_empty_input(self):
        """Empty array produces no buckets."""
        keys = np.array([], dtype=np.uint64)
        buckets = list(_buckets_from_sorted_keys(keys, 0))
        assert len(buckets) == 0

    def test_preserves_doc_indices(self):
        """Doc indices in output match what was packed into keys."""
        doc_ids = [100, 200, 300]
        band_value = np.uint64(5)
        keys = np.array(
            [(band_value << np.uint64(32)) | np.uint64(d) for d in doc_ids],
            dtype=np.uint64,
        )
        buckets = list(_buckets_from_sorted_keys(keys, len(keys)))
        assert len(buckets) == 1
        assert sorted(buckets[0]) == sorted(doc_ids)


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

    def test_representative_is_alphabetically_first(self, tmp_path: Path):
        """Test that the cluster representative is the alphabetically first member."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        identical_hash = 123456789012345678901234567890
        # bookZ comes first in file order, but bookA should be representative
        simhash1 = {"book_id": "bookZ", "simhashes": [identical_hash]}
        simhash2 = {"book_id": "bookA", "simhashes": [identical_hash]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash1))
        (input_dir / "shard2.simhashes.jsonl").write_text(json.dumps(simhash2))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            ["--input-dir", str(input_dir), "--output-file", str(output_file)],
        )

        assert result.exit_code == 0

        output_data = json.loads(output_file.read_text())
        clusters = output_data["clusters"]
        assert len(clusters) == 1
        rep = list(clusters.keys())[0]
        assert rep == "bookA.0", f"Expected bookA.0 as representative, got {rep}"
        assert sorted(clusters[rep]) == ["bookA.0", "bookZ.0"]

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

    def test_benchmark_flag(self, tmp_path: Path):
        """Test that benchmark flag works without errors."""
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
                "--benchmark",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        # Benchmark timing output goes to stderr via loguru (visible in test logs)

    def test_workers_parameter(self, tmp_path: Path):
        """Test that workers parameter is accepted."""
        input_dir = tmp_path / "simhashes"
        input_dir.mkdir()
        output_file = tmp_path / "clusters.json"

        simhash = {"book_id": "book1", "simhashes": [12345]}
        (input_dir / "shard1.simhashes.jsonl").write_text(json.dumps(simhash))

        runner = CliRunner()
        result = runner.invoke(
            dedup_find_main,
            [
                "--input-dir", str(input_dir),
                "--output-file", str(output_file),
                "--workers", "2",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
