"""
dedup_find_duplicates.py - Find duplicate paragraphs across all shards using LSH.

This is phase 2 of the deduplication workflow:
1. Compute simhashes - parallelizable per shard
2. Find duplicates (this step) - requires all simhashes
3. Annotate - parallelizable per shard
"""

from __future__ import annotations

import array
import ctypes
import itertools
import json
import mmap
import os
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import click
from loguru import logger
from tqdm import tqdm

from utils.atomic_write import atomic_write_json
from utils.simhash_fast import extract_bands, hamming_distance
from utils.unionfind import UnionFindInt

if TYPE_CHECKING:
    from concurrent.futures import Future

# Use C++ extension for fast bucket processing when available
_cpp_bucket_module = None
try:
    from extensions.built import _simhash_cpp

    if hasattr(_simhash_cpp, "process_bucket_batch"):
        _cpp_bucket_module = _simhash_cpp
except ImportError:
    pass

# Worker processes shared memory buffers and states
_worker_hashes_mmap: mmap.mmap | None = None
_worker_hashes_array: ctypes.Array[ctypes.c_uint64] | None = None
_worker_hashes_buffer: array.array[int] | None = None  # For C++ (buffer protocol)
_worker_file_handle = None


def _init_worker_mmap(hash_file_path: str, n_hashes: int) -> None:
    """
    Initialize worker with mmap'd hash file.

    Each worker mmaps the same binary file containing all hashes.

    Args:
        hash_file_path: Path to binary file with uint64 hash values
        n_hashes: Number of uint64 values in the file (2 per document)
    """
    global _worker_hashes_mmap, _worker_hashes_array, _worker_hashes_buffer, _worker_file_handle

    _worker_file_handle = open(hash_file_path, "rb")
    _worker_hashes_mmap = mmap.mmap(_worker_file_handle.fileno(), 0, access=mmap.ACCESS_READ)

    if _cpp_bucket_module is not None:
        # For C++: create an array that exposes buffer protocol
        _worker_hashes_buffer = array.array("Q")
        _worker_hashes_buffer.frombytes(_worker_hashes_mmap[:])
    else:
        # For Python: cast mmap to array of uint64 for indexed access
        _worker_hashes_array = (ctypes.c_uint64 * n_hashes).from_buffer_copy(_worker_hashes_mmap)


def _process_bucket_batch_mmap(
    buckets: list[list[int]],
    threshold: int,
    max_bucket_size: int,
) -> list[tuple[int, int]]:
    """
    Process a batch of buckets using mmap'd hashes.

    Args:
        buckets: List of buckets, each bucket is a list of document indices
        threshold: Maximum Hamming distance to consider a duplicate
        max_bucket_size: Skip buckets larger than this

    Returns:
        List of (doc_idx1, doc_idx2) pairs that are duplicates
    """
    if _cpp_bucket_module is not None and _worker_hashes_buffer is not None:
        # Use C++ implementation with buffer protocol
        return list(
            _cpp_bucket_module.process_bucket_batch(  # type: ignore[union-attr]
                buckets, _worker_hashes_buffer, threshold, max_bucket_size
            )
        )

    # Python fallback
    if _worker_hashes_array is None:
        raise RuntimeError("Worker not initialized: _worker_hashes_array is None")

    all_verified: list[tuple[int, int]] = []
    hashes = _worker_hashes_array  # Local reference for type checker
    for doc_indices in buckets:
        if len(doc_indices) > max_bucket_size:
            continue

        doc_indices_sorted = sorted(doc_indices)
        for i, d1 in enumerate(doc_indices_sorted):
            # Read 128-bit hash as two 64-bit values
            h1 = int(hashes[d1 * 2]) | (int(hashes[d1 * 2 + 1]) << 64)
            for d2 in doc_indices_sorted[i + 1 :]:
                h2 = int(hashes[d2 * 2]) | (int(hashes[d2 * 2 + 1]) << 64)
                if hamming_distance(h1, h2) <= threshold:
                    all_verified.append((d1, d2))

    return all_verified


# ============================================================================
# External sort helpers
# ============================================================================


def _write_band_entries(
    records_iter: Iterator[tuple[int, int]],
    output_path: Path,
    total: int | None = None,
) -> None:
    """
    Write band entries to a tab-separated file for external sorting.

    Format: band_idx<TAB>band_value<TAB>doc_idx

    Rationale:
        For large collections, the band entries themselves are too large to
        fit in memory. We write them to disk instead. There is IO overhead
        from using the disk, but this is necessary for large collections.

    Args:
        records_iter: Iterator yielding (doc_idx, hash_value) tuples
        output_path: Path to write the TSV file
        total: Total count for progress bar (optional)
    """
    with open(output_path, "w") as f:
        for doc_idx, hash_val in tqdm(records_iter, total=total, desc="Writing bands"):
            bands = extract_bands(hash_val)
            for band_idx, band_value in enumerate(bands):
                f.write(f"{band_idx}\t{band_value}\t{doc_idx}\n")


def _external_sort(input_path: Path, output_path: Path, temp_dir: Path) -> None:
    """
    Sort band entries file using unix sort (external merge sort).

    Args:
        input_path: Path to unsorted TSV file
        output_path: Path for sorted output
        temp_dir: Directory for sort temp files
    """
    cmd = [
        "sort",
        "-t\t",
        "-k1,1n",  # Sort by band_idx (numeric)
        "-k2,2n",  # Then by band_value (numeric)
        f"-T{temp_dir}",
        "-S",
        "4G",  # Use up to 4GB for sort buffer
        "-o",
        str(output_path),
        str(input_path),
    ]
    logger.info(f"Running external sort: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Sort command failed: {' '.join(cmd)}\nStderr: {result.stderr}")


def _stream_buckets(sorted_path: Path) -> Iterator[tuple[tuple[int, int], list[int]]]:
    """
    Stream through sorted band file, yielding buckets.

    Args:
        sorted_path: Path to sorted TSV file

    Yields:
        (band_key, doc_indices) tuples for buckets with 2+ documents
    """
    current_key: tuple[int, int] | None = None
    current_docs: list[int] = []

    with open(sorted_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            band_idx = int(parts[0])
            band_value = int(parts[1])
            doc_idx = int(parts[2])
            key = (band_idx, band_value)

            if key != current_key:
                if current_key is not None and len(current_docs) >= 2:
                    yield current_key, current_docs
                current_key = key
                current_docs = [doc_idx]
            else:
                current_docs.append(doc_idx)

    # Yield the last bucket
    if current_key is not None and len(current_docs) >= 2:
        yield current_key, current_docs


def find_duplicates(
    simhash_files: list[Path],
    output_file: Path,
    threshold: int = 5,
    workers: int | None = None,
    max_bucket_size: int = 10_000,
    benchmark: bool = False,
) -> None:
    """
    Find duplicate paragraphs across all shards using LSH with external sort.

    NOTE:
        Memory usage: 16 bytes per paragraph (hash array) + 8 bytes per paragraph
        (union-find) + sort buffer. For 1B paragraphs: approx 40GB.

    Args:
        simhash_files: List of paths to simhash JSONL files
        output_file: Path for output clusters JSON
        threshold: Hamming distance threshold for duplicates
        workers: Number of parallel workers (default: approx CPU count)
        max_bucket_size: Skip buckets with more documents than this
        benchmark: If True, print timing breakdown
    """
    # Don't actually use all CPUs
    cpu_workers = 0
    if os.cpu_count() is not None:
        cpu_workers = os.cpu_count() - 5  # type: ignore
    num_workers = workers or (cpu_workers) or 1  # type: ignore
    timings: dict[str, float] = {}

    # Phase 1: Load records into memory-efficient structures
    t0 = time.perf_counter()
    logger.info(f"Loading simhash records from {len(simhash_files)} files...")

    doc_id_list: list[str] = []  # index -> doc_id string
    hashes = array.array("Q")  # 128-bit hashes as pairs of uint64

    num_books = 0
    for path in tqdm(simhash_files, desc="Loading files"):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                book_id = data["book_id"]
                num_books += 1
                for i, h in enumerate(data["simhashes"]):
                    doc_id = f"{book_id}.{i}"
                    doc_id_list.append(doc_id)
                    hash_int = int(h, 16) if isinstance(h, str) else h
                    # Split 128-bit hash into two 64-bit values
                    low = hash_int & ((1 << 64) - 1)
                    high = hash_int >> 64
                    hashes.append(low)
                    hashes.append(high)

    n_records = len(doc_id_list)
    timings["1_load_records"] = time.perf_counter() - t0
    logger.info(f"Loaded {n_records:,} paragraph records from {num_books:,} books")

    if n_records == 0:
        raise ValueError("No records found")

    # Phase 2: Write band entries to temp file
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        bands_unsorted = temp_path / "bands_unsorted.tsv"
        bands_sorted = temp_path / "bands_sorted.tsv"

        logger.info("Writing band entries to disk...")

        def record_iter() -> Iterator[tuple[int, int]]:
            for idx in range(n_records):
                h = hashes[idx * 2] | (hashes[idx * 2 + 1] << 64)
                yield idx, h

        _write_band_entries(record_iter(), bands_unsorted, total=n_records)
        timings["2_write_bands"] = time.perf_counter() - t0

        # Phase 3: External sort
        t0 = time.perf_counter()
        logger.info("Sorting band entries (external sort)...")
        _external_sort(bands_unsorted, bands_sorted, temp_path)
        timings["3_external_sort"] = time.perf_counter() - t0

        # Remove unsorted file to free disk space
        bands_unsorted.unlink()

        # Phase 4: Process buckets with mmap'd hashes
        t0 = time.perf_counter()
        logger.info(f"Processing buckets with {num_workers} workers...")
        uf = UnionFindInt(n_records)
        duplicates_found = 0
        buckets_skipped = 0

        # Write hashes to a binary file for mmap
        hash_file = temp_path / "hashes.bin"
        logger.info(f"Writing hashes to {hash_file}...")
        with open(hash_file, "wb") as f:
            hashes.tofile(f)
        n_hashes = len(hashes)

        # Validate hash file was written completely
        expected_size = n_hashes * 8  # 8 bytes per uint64
        actual_size = hash_file.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Hash file incomplete: expected {expected_size} bytes, got {actual_size}"
            )

        # Collect buckets from sorted file
        logger.info("Collecting buckets...")
        all_buckets: list[list[int]] = []
        for _band_key, doc_indices in _stream_buckets(bands_sorted):
            if len(doc_indices) > max_bucket_size:
                buckets_skipped += 1
                continue
            all_buckets.append(doc_indices)

        logger.info(f"Found {len(all_buckets):,} buckets to process ({buckets_skipped:,} skipped)")

        # Partition buckets into chunks for parallel processing
        max_buckets_per_chunk = 10_000
        chunk_size = min(max_buckets_per_chunk, max(1, len(all_buckets) // (num_workers * 4)))
        bucket_chunks = [
            all_buckets[i : i + chunk_size] for i in range(0, len(all_buckets), chunk_size)
        ]
        logger.info(
            f"Split into {len(bucket_chunks):,} chunks of up to {chunk_size:,} buckets each"
        )

        # Process bucket chunks in parallel using mmap'd hashes
        seen_pairs: set[tuple[int, int]] = set()
        max_pending = num_workers * 2

        # Diagnostic timing for bottleneck analysis
        time_wait = 0.0
        time_process = 0.0
        time_submit = 0.0

        def process_result(future: Future[list[tuple[int, int]]]) -> None:
            """Process a completed future's results."""
            nonlocal duplicates_found
            verified_pairs = future.result()
            for d1, d2 in verified_pairs:
                pair = (d1, d2) if d1 < d2 else (d2, d1)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    uf.union(d1, d2)
                    duplicates_found += 1

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker_mmap,
            initargs=(str(hash_file), n_hashes),
        ) as executor:
            pending: set[Future[list[tuple[int, int]]]] = set()
            chunk_iter = iter(bucket_chunks)

            with tqdm(total=len(bucket_chunks), desc="Processing batches") as pbar:
                # Initial fill: submit up to max_pending chunks
                t_submit = time.perf_counter()
                for chunk in itertools.islice(chunk_iter, max_pending):
                    pending.add(
                        executor.submit(
                            _process_bucket_batch_mmap, chunk, threshold, max_bucket_size
                        )
                    )
                time_submit += time.perf_counter() - t_submit

                # Process results and keep submitting until done
                while pending:
                    t_wait = time.perf_counter()
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    time_wait += time.perf_counter() - t_wait

                    t_process = time.perf_counter()
                    for future in done:
                        process_result(future)
                        pbar.update(1)
                    time_process += time.perf_counter() - t_process

                    t_submit = time.perf_counter()
                    for chunk in itertools.islice(chunk_iter, len(done)):
                        pending.add(
                            executor.submit(
                                _process_bucket_batch_mmap, chunk, threshold, max_bucket_size
                            )
                        )
                    time_submit += time.perf_counter() - t_submit

        timings["4_process_buckets"] = time.perf_counter() - t0

        if benchmark:
            total_inner = time_wait + time_process + time_submit
            if total_inner > 0:
                logger.info("-" * 40)
                logger.info("Phase 4 breakdown (main process):")
                logger.info(
                    f"  wait (workers):   {time_wait:.2f}s ({100 * time_wait / total_inner:.1f}%)"
                )
                logger.info(
                    f"  process results:  {time_process:.2f}s ({100 * time_process / total_inner:.1f}%)"
                )
                logger.info(
                    f"  submit:           {time_submit:.2f}s ({100 * time_submit / total_inner:.1f}%)"
                )
                logger.info("-" * 40)

        if buckets_skipped > 0:
            logger.info(f"Skipped {buckets_skipped:,} buckets exceeding {max_bucket_size:,} docs")

    # Phase 5: Build clusters
    t0 = time.perf_counter()
    logger.info(f"Found {duplicates_found:,} duplicate pairs")
    logger.info("Building clusters...")

    raw_clusters = uf.get_clusters()

    # Filter to clusters with duplicates and convert to string doc_ids
    clusters: dict[str, list[str]] = {}
    for root, members in raw_clusters.items():
        if len(members) > 1:
            root_doc_id = doc_id_list[root]
            clusters[root_doc_id] = sorted(doc_id_list[m] for m in members)

    timings["5_build_clusters"] = time.perf_counter() - t0

    # Phase 6: Write output
    t0 = time.perf_counter()
    output_data = {
        "clusters": clusters,
        "statistics": {
            "total_records": n_records,
            "duplicate_pairs": duplicates_found,
            "clusters": len(clusters),
        },
    }
    logger.debug("Writing dedup information to files...")
    atomic_write_json(output_data, output_file)
    timings["6_write_output"] = time.perf_counter() - t0

    logger.info(f"Clusters written to {output_file}")
    logger.info(
        f"Statistics: {output_data['statistics']['clusters']:,} clusters, "
        + f"{output_data['statistics']['duplicate_pairs']:,} duplicate pairs"
    )

    if benchmark:
        total = sum(timings.values())
        logger.info("=" * 60)
        logger.info("BENCHMARK TIMING BREAKDOWN")
        logger.info("=" * 60)
        for phase, duration in timings.items():
            pct = 100 * duration / total if total > 0 else 0
            logger.info(f"  {phase}: {duration:.2f}s ({pct:.1f}%)")
        logger.info("-" * 60)
        logger.info(f"  TOTAL: {total:.2f}s")
        logger.info("=" * 60)


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing *.simhashes.jsonl files",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSON file for cluster information",
)
@click.option(
    "--threshold",
    type=int,
    default=5,
    help="Hamming distance threshold for duplicates (default: 5)",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: number of CPUs)",
)
@click.option(
    "--benchmark",
    is_flag=True,
    default=False,
    help="Print timing breakdown for each phase",
)
def main(
    input_dir: Path,
    output_file: Path,
    threshold: int,
    workers: int | None,
    benchmark: bool,
) -> None:
    """
    Find duplicate paragraphs across all shards using LSH.

    Reads all simhash files from input directory, builds an LSH index using
    external sort for memory efficiency, finds candidate pairs, verifies them,
    and outputs clusters.

    Note: expect at least 40GB or memory for 1B paragraphs.

    Output format (clusters.json):

    \b
        {
            "clusters": {"rep_doc_id": ["member1", "member2", ...]},
            "statistics": {"total_records": N, "duplicate_pairs": N, "clusters": N}
        }

    Example:

    \b
        python -m commands.dedup_find_duplicates \\
            --input-dir DATA/dedup/simhashes \\
            --output-file DATA/dedup/clusters.json \\
            --benchmark
    """
    simhash_files = sorted(input_dir.glob("*.simhashes.jsonl"))
    if not simhash_files:
        raise click.ClickException(f"No *.simhashes.jsonl files found in {input_dir}")

    logger.info(f"Found {len(simhash_files)} simhash files")

    find_duplicates(
        simhash_files=simhash_files,
        output_file=output_file,
        threshold=threshold,
        workers=workers,
        benchmark=benchmark,
    )


if __name__ == "__main__":
    main()
