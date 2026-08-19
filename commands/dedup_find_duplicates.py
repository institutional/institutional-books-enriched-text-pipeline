"""
dedup_find_duplicates.py - Find duplicate paragraphs across all shards using LSH.

This is phase 2 of the deduplication workflow:

1. Compute simhashes
2. Find duplicates (this step)
3. Build lookups
4. Annotate

This is the part where we take all simhash data, put it into a single big pile,
and produce a single duplicate cluster graph (clusters.json).

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

import array
import bisect
import ctypes
import json
import mmap
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np
from loguru import logger
from tqdm import tqdm

from utils.atomic_write import atomic_write_json
from utils.simhash import BAND_BITS, NUM_BANDS
from utils.simhash_fast import hamming_distance
from utils.unionfind import UnionFindInt

if TYPE_CHECKING:
    from concurrent.futures import Future
    from typing import Iterator

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
_worker_file_handle = None


def _init_worker_mmap(hash_file_path: str, n_hashes: int) -> None:
    """
    Initialize worker with mmap'd hash file.

    Each worker mmaps the same binary file containing all hashes.
    The mmap is read-only and shared across all workers by the OS.

    Args:
        hash_file_path: Path to binary file with uint64 hash values
        n_hashes: Number of uint64 values in the file (2 per document)
    """
    global _worker_hashes_mmap, _worker_hashes_array, _worker_file_handle

    _worker_file_handle = open(hash_file_path, "rb")
    _worker_hashes_mmap = mmap.mmap(_worker_file_handle.fileno(), 0, access=mmap.ACCESS_READ)

    if _cpp_bucket_module is None:
        # For Python: cast mmap to array of uint64 for indexed access (zero-copy)
        _worker_hashes_array = (ctypes.c_uint64 * n_hashes).from_buffer(_worker_hashes_mmap)


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
    if _cpp_bucket_module is not None and _worker_hashes_mmap is not None:
        # Use C++ implementation — cast mmap to uint64 view
        hashes_view = memoryview(_worker_hashes_mmap).cast("Q")
        return list(
            _cpp_bucket_module.process_bucket_batch(  # type: ignore[union-attr]
                buckets, hashes_view, threshold, max_bucket_size
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
# Per-band in-memory sort
# ============================================================================


def _extract_band_values(hashes_lo: np.ndarray, hashes_hi: np.ndarray, band_idx: int) -> np.ndarray:
    """
    Extract band values for all records using vectorized numpy operations.

    Args:
        hashes_lo: Array of low 64 bits of each hash (uint64)
        hashes_hi: Array of high 64 bits of each hash (uint64)
        band_idx: Which band to extract (0-5)

    Returns:
        Array of band values (uint32)
    """
    offset = sum(BAND_BITS[:band_idx])
    bits = BAND_BITS[band_idx]
    mask = np.uint64((1 << bits) - 1)

    if offset + bits <= 64:
        return ((hashes_lo >> np.uint64(offset)) & mask).astype(np.uint32)
    elif offset >= 64:
        hi_offset = offset - 64
        return ((hashes_hi >> np.uint64(hi_offset)) & mask).astype(np.uint32)
    else:
        # Spans low and high
        bits_from_lo = 64 - offset
        bits_from_hi = bits - bits_from_lo
        lo_part = (hashes_lo >> np.uint64(offset)).astype(np.uint64)
        hi_mask = np.uint64((1 << bits_from_hi) - 1)
        hi_part = (hashes_hi & hi_mask).astype(np.uint64)
        combined = (lo_part | (hi_part << np.uint64(bits_from_lo))) & mask
        return combined.astype(np.uint32)


def _buckets_from_sorted_keys(sorted_keys: np.ndarray, n_records: int) -> Iterator[list[int]]:
    """
    Scan a sorted array of packed (band_value << 32 | doc_idx) keys and yield
    buckets (groups of doc_indices sharing the same band_value) with 2+ members.

    Args:
        sorted_keys: Sorted uint64 array of packed keys
        n_records: Total number of records

    Yields:
        Lists of doc_indices that share the same band value
    """
    band_values = (sorted_keys >> np.uint64(32)).astype(np.uint32)
    doc_indices = (sorted_keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    # Find boundaries where band_value changes
    changes = np.flatnonzero(np.diff(band_values)) + 1
    starts = np.empty(len(changes) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = changes

    ends = np.empty(len(changes) + 1, dtype=np.intp)
    ends[:-1] = changes
    ends[-1] = n_records

    for i in range(len(starts)):
        size = ends[i] - starts[i]
        if size >= 2:
            yield doc_indices[starts[i] : ends[i]].tolist()


def find_duplicates(
    simhash_files: list[Path],
    output_file: Path,
    threshold: int = 5,
    workers: int | None = None,
    max_bucket_size: int = 30_000,
    benchmark: bool = False,
    temp_dir: Path | None = None,
    resume: bool = False,
) -> None:
    """
    Find duplicate paragraphs across all shards using LSH with per-band in-memory sort.

    NOTE:
        Memory usage: 16 bytes per paragraph (hash arrays) + 8 bytes per paragraph
        (sort buffer per band) + 8 bytes per paragraph (union-find).
        For 1B paragraphs: approx 32-40GB peak.

    Args:
        simhash_files: List of paths to simhash JSONL files
        output_file: Path for output clusters JSON
        threshold: Hamming distance threshold for duplicates
        workers: Number of parallel workers (default: approx CPU count)
        max_bucket_size: Skip buckets with more documents than this
        benchmark: If True, print timing breakdown
        temp_dir: Directory for hash file (used by workers for mmap)
        resume: If True, reuse existing hashes.bin from temp_dir
    """
    cpu_workers = 0
    if os.cpu_count() is not None:
        cpu_workers = max(1, os.cpu_count() - 5)  # type: ignore
    num_workers = workers or (cpu_workers) or 1  # type: ignore
    timings: dict[str, float] = {}

    # Phase 1: Load records and build compact book index
    t0 = time.perf_counter()
    logger.info(f"Loading simhash records from {len(simhash_files)} files...")

    book_ids: list[str] = []
    book_offsets = array.array("q")
    hashes_lo_list: list[int] = []
    hashes_hi_list: list[int] = []

    n_records = 0
    num_books = 0
    for path in tqdm(simhash_files, desc="Loading files"):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                book_id = data["book_id"]
                num_books += 1
                book_ids.append(book_id)
                book_offsets.append(n_records)
                for h in data["simhashes"]:
                    hash_int = int(h, 16) if isinstance(h, str) else h
                    hashes_lo_list.append(hash_int & ((1 << 64) - 1))
                    hashes_hi_list.append(hash_int >> 64)
                    n_records += 1

    timings["1_load_records"] = time.perf_counter() - t0
    logger.info(f"Loaded {n_records:,} paragraph records from {num_books:,} books")

    if n_records == 0:
        raise ValueError("No records found")

    if n_records > 2**32:
        raise ValueError(
            f"Too many records ({n_records:,}) for uint32 doc indices. "
            "Maximum supported is 4,294,967,295."
        )

    # Convert to numpy arrays for vectorized band extraction
    t0 = time.perf_counter()
    hashes_lo = np.array(hashes_lo_list, dtype=np.uint64)
    hashes_hi = np.array(hashes_hi_list, dtype=np.uint64)
    del hashes_lo_list, hashes_hi_list
    timings["1b_build_numpy"] = time.perf_counter() - t0

    # Write hashes.bin for mmap-based worker verification
    temp_ctx = tempfile.TemporaryDirectory() if temp_dir is None else None
    if temp_ctx is not None:
        temp_path = Path(temp_ctx.name)
    else:
        assert temp_dir is not None
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir

    try:
        hash_file = temp_path / "hashes.bin"

        if resume and hash_file.exists():
            logger.info("Resuming: reusing existing hashes.bin")
            n_hashes = hash_file.stat().st_size // 8
            if n_hashes != n_records * 2:
                raise RuntimeError(
                    f"Resume failed: hashes.bin has {n_hashes // 2:,} records "
                    f"but simhash files have {n_records:,}"
                )
        else:
            t0 = time.perf_counter()
            logger.info(f"Writing hashes to {hash_file}...")
            # Interleave lo/hi into flat array: [lo0, hi0, lo1, hi1, ...]
            interleaved = np.empty(n_records * 2, dtype=np.uint64)
            interleaved[0::2] = hashes_lo
            interleaved[1::2] = hashes_hi
            interleaved.tofile(hash_file)
            del interleaved
            timings["2_write_hashes"] = time.perf_counter() - t0

        n_hashes = n_records * 2

        # Phase 2-3: Per-band sort and bucket extraction
        t0 = time.perf_counter()
        logger.info(f"Processing {NUM_BANDS} bands with in-memory sort...")
        uf = UnionFindInt(n_records)
        duplicates_found = 0
        buckets_skipped = 0
        total_buckets = 0

        # Diagnostic timing
        time_sort = 0.0
        time_wait = 0.0
        time_process = 0.0
        time_submit = 0.0

        max_pending = num_workers * 2
        chunk_size = 10_000

        def process_result(future: Future[list[tuple[int, int]]]) -> None:
            nonlocal duplicates_found
            verified_pairs = future.result()
            for d1, d2 in verified_pairs:
                if uf.union(d1, d2):
                    duplicates_found += 1

        skipped_buckets_file = output_file.with_suffix(".skipped_buckets.jsonl")
        skipped_f = open(skipped_buckets_file, "w")

        doc_indices_arr = np.arange(n_records, dtype=np.uint64)

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker_mmap,
            initargs=(str(hash_file), n_hashes),
        ) as executor:
            pending: set[Future[list[tuple[int, int]]]] = set()
            chunks_submitted = 0
            chunks_completed = 0

            def submit_chunk(current_chunk: list[list[int]]) -> None:
                nonlocal chunks_submitted
                if not current_chunk:
                    return
                pending.add(
                    executor.submit(
                        _process_bucket_batch_mmap, list(current_chunk), threshold, max_bucket_size
                    )
                )
                chunks_submitted += 1

            def drain_completed(block: bool = False) -> None:
                nonlocal chunks_completed, time_process
                if not pending:
                    return
                if block:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                else:
                    done = {f for f in pending if f.done()}
                for future in done:
                    pending.discard(future)
                    t_proc = time.perf_counter()
                    process_result(future)
                    chunks_completed += 1
                    time_process += time.perf_counter() - t_proc

            for band_idx in range(NUM_BANDS):
                logger.info(f"  Band {band_idx}/{NUM_BANDS}: sorting and extracting buckets...")
                t_band = time.perf_counter()

                # Extract band values (vectorized)
                band_values = _extract_band_values(hashes_lo, hashes_hi, band_idx)

                # Pack (band_value, doc_idx) into uint64 for single-key sort
                keys = (band_values.astype(np.uint64) << np.uint64(32)) | doc_indices_arr
                del band_values

                keys.sort()
                t_sort_end = time.perf_counter()
                time_sort += t_sort_end - t_band
                logger.info(
                    f"  Band {band_idx}/{NUM_BANDS}: sort complete "
                    f"({t_sort_end - t_band:.1f}s), submitting buckets to workers..."
                )

                # Extract buckets from sorted keys
                current_chunk: list[list[int]] = []
                band_buckets = 0
                band_skipped = 0

                pbar = tqdm(
                    _buckets_from_sorted_keys(keys, n_records),
                    desc=f"  Band {band_idx} buckets",
                    unit=" buckets",
                )
                for doc_indices in pbar:
                    if len(doc_indices) > max_bucket_size:
                        band_skipped += 1
                        skipped_f.write(
                            json.dumps(
                                {
                                    "band_idx": band_idx,
                                    "size": len(doc_indices),
                                    "doc_indices": doc_indices,
                                }
                            )
                            + "\n"
                        )
                        continue
                    band_buckets += 1
                    current_chunk.append(doc_indices)

                    if len(current_chunk) >= chunk_size:
                        if len(pending) >= max_pending:
                            t_w = time.perf_counter()
                            drain_completed(block=True)
                            time_wait += time.perf_counter() - t_w

                        t_s = time.perf_counter()
                        submit_chunk(current_chunk)
                        time_submit += time.perf_counter() - t_s
                        current_chunk = []

                pbar.close()

                # Submit remaining buckets for this band
                if current_chunk:
                    if len(pending) >= max_pending:
                        t_w = time.perf_counter()
                        drain_completed(block=True)
                        time_wait += time.perf_counter() - t_w
                    t_s = time.perf_counter()
                    submit_chunk(current_chunk)
                    time_submit += time.perf_counter() - t_s

                del keys
                total_buckets += band_buckets
                buckets_skipped += band_skipped

                logger.info(
                    f"  Band {band_idx}/{NUM_BANDS}: {band_buckets:,} buckets, "
                    f"{band_skipped:,} skipped, "
                    f"sort {t_sort_end - t_band:.1f}s | "
                    f"duplicates so far: {duplicates_found:,}"
                )

            # Drain all remaining futures
            logger.info(
                f"  All bands processed. Draining {len(pending)} remaining worker chunks..."
            )
            while pending:
                t_w = time.perf_counter()
                drain_completed(block=True)
                time_wait += time.perf_counter() - t_w

        skipped_f.close()
        if buckets_skipped > 0:
            logger.info(f"Wrote {buckets_skipped:,} oversized buckets to {skipped_buckets_file}")
        else:
            skipped_buckets_file.unlink(missing_ok=True)

        timings["2_sort_and_buckets"] = time.perf_counter() - t0

        logger.info(
            f"Processed {total_buckets:,} buckets in {chunks_submitted:,} chunks "
            f"({buckets_skipped:,} oversized buckets skipped)"
        )

        if benchmark:
            total_inner = time_sort + time_wait + time_process + time_submit
            if total_inner > 0:
                logger.info("-" * 40)
                logger.info("Sort & bucket processing breakdown:")
                logger.info(
                    f"  numpy sort:       {time_sort:.2f}s ({100 * time_sort / total_inner:.1f}%)"
                )
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

        # Free hash arrays before cluster building
        del hashes_lo, hashes_hi, doc_indices_arr
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    # Phase 5: Build clusters
    t0 = time.perf_counter()
    logger.info(f"Found {duplicates_found:,} duplicate pairs")
    logger.info("Building clusters...")

    def doc_idx_to_id(idx: int) -> str:
        """Convert integer doc index to string ID using compact book index."""
        book_i = bisect.bisect_right(book_offsets, idx) - 1
        para_i = idx - book_offsets[book_i]
        return f"{book_ids[book_i]}.{para_i}"

    raw_clusters = uf.get_clusters()

    # Filter to clusters with duplicates and convert to string doc_ids
    # Use alphabetically first member as representative for determinism
    clusters: dict[str, list[str]] = {}
    for _, members in tqdm(raw_clusters.items(), desc="Clusters"):
        if len(members) > 1:
            member_ids = sorted(doc_idx_to_id(m) for m in members)
            clusters[member_ids[0]] = member_ids

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
@click.option(
    "--temp-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for temp files (hashes.bin for worker mmap). Not cleaned up if specified.",
)
def main(
    input_dir: Path,
    output_file: Path,
    threshold: int,
    workers: int | None,
    benchmark: bool,
    temp_dir: Path | None,
) -> None:
    """
    Find duplicate paragraphs across all shards using LSH.

    Reads all simhash files from input directory, performs per-band in-memory
    sorting to find candidate pairs, verifies them with Hamming distance,
    and outputs clusters.

    Note: expect at least 32-40GB of memory for 1B paragraphs.

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
        temp_dir=temp_dir,
    )


if __name__ == "__main__":
    main()
