"""
dedup_find_duplicates.py - Find duplicate paragraphs across all shards using LSH.

This is phase 2 of the deduplication workflow:
1. Compute simhashes - parallelizable per shard
2. Find duplicates (this step) - requires all simhashes in memory
3. Annotate - parallelizable per shard
"""

import array
import itertools
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from utils.atomic_write import atomic_write_json
from utils.simhash_fast import extract_bands, hamming_distance
from utils.unionfind import UnionFind, UnionFindInt


def _process_buckets(
    buckets: list[set[str]], max_bucket_size: int
) -> tuple[set[tuple[str, str]], int]:
    """Process a chunk of buckets, returning candidate pairs and skipped count."""
    candidates: set[tuple[str, str]] = set()
    skipped = 0
    for doc_ids in buckets:
        if len(doc_ids) < 2:
            continue
        if len(doc_ids) > max_bucket_size:
            skipped += 1
            continue
        doc_list = sorted(doc_ids)
        for i, d1 in enumerate(doc_list):
            for d2 in doc_list[i + 1 :]:
                candidates.add((d1, d2))
    return candidates, skipped


def _verify_pairs(
    pairs: list[tuple[str, str]], hash_lookup: dict[str, int], threshold: int
) -> list[tuple[str, str]]:
    """Verify candidate pairs, returning those within threshold."""
    duplicates = []
    for d1, d2 in pairs:
        h1, h2 = hash_lookup[d1], hash_lookup[d2]
        if hamming_distance(h1, h2) <= threshold:
            duplicates.append((d1, d2))
    return duplicates


# ============================================================================
# Code for disk-based duplication
# ============================================================================


def _write_band_entries(
    records_iter,
    output_path: Path,
    total: int | None = None,
) -> None:
    """Write band entries to a tab-separated file for external sorting.

    Format: band_idx<TAB>band_value<TAB>doc_idx
    """
    with open(output_path, "w") as f:
        for doc_idx, hash_val in tqdm(records_iter, total=total, desc="Writing bands"):
            bands = extract_bands(hash_val)
            for band_idx, band_value in enumerate(bands):
                f.write(f"{band_idx}\t{band_value}\t{doc_idx}\n")


def _external_sort(input_path: Path, output_path: Path, temp_dir: Path) -> None:
    """Sort band entries file using unix sort (external merge sort)."""
    # Sort by band_idx (numeric), then band_value (numeric)
    # -t$'\t' sets tab as delimiter
    # -k1,1n sorts by first field numerically
    # -k2,2n sorts by second field numerically
    # -T sets temp directory for large sorts
    # -S sets buffer size (use available memory)
    cmd = [
        "sort",
        "-t\t",
        "-k1,1n",
        "-k2,2n",
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
        raise RuntimeError(f"Sort failed: {result.stderr}")


def _stream_buckets(sorted_path: Path):
    """Stream through sorted band file, yielding (band_key, [doc_indices]) groups."""
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

    # Don't forget the last bucket
    if current_key is not None and len(current_docs) >= 2:
        yield current_key, current_docs


def _process_bucket_streaming(
    doc_indices: list[int],
    hash_lookup: dict[int, int],
    threshold: int,
    max_bucket_size: int,
) -> list[tuple[int, int]]:
    """Process a single bucket: generate pairs, verify, return duplicates."""
    if len(doc_indices) > max_bucket_size:
        return []  # Skip oversized buckets

    verified = []
    doc_indices_sorted = sorted(doc_indices)
    for i, d1 in enumerate(doc_indices_sorted):
        h1 = hash_lookup[d1]
        for d2 in doc_indices_sorted[i + 1 :]:
            h2 = hash_lookup[d2]
            if hamming_distance(h1, h2) <= threshold:
                verified.append((d1, d2))
    return verified


def _process_bucket_batch(
    buckets: list[list[int]],
    hash_lookup: dict[int, int],
    threshold: int,
    max_bucket_size: int,
) -> list[tuple[int, int]]:
    """Process a batch of buckets, returning all verified pairs."""
    all_verified = []
    for doc_indices in buckets:
        verified = _process_bucket_streaming(doc_indices, hash_lookup, threshold, max_bucket_size)
        all_verified.extend(verified)
    return all_verified


def _run_streaming_mode(
    simhash_files: list[Path],
    output_file: Path,
    threshold: int,
    benchmark: bool,
    workers: int | None = None,
    max_bucket_size: int = 10_000,
) -> None:
    """Run deduplication in streaming mode with external sort.

    Memory usage: ~16 bytes per paragraph (for hash array) + ~8 bytes per
    paragraph (for union-find) + sort buffer. For 1B paragraphs: ~35GB.
    """
    num_workers = workers or os.cpu_count() or 1
    timings: dict[str, float] = {}

    # Phase 1: Load records into memory-efficient structures
    t0 = time.perf_counter()
    logger.info(f"Loading simhash records from {len(simhash_files)} files...")

    # We'll store doc_ids as strings for final output, but use integer indices internally
    doc_id_list: list[str] = []  # index -> doc_id string
    # Store 128-bit hashes as pairs of 64-bit values in an array
    # Using unsigned long long ('Q') = 8 bytes each, so 16 bytes per hash
    hashes = array.array("Q")

    num_books = 0
    for path in tqdm(simhash_files):
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
    logger.info(f"Loaded {n_records} paragraph records from {num_books} books")

    if n_records == 0:
        raise ValueError("No records found")

    # Phase 2: Write band entries to temp file
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        bands_unsorted = temp_path / "bands_unsorted.tsv"
        bands_sorted = temp_path / "bands_sorted.tsv"

        logger.info("Writing band entries to disk...")

        def record_iter():
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

        # Phase 4: Stream through sorted file, process buckets, build union-find
        t0 = time.perf_counter()
        logger.info(f"Processing buckets with {num_workers} workers...")
        uf = UnionFindInt(n_records)
        duplicates_found = 0
        buckets_skipped = 0

        # Build full hash lookup once (dict of int -> int, not the array)
        # This takes far less RAM than the whole array.
        hash_lookup: dict[int, int] = {}
        for idx in range(n_records):
            hash_lookup[idx] = hashes[idx * 2] | (hashes[idx * 2 + 1] << 64)

        # Count buckets and collect them (we need to iterate twice anyway for progress)
        logger.info("Collecting buckets...")
        all_buckets: list[list[int]] = []
        for _band_key, doc_indices in _stream_buckets(bands_sorted):
            if len(doc_indices) > max_bucket_size:
                buckets_skipped += 1
                continue
            all_buckets.append(doc_indices)

        logger.info(f"Found {len(all_buckets)} buckets to process ({buckets_skipped} skipped)")

        # Partition buckets into chunks for parallel processing
        # Use smaller chunks than num_workers for better load balancing and smaller hash subsets
        max_buckets_per_chunk = 10_000  # Keeps hash subset to ~100K entries (~2-5 MB)
        chunk_size = min(max_buckets_per_chunk, max(1, len(all_buckets) // (num_workers * 4)))
        bucket_chunks = [
            all_buckets[i : i + chunk_size] for i in range(0, len(all_buckets), chunk_size)
        ]
        logger.info(f"Split into {len(bucket_chunks)} chunks of up to {chunk_size} buckets each")

        # Process bucket chunks in parallel, passing only needed hashes per chunk
        # Use a sliding window to interleave extraction, submission, and result collection
        seen_pairs: set[tuple[int, int]] = set()  # Dedupe across bands
        max_pending = num_workers * 2  # Keep queue fed but not too deep

        def extract_and_submit(executor, chunk):
            """Extract needed hashes and submit chunk for processing."""
            needed_ids = {doc_id for bucket in chunk for doc_id in bucket}
            chunk_hashes = {doc_id: hash_lookup[doc_id] for doc_id in needed_ids}
            return executor.submit(
                _process_bucket_batch, chunk, chunk_hashes, threshold, max_bucket_size
            )

        def process_result(future):
            """Process a completed future's results."""
            nonlocal duplicates_found
            verified_pairs = future.result()
            for d1, d2 in verified_pairs:
                pair = (d1, d2) if d1 < d2 else (d2, d1)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    uf.union(d1, d2)
                    duplicates_found += 1

        from concurrent.futures import FIRST_COMPLETED, wait

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pending: set = set()
            chunk_iter = iter(bucket_chunks)

            with tqdm(total=len(bucket_chunks), desc="Processing batches") as pbar:
                # Initial fill: submit up to max_pending chunks
                for chunk in itertools.islice(chunk_iter, max_pending):
                    pending.add(extract_and_submit(executor, chunk))

                # Process results and keep submitting until done
                while pending:
                    # Wait for at least one to complete
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)

                    # Process completed futures
                    for future in done:
                        process_result(future)
                        pbar.update(1)

                    # Submit more work to keep the queue full
                    for chunk in itertools.islice(chunk_iter, len(done)):
                        pending.add(extract_and_submit(executor, chunk))

        timings["4_process_buckets"] = time.perf_counter() - t0

        if buckets_skipped > 0:
            logger.info(f"Skipped {buckets_skipped} buckets exceeding {max_bucket_size} docs")

    # Phase 5: Build clusters
    t0 = time.perf_counter()
    logger.info(f"Found {duplicates_found} duplicate pairs")
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
        f"Statistics: {output_data['statistics']['clusters']} clusters, "
        f"{output_data['statistics']['duplicate_pairs']} duplicate pairs"
    )

    if benchmark:
        total = sum(timings.values())
        logger.info("=" * 60)
        logger.info("BENCHMARK TIMING BREAKDOWN (streaming mode)")
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
    "--streaming",
    is_flag=True,
    default=False,
    help="Use disk-based streaming mode for low memory usage (~35GB for 1B paragraphs)",
)
def main(
    input_dir: Path,
    output_file: Path,
    threshold: int,
    workers: int | None,
    benchmark: bool,
    streaming: bool,
):
    """
    Find duplicate paragraphs across all shards using LSH.

    Reads all simhash files from input directory, builds an LSH index,
    finds candidate pairs, verifies them, and outputs clusters.

    Two modes available:
    - Default: In-memory LSH index (faster, ~300GB RAM for 1B paragraphs)
    - Streaming (--streaming): Disk-based external sort (~35GB RAM for 1B paragraphs)

    Output format (clusters.json):
        {
            "clusters": {"rep_doc_id": ["member1", "member2", ...]},
            "statistics": {"total_records": N, "duplicate_pairs": N, "clusters": N}
        }

    Example:
        python -m commands.dedup_find_duplicates \\
            --input-dir DATA/dedup/simhashes \\
            --output-file DATA/dedup/clusters.json

        # For large datasets (low memory mode):
        python -m commands.dedup_find_duplicates \\
            --input-dir DATA/dedup/simhashes \\
            --output-file DATA/dedup/clusters.json \\
            --streaming --benchmark
    """
    simhash_files = sorted(input_dir.glob("*.simhashes.jsonl"))
    if not simhash_files:
        raise click.ClickException(f"No *.simhashes.jsonl files found in {input_dir}")

    # Dispatch to streaming mode if requested
    if streaming:
        logger.info("Running in STREAMING mode (disk-based, low memory)")
        _run_streaming_mode(simhash_files, output_file, threshold, benchmark, workers)
        return

    # --- In-memory mode below ---
    if workers:
        logger.info(f"Running in IN-MEMORY mode with {workers} workers")

    timings: dict[str, float] = {}

    # Phase 1: Load records
    t0 = time.perf_counter()
    logger.info(f"Loading simhash records from {len(simhash_files)} files...")
    records: list[tuple[str, int]] = []  # (doc_id, hash)

    num_books = 0
    for path in tqdm(simhash_files):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                book_id = data["book_id"]
                num_books += 1
                # Create docid references, converting hex strings to ints
                for i, h in enumerate(data["simhashes"]):
                    doc_id = f"{book_id}.{i}"
                    hash_int = int(h, 16) if isinstance(h, str) else h
                    records.append((doc_id, hash_int))

    timings["1_load_records"] = time.perf_counter() - t0
    logger.info(f"Loaded {len(records)} paragraph records")
    logger.info(f"from {num_books} many books.")

    if not records:
        raise ValueError("No records found")

    # Phase 2: Build LSH index
    t0 = time.perf_counter()
    logger.info("Building LSH index...")
    band_index: dict[tuple[int, int], set[str]] = {}

    for doc_id, h in tqdm(records):
        bands = extract_bands(h)
        for band_idx, band_value in enumerate(bands):
            key = (band_idx, band_value)
            if key not in band_index:
                band_index[key] = set()
            band_index[key].add(doc_id)

    timings["2_build_lsh_index"] = time.perf_counter() - t0

    # Phase 3: Find candidate pairs (docs that share at least one band)
    t0 = time.perf_counter()
    logger.info("Finding candidate pairs...")
    hash_lookup = {doc_id: h for doc_id, h in records}
    max_bucket_size = 10_000
    num_workers = workers or os.cpu_count() or 1

    # Partition buckets for parallel processing
    all_buckets = list(band_index.values())
    chunk_size = max(1, len(all_buckets) // num_workers)
    bucket_chunks = [
        all_buckets[i : i + chunk_size] for i in range(0, len(all_buckets), chunk_size)
    ]

    candidates: set[tuple[str, str]] = set()
    buckets_skipped = 0

    logger.info(f"Processing {len(all_buckets)} buckets with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_process_buckets, chunk, max_bucket_size) for chunk in bucket_chunks
        ]
        for future in tqdm(futures, desc="Finding pairs"):
            chunk_candidates, chunk_skipped = future.result()
            candidates.update(chunk_candidates)
            buckets_skipped += chunk_skipped

    timings["3_find_candidates"] = time.perf_counter() - t0

    if buckets_skipped > 0:
        logger.info(f"Skipped {buckets_skipped} buckets exceeding {max_bucket_size} docs")

    logger.info(f"Found {len(candidates)} candidate pairs")

    # Phase 4: Verify candidates using actual Hamming distance
    t0 = time.perf_counter()
    logger.info("Verifying candidates...")
    uf = UnionFind()

    # Partition candidates for parallel verification
    candidates_list = list(candidates)
    chunk_size = max(1, len(candidates_list) // num_workers)
    pair_chunks = [
        candidates_list[i : i + chunk_size] for i in range(0, len(candidates_list), chunk_size)
    ]

    verified_pairs: list[tuple[str, str]] = []
    logger.info(f"Verifying {len(candidates_list)} pairs with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_verify_pairs, chunk, hash_lookup, threshold) for chunk in pair_chunks
        ]
        for future in tqdm(futures, desc="Verifying pairs"):
            verified_pairs.extend(future.result())

    timings["4_verify_candidates"] = time.perf_counter() - t0

    # Phase 5: Build union-find clusters
    t0 = time.perf_counter()
    # Apply verified duplicates to union-find (must be sequential)
    for d1, d2 in verified_pairs:
        uf.union(d1, d2)

    duplicates_found = len(verified_pairs)
    logger.info(f"Found {duplicates_found} duplicate pairs")

    # Build clusters from union-find
    # First ensure all doc_ids are in the union-find structure
    for doc_id in hash_lookup:
        uf.find(doc_id)

    raw_clusters = uf.get_clusters()

    # Filter to clusters with actual duplicates
    clusters = {k: sorted(v) for k, v in raw_clusters.items() if len(v) > 1}

    timings["5_build_clusters"] = time.perf_counter() - t0

    # Phase 6: Write output
    t0 = time.perf_counter()
    output_data = {
        "clusters": clusters,
        "statistics": {
            "total_records": len(records),
            "duplicate_pairs": duplicates_found,
            "clusters": len(clusters),
        },
    }
    logger.debug("Writing dedup information to files...")
    atomic_write_json(output_data, output_file)
    timings["6_write_output"] = time.perf_counter() - t0

    logger.info(f"Clusters written to {output_file}")
    logger.info(
        f"Statistics: {output_data['statistics']['clusters']} clusters, "
        f"{output_data['statistics']['duplicate_pairs']} duplicate pairs"
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


if __name__ == "__main__":
    main()
