"""
dedup_find_duplicates.py - Find duplicate paragraphs across all shards using LSH.

This is phase 2 of the deduplication workflow:
1. Compute simhashes - parallelizable per shard
2. Find duplicates (this step) - requires all simhashes in memory
3. Annotate - parallelizable per shard
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from utils.atomic_write import atomic_write_json
from utils.simhash_fast import extract_bands, hamming_distance
from utils.unionfind import UnionFind


def _process_buckets(buckets: list[set[str]], max_bucket_size: int) -> tuple[set[tuple[str, str]], int]:
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
def main(input_dir: Path, output_file: Path, threshold: int, workers: int | None, benchmark: bool):
    """
    Find duplicate paragraphs across all shards using LSH.

    Reads all simhash files from input directory, builds an LSH index,
    finds candidate pairs, verifies them, and outputs clusters.

    Output format (clusters.json):
        {
            "clusters": {"rep_doc_id": ["member1", "member2", ...]},
            "statistics": {"total_records": N, "duplicate_pairs": N, "clusters": N}
        }

    Example:
        python -m commands.dedup_find_duplicates \\
            --input-dir DATA/dedup/simhashes \\
            --output-file DATA/dedup/clusters.json
    """
    simhash_files = sorted(input_dir.glob("*.simhashes.jsonl"))
    if not simhash_files:
        raise click.ClickException(f"No *.simhashes.jsonl files found in {input_dir}")

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
            executor.submit(_process_buckets, chunk, max_bucket_size)
            for chunk in bucket_chunks
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
        candidates_list[i : i + chunk_size]
        for i in range(0, len(candidates_list), chunk_size)
    ]

    verified_pairs: list[tuple[str, str]] = []
    logger.info(f"Verifying {len(candidates_list)} pairs with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_verify_pairs, chunk, hash_lookup, threshold)
            for chunk in pair_chunks
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
