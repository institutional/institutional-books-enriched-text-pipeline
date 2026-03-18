"""
dedup_find_duplicates.py - Find duplicate paragraphs across all shards using LSH.

This is phase 2 of the deduplication workflow:
1. Compute simhashes - parallelizable per shard
2. Find duplicates (this step) - requires all simhashes in memory
3. Annotate - parallelizable per shard
"""

import json
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from utils.atomic_write import atomic_write_json
from utils.simhash_fast import extract_bands, hamming_distance
from utils.unionfind import UnionFind


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
def main(input_dir: Path, output_file: Path, threshold: int):
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

    logger.info(f"Loaded {len(records)} paragraph records")
    logger.info(f"from {num_books} many books.")

    if not records:
        raise ValueError("No records found")

    # Build LSH index (band_key -> set of doc_ids)
    logger.info("Building LSH index...")
    band_index: dict[tuple[int, int], set[str]] = {}

    for doc_id, h in tqdm(records):
        bands = extract_bands(h)
        for band_idx, band_value in enumerate(bands):
            key = (band_idx, band_value)
            if key not in band_index:
                band_index[key] = set()
            band_index[key].add(doc_id)

    # Find candidate pairs (docs that share at least one band)
    logger.info("Finding candidate pairs...")
    hash_lookup = {doc_id: h for doc_id, h in records}
    candidates: set[tuple[str, str]] = set()
    buckets_skipped = 0
    max_bucket_size = 10_000

    for doc_ids in tqdm(band_index.values()):
        if len(doc_ids) < 2:
            continue
        if len(doc_ids) > max_bucket_size:
            # Skip very large buckets
            buckets_skipped += 1
            logger.warning(
                f"Skipping bucket with {len(doc_ids)} docs (>{max_bucket_size}). "
                + f"Doc IDs: {sorted(doc_ids)}"
            )
            continue
        doc_list = sorted(doc_ids)
        for i, d1 in enumerate(doc_list):
            for d2 in doc_list[i + 1 :]:
                candidates.add((d1, d2))

    if buckets_skipped > 0:
        logger.info(f"Skipped {buckets_skipped} buckets exceeding {max_bucket_size} docs")

    logger.info(f"Found {len(candidates)} candidate pairs")

    # Verify candidates using actual Hamming distance
    logger.info("Verifying candidates...")
    uf = UnionFind()
    duplicates_found = 0

    for d1, d2 in tqdm(candidates):
        h1, h2 = hash_lookup[d1], hash_lookup[d2]
        if hamming_distance(h1, h2) <= threshold:
            uf.union(d1, d2)
            duplicates_found += 1

    logger.info(f"Found {duplicates_found} duplicate pairs")

    # Build clusters from union-find
    # First ensure all doc_ids are in the union-find structure
    for doc_id in hash_lookup:
        uf.find(doc_id)

    raw_clusters = uf.get_clusters()

    # Filter to clusters with actual duplicates
    clusters = {k: sorted(v) for k, v in raw_clusters.items() if len(v) > 1}

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
    logger.info(f"Clusters written to {output_file}")
    logger.info(
        f"Statistics: {output_data['statistics']['clusters']} clusters, "
        f"{output_data['statistics']['duplicate_pairs']} duplicate pairs"
    )


if __name__ == "__main__":
    main()
