"""
Sample duplicate clusters and display the actual paragraph text.

Picks clusters across different size categories, looks up the paragraph text
from parquet shard files, and prints them side by side for inspection.

Usage:
    python scripts/sample_duplicate_paragraphs.py \
        DATA/Cluster/clusters.json \
        --parquet-dir DATA/Cluster/parquet_shards \
        --samples-per-size 3

    # Only sample clusters of a specific size
    python scripts/sample_duplicate_paragraphs.py \
        DATA/Cluster/clusters.json \
        --parquet-dir DATA/Cluster/parquet_shards \
        --cluster-size 2

    # Sample cross-book only (skip within-book clusters)
    python scripts/sample_duplicate_paragraphs.py \
        DATA/Cluster/clusters.json \
        --parquet-dir DATA/Cluster/parquet_shards \
        --cross-book-only

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import ijson
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_cluster_id(doc_id: str) -> tuple[str, int]:
    """Split 'BARCODE.para_idx' into (barcode, para_idx)."""
    dot = doc_id.rfind(".")
    return doc_id[:dot], int(doc_id[dot + 1 :])


def stream_clusters(path: Path):
    """Stream clusters from clusters.json using ijson."""
    with open(path, "rb") as f:
        parser = ijson.parse(f)
        current_key = None
        current_members: list[str] = []

        for prefix, event, value in parser:
            if prefix == "clusters" and event == "map_key":
                if current_key is not None:
                    yield current_key, current_members
                current_key = value
                current_members = []
            elif prefix.startswith("clusters.") and event == "string":
                current_members.append(value)

        if current_key is not None:
            yield current_key, current_members


def sample_clusters(
    clusters_path: Path,
    samples_per_size: int,
    target_size: int | None,
    cross_book_only: bool,
    seed: int,
) -> dict[int, list[tuple[str, list[str]]]]:
    """
    Use reservoir sampling to pick clusters across size categories.

    Returns: {cluster_size: [(rep_id, [members]), ...]}
    """
    rng = random.Random(seed)
    # Reservoir per size bucket
    reservoirs: dict[int, list[tuple[str, list[str]]]] = {}
    counts: dict[int, int] = {}

    print(f"Sampling clusters from {clusters_path} ...")
    for rep_id, members in stream_clusters(clusters_path):
        size = len(members)

        if target_size is not None and size != target_size:
            continue

        if cross_book_only:
            books = {parse_cluster_id(m)[0] for m in members}
            if len(books) == 1:
                continue

        counts[size] = counts.get(size, 0) + 1
        n = counts[size]

        if size not in reservoirs:
            reservoirs[size] = []

        if len(reservoirs[size]) < samples_per_size:
            reservoirs[size].append((rep_id, members))
        else:
            j = rng.randint(0, n - 1)
            if j < samples_per_size:
                reservoirs[size][j] = (rep_id, members)

    print(f"  Sampled from {sum(counts.values()):,} matching clusters"
          f" across {len(reservoirs)} size categories.")
    return reservoirs



BOOK_COLUMNS = ["barcode_src", "middlematter_sentences", "subtopic_paragraph_start_indices"]


def load_parquet_books(
    parquet_path: Path,
    needed: set[str],
    book_cache: dict[str, tuple[list[str], list[int]]],
) -> None:
    """Load all needed books from a parquet file into cache."""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=BOOK_COLUMNS, batch_size=200):
        barcodes = batch.column("barcode_src").to_pylist()
        for i, bc in enumerate(barcodes):
            if bc in needed and bc not in book_cache:
                sents = batch.column("middlematter_sentences")[i].as_py()
                starts = batch.column("subtopic_paragraph_start_indices")[i].as_py()
                book_cache[bc] = (sents or [], starts or [])


def extract_paragraph(
    barcode: str,
    para_idx: int,
    book_cache: dict[str, tuple[list[str], list[int]]],
) -> str | None:
    """Extract the text of a specific paragraph from a cached book."""
    if barcode not in book_cache:
        return None

    sents, starts = book_cache[barcode]
    if not starts or para_idx >= len(starts):
        return None

    sent_start = starts[para_idx]
    sent_end = starts[para_idx + 1] if para_idx + 1 < len(starts) else len(sents)
    return " ".join(sents[sent_start:sent_end])


def main():
    parser = argparse.ArgumentParser(description="Sample and display duplicate paragraphs")
    parser.add_argument("clusters_file", type=Path)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument(
        "--samples-per-size", type=int, default=3,
        help="Number of sample clusters per size category (default: 3)",
    )
    parser.add_argument(
        "--cluster-size", type=int, default=None,
        help="Only sample clusters of this exact size",
    )
    parser.add_argument(
        "--cross-book-only", action="store_true",
        help="Only sample clusters spanning multiple books",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-text-len", type=int, default=500,
        help="Max characters to display per paragraph (default: 500)",
    )
    parser.add_argument(
        "--max-sizes", type=int, default=10,
        help="Max number of size categories to display (default: 10). "
             "Picks a spread: smallest, largest, and evenly spaced in between.",
    )
    args = parser.parse_args()

    # Step 1: Sample clusters
    reservoirs = sample_clusters(
        args.clusters_file,
        args.samples_per_size,
        args.cluster_size,
        args.cross_book_only,
        args.seed,
    )

    if not reservoirs:
        print("No matching clusters found.", file=sys.stderr)
        sys.exit(1)

    # Trim to --max-sizes categories, picking a spread across the range
    all_sizes = sorted(reservoirs.keys())
    if len(all_sizes) > args.max_sizes:
        # Always include smallest and largest, spread the rest evenly
        indices = [int(i * (len(all_sizes) - 1) / (args.max_sizes - 1))
                   for i in range(args.max_sizes)]
        keep = {all_sizes[i] for i in indices}
        reservoirs = {sz: samples for sz, samples in reservoirs.items() if sz in keep}
        print(f"  Trimmed to {len(reservoirs)} size categories: {sorted(reservoirs.keys())}")

    # Collect all barcodes we need
    needed_barcodes: set[str] = set()
    for samples in reservoirs.values():
        for _, members in samples:
            for m in members:
                needed_barcodes.add(parse_cluster_id(m)[0])

    print(f"\nNeed to look up {len(needed_barcodes):,} books.")

    # Step 2: Find which parquet files contain our barcodes and load them
    barcode_to_file: dict[str, Path] = {}
    remaining = set(needed_barcodes)
    files = sorted(args.parquet_dir.glob("batch_*.parquet"))
    print(f"Searching {len(files)} parquet files for matching books...")
    for path in tqdm(files, desc="Searching"):
        if not remaining:
            break
        table = pq.read_table(path, columns=["barcode_src"])
        for bc in table.column("barcode_src").to_pylist():
            if bc in remaining:
                barcode_to_file[bc] = path
                remaining.discard(bc)
    if remaining:
        print(f"  Warning: {len(remaining)} barcodes not found in parquet files.")

    # Pre-load book data grouped by parquet file
    book_cache: dict[str, tuple[list[str], list[int]]] = {}
    files_to_load: dict[Path, set[str]] = {}
    for bc, path in barcode_to_file.items():
        files_to_load.setdefault(path, set()).add(bc)
    print(f"Loading book data from {len(files_to_load)} parquet files...")
    for path, barcodes_in_file in tqdm(files_to_load.items(), desc="Loading"):
        load_parquet_books(path, barcodes_in_file, book_cache)
    print(f"  Loaded {len(book_cache):,} books into cache.")

    # Step 3: Display
    wrapper = textwrap.TextWrapper(width=100, initial_indent="      ", subsequent_indent="      ")

    for size in sorted(reservoirs.keys()):
        samples = reservoirs[size]
        print(f"\n{'='*80}")
        print(f"  CLUSTER SIZE: {size}")
        print(f"{'='*80}")

        for sample_idx, (rep_id, members) in enumerate(samples):
            parsed = [parse_cluster_id(m) for m in members]
            books = {bc for bc, _ in parsed}
            scope = "within-book" if len(books) == 1 else f"cross-book ({len(books)} books)"

            print(f"\n  --- Sample {sample_idx + 1} (cluster {rep_id}, {scope}) ---")

            for member_idx, (barcode, para_idx) in enumerate(parsed):
                print(f"\n    [{member_idx + 1}] {barcode} paragraph {para_idx}")

                if barcode not in book_cache:
                    print("      (book not found in parquet files)")
                    continue

                text = extract_paragraph(barcode, para_idx, book_cache)
                if text is None:
                    print("      (paragraph not found)")
                    continue

                display = text[:args.max_text_len]
                if len(text) > args.max_text_len:
                    display += "..."
                for line in wrapper.wrap(display):
                    print(line)

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
