"""
Inspect skipped buckets from dedup_find_duplicates.

Loads the book index from simhash files, then decodes doc indices
from the skipped buckets JSONL into human-readable book_id.paragraph_idx
form. Prints a summary of each bucket with sample paragraphs.

Usage:
    python scripts/inspect_skipped_buckets.py \
        --simhash-dir DATA/Cluster/dedup \
        --skipped-file DATA/Cluster/clusters.skipped_buckets.jsonl \
        --sample-size 10

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

import array
import bisect
import json
from pathlib import Path

import click
from tqdm import tqdm


def build_book_index(
    simhash_dir: Path,
) -> tuple[list[str], array.array[int], int]:
    """Build compact book index from simhash files. Returns (book_ids, book_offsets, n_records)."""
    book_ids: list[str] = []
    book_offsets = array.array("q")
    n_records = 0

    simhash_files = sorted(simhash_dir.glob("*.simhashes.jsonl"))
    for path in tqdm(simhash_files, desc="Loading book index"):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                book_ids.append(data["book_id"])
                book_offsets.append(n_records)
                n_records += len(data["simhashes"])

    return book_ids, book_offsets, n_records


def doc_idx_to_id(idx: int, book_ids: list[str], book_offsets: array.array[int]) -> str:
    """Convert integer doc index to string ID."""
    book_i = bisect.bisect_right(book_offsets, idx) - 1
    para_i = idx - book_offsets[book_i]
    return f"{book_ids[book_i]}.{para_i}"


@click.command()
@click.option("--simhash-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--skipped-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--sample-size", type=int, default=10, help="Number of sample doc IDs per bucket")
def main(simhash_dir: Path, skipped_file: Path, sample_size: int) -> None:
    """Inspect skipped buckets and show sample document IDs."""
    book_ids, book_offsets, n_records = build_book_index(simhash_dir)
    print(f"\nBook index: {len(book_ids):,} books, {n_records:,} paragraphs\n")

    with open(skipped_file) as f:
        for i, line in enumerate(f):
            bucket = json.loads(line)
            band_idx = bucket["band_idx"]
            band_value = bucket["band_value"]
            size = bucket["size"]
            doc_indices = bucket["doc_indices"]

            # Decode sample doc IDs
            sample_indices = doc_indices[:sample_size]
            sample_ids = [doc_idx_to_id(idx, book_ids, book_offsets) for idx in sample_indices]

            # Count unique books in the full bucket
            unique_books = set()
            for idx in doc_indices:
                book_i = bisect.bisect_right(book_offsets, idx) - 1
                unique_books.add(book_ids[book_i])

            print(f"Bucket {i + 1}: band_idx={band_idx}, band_value={band_value}, "
                  f"size={size:,}, unique_books={len(unique_books):,}")
            for doc_id in sample_ids:
                print(f"  {doc_id}")
            print()


if __name__ == "__main__":
    main()
