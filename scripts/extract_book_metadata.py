"""
Extract per-book metadata from parquet shards for use in cluster analysis.

Usage:
    python scripts/extract_book_metadata.py \
        --parquet-dir DATA/Cluster/parquet_shards \
        --output-file DATA/Cluster/book_metadata.jsonl

Output format (one JSON object per line):
    {"barcode": "32044000000018", "language": "eng", "n_paragraphs": 1252}

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract per-book metadata from parquet shards")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        required=True,
        help="Directory containing batch_*.parquet files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSONL file",
    )
    args = parser.parse_args()

    files = sorted(args.parquet_dir.glob("batch_*.parquet"))
    if not files:
        raise SystemExit(f"No batch_*.parquet files found in {args.parquet_dir}")

    print(f"Reading {len(files)} parquet files...")

    columns = [
        "barcode_src",
        "language_gen",
        "subtopic_paragraph_start_indices",
    ]

    n_books = 0
    with open(args.output_file, "w") as out:
        for path in tqdm(files, desc="Parquet files"):
            table = pq.read_table(path, columns=columns)
            barcodes = table.column("barcode_src")
            languages = table.column("language_gen")
            para_starts = table.column("subtopic_paragraph_start_indices")

            for i in range(len(table)):
                barcode = barcodes[i].as_py()
                language = languages[i].as_py()

                # Number of paragraphs = length of paragraph start indices
                starts = para_starts[i].as_py()
                n_paragraphs = len(starts) if starts else 0

                out.write(
                    json.dumps(
                        {
                            "barcode": barcode,
                            "language": language,
                            "n_paragraphs": n_paragraphs,
                        }
                    )
                    + "\n"
                )
                n_books += 1

    print(f"Wrote {n_books:,} books to {args.output_file}")


if __name__ == "__main__":
    main()
