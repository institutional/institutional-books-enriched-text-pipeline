"""
Build a barcode -> parquet file index for fast book lookup.

This is a simple script that makes reverse shard manipulation across a sharded
dataset easier.

Usage:
    python scripts/build_barcode_index.py \
        --parquet-dir DATA/Cluster/parquet_shards \
        --output-file DATA/Cluster/barcode_index.json

Output format:
    {"32044000000018": "batch_00000.parquet", ...}

    (Values are filenames relative to the parquet directory, not full paths.)

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Build barcode -> parquet file index")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.parquet_dir.glob("batch_*.parquet"))
    if not files:
        raise SystemExit(f"No batch_*.parquet files found in {args.parquet_dir}")

    print(f"Indexing {len(files)} parquet files...")
    index: dict[str, str] = {}
    for path in tqdm(files, desc="Indexing"):
        table = pq.read_table(path, columns=["barcode_src"])
        for barcode in table.column("barcode_src").to_pylist():
            index[barcode] = path.name

    print(f"Indexed {len(index):,} books.")

    with open(args.output_file, "w") as f:
        json.dump(index, f)

    size_mb = args.output_file.stat().st_size / 1024 / 1024
    print(f"Wrote {args.output_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
