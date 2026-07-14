#!/usr/bin/env python3
"""
Extract sampled books from source JSONL shards into new output shards,
separated by segmenter type (nupunkt vs sat).

Reads sampled_books.json (from sample_books_by_language.py), finds each book
in the corresponding source shard, and writes output shards with at most
--shard-size books each. Output files are named:

    shard0001_nupunkt.jsonl, shard0002_nupunkt.jsonl, ...
    shard0001_sat.jsonl, shard0002_sat.jsonl, ...

Usage:
    python scripts/extract_sampled_books.py \
        --sampled-books sampled_books.json \
        --source-dir DATA/shards/raw \
        --output-dir DATA/sampled_shards \
        --shard-size 10
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from const.languages import is_nupunkt_language


def find_source_file(source_dir: Path, shard_number: str) -> Path | None:
    """Find a JSONL file in source_dir matching the given shard number.

    Tries exact match first (shardNNNN.jsonl), then glob for any suffix
    (shardNNNN_*.jsonl).
    """
    padded = shard_number.zfill(4)
    exact = source_dir / f"shard{padded}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(source_dir.glob(f"shard{padded}*.jsonl"))
    if candidates:
        return candidates[0]
    return None


def extract_books_from_shard(source_file: Path, target_barcodes: set[str]) -> dict[str, dict]:
    """Scan a JSONL file and return matching books keyed by barcode."""
    found: dict[str, dict] = {}
    with open(source_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            book = json.loads(line)
            barcode = str(book.get("barcode_src", ""))
            if barcode in target_barcodes:
                found[barcode] = book
                if len(found) == len(target_barcodes):
                    break  # all found, stop early
    return found


class ShardWriter:
    """Writes books to numbered shard files, flushing every shard_size books."""

    def __init__(self, prefix: str, output_dir: Path, shard_size: int):
        self.prefix = prefix
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[dict] = []
        self.shard_count = 0
        self.total_books = 0

    def add(self, book: dict) -> None:
        self.buffer.append(book)
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def close(self) -> None:
        if self.buffer:
            self._flush()

    def _flush(self) -> None:
        self.shard_count += 1
        shard_num = str(self.shard_count).zfill(4)
        out_path = self.output_dir / f"shard{shard_num}_{self.prefix}.jsonl"
        with open(out_path, "w") as f:
            for book in self.buffer:
                f.write(json.dumps(book, ensure_ascii=False) + "\n")
        self.total_books += len(self.buffer)
        self.buffer.clear()


def main():
    parser = argparse.ArgumentParser(
        description="Extract sampled books into segmenter-separated shards."
    )
    parser.add_argument("--sampled-books", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.sampled_books) as f:
        sampled = json.load(f)

    # Group sampled books by shard number
    shard_to_entries: dict[str, list[dict]] = defaultdict(list)
    for entry in sampled:
        shard_to_entries[entry["shard_number"]].append(entry)

    # Stream books to output shards
    nupunkt_writer = ShardWriter("nupunkt", args.output_dir, args.shard_size)
    sat_writer = ShardWriter("sat", args.output_dir, args.shard_size)
    missing: list[str] = []

    for shard_num in sorted(shard_to_entries):
        entries = shard_to_entries[shard_num]
        target_barcodes = {e["book_id"] for e in entries}
        lang_map = {e["book_id"]: e["language"] for e in entries}

        source_file = find_source_file(args.source_dir, shard_num)
        if source_file is None:
            print(
                f"WARNING: No source file found for shard {shard_num}, "
                f"skipping {len(entries)} books"
            )
            missing.extend(e["book_id"] for e in entries)
            continue

        print(f"Scanning {source_file.name} for {len(target_barcodes)} books...")
        found = extract_books_from_shard(source_file, target_barcodes)

        for barcode in sorted(target_barcodes):
            if barcode not in found:
                print(f"  WARNING: {barcode} not found in {source_file.name}")
                missing.append(barcode)
                continue
            lang = lang_map[barcode]
            if is_nupunkt_language(lang):
                nupunkt_writer.add(found[barcode])
            else:
                sat_writer.add(found[barcode])

    nupunkt_writer.close()
    sat_writer.close()

    print(f"\nNupunkt: {nupunkt_writer.total_books} books in {nupunkt_writer.shard_count} shards")
    print(f"SaT:     {sat_writer.total_books} books in {sat_writer.shard_count} shards")
    if missing:
        print(f"Missing: {len(missing)} books not found in source")


if __name__ == "__main__":
    main()
