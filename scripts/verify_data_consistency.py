"""Verify consistency across parquet_shards, dedup (simhashes), and perplexity data."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


DATA_ROOT = Path("DATA/Cluster")
PARQUET_DIR = DATA_ROOT / "parquet_shards"
DEDUP_DIR = DATA_ROOT / "dedup"
PERPLEXITY_DIR = DATA_ROOT / "perplexity"


def load_parquet_books():
    """Return dict: barcode -> paragraph_count from all parquet files."""
    books = {}
    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
    for f in tqdm(parquet_files, desc="Parquet files"):
        table = pq.read_table(f, columns=["barcode_src", "subtopic_paragraph_start_indices"])
        d = table.to_pydict()
        for barcode, para_starts in zip(d["barcode_src"], d["subtopic_paragraph_start_indices"]):
            if barcode in books:
                print(f"  WARNING: duplicate barcode in parquet: {barcode}")
            books[barcode] = len(para_starts) if para_starts is not None else 0
    return books


def load_jsonl_books(directory, suffix, count_field):
    """Return dict: book_id -> count from all JSONL files in directory."""
    books = {}
    files = sorted(directory.glob(f"*{suffix}"))
    for f in tqdm(files, desc=f"Loading {directory.name}"):
        with open(f) as fh:
            for line_num, line in enumerate(fh, 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: bad JSON in {f.name} line {line_num}")
                    continue
                book_id = rec["book_id"]
                if book_id in books:
                    print(f"  WARNING: duplicate book_id in {directory.name}: {book_id}")
                books[book_id] = len(rec[count_field]) if rec[count_field] is not None else 0
    return books


def main():
    print("Loading parquet books...")
    parquet = load_parquet_books()
    print(f"  {len(parquet):,} books")

    print("Loading simhash books...")
    simhash = load_jsonl_books(DEDUP_DIR, ".simhashes.jsonl", "simhashes")
    print(f"  {len(simhash):,} books")

    print("Loading perplexity books...")
    perplexity = load_jsonl_books(PERPLEXITY_DIR, ".perplexity.jsonl", "perplexities")
    print(f"  {len(perplexity):,} books")

    # --- Barcode set comparison ---
    p_set = set(parquet)
    s_set = set(simhash)
    x_set = set(perplexity)

    print("\n=== Barcode Set Comparison ===")
    print(f"Parquet:    {len(p_set):,}")
    print(f"Simhash:    {len(s_set):,}")
    print(f"Perplexity: {len(x_set):,}")
    print(f"All three:  {len(p_set & s_set & x_set):,}")

    only_parquet = p_set - s_set - x_set
    only_simhash = s_set - p_set - x_set
    only_perplexity = x_set - p_set - s_set
    missing_simhash = p_set - s_set
    missing_perplexity = p_set - x_set
    missing_parquet_from_simhash = s_set - p_set
    missing_parquet_from_perplexity = x_set - p_set

    problems = False

    for label, ids in [
        ("In parquet but NOT simhash", missing_simhash),
        ("In parquet but NOT perplexity", missing_perplexity),
        ("In simhash but NOT parquet", missing_parquet_from_simhash),
        ("In perplexity but NOT parquet", missing_parquet_from_perplexity),
    ]:
        if ids:
            problems = True
            print(f"\n  {label}: {len(ids):,}")
            for bc in sorted(ids)[:10]:
                print(f"    {bc}")
            if len(ids) > 10:
                print(f"    ... and {len(ids) - 10} more")

    if not problems:
        print("\nAll three datasets have identical barcode sets.")

    # --- Paragraph count comparison ---
    print("\n=== Paragraph Count Comparison ===")
    common = p_set & s_set & x_set
    mismatches = defaultdict(list)

    for bc in sorted(common):
        p_count = parquet[bc]
        s_count = simhash[bc]
        x_count = perplexity[bc]
        if not (p_count == s_count == x_count):
            mismatches[bc] = (p_count, s_count, x_count)

    if mismatches:
        print(f"  {len(mismatches):,} books with mismatched paragraph counts:")
        for i, (bc, (p, s, x)) in enumerate(sorted(mismatches.items())):
            if i >= 20:
                print(f"  ... and {len(mismatches) - 20} more")
                break
            print(f"  {bc}: parquet={p}, simhash={s}, perplexity={x}")
    else:
        print(f"  All {len(common):,} common books have matching paragraph counts.")

    # --- Summary ---
    print("\n=== Summary ===")
    all_good = not problems and not mismatches
    if all_good:
        print("ALL CHECKS PASSED")
    else:
        print("ISSUES FOUND - see above")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
