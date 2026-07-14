#!/usr/bin/env python3
"""
Collect per-book summary statistics from the step-10 ``*.complete.jsonl`` shards
into a single Parquet file for analysis / the technical report.

Each output row is one book. Columns:

Identity / provenance
  - shard:        shard id parsed from the filename (e.g. "0001")
  - barcode_src:  book barcode (unique id)
  - title_src, author_src, date1_src, date2_src

Cleaning statistics (the ``_``-prefixed per-book counts). These are only written
to the source JSON when nonzero, so a missing key is treated as 0:
  - dehyphenations            (step 4)
  - duplicate_pages_removed   (step 2)
  - headers_footers_removed   (step 5)
  - page_numbers_removed      (step 6)
  - stray_numbers_removed     (step 9)

Size / quality / language / topic
  - page_count_src, token_count_o200k_base_gen
  - ocr_score_src, ocr_score_gen
  - language_src, language_gen, n_languages_gen, dominant_language_proportion_gen
  - topic_or_subject_src, topic_or_subject_gen, topic_or_subject_score_gen

Derived structural counts (from list lengths)
  - n_frontmatter_pages, n_backmatter_pages
  - n_middlematter_sentences, n_paragraphs, n_sections

Usage:
    python scripts/collect_book_metadata.py \
        --input-dir DATA/Cluster/bak/final/processed \
        --output DATA/book_metadata.parquet \
        --workers 8
"""

import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import pandas as pd

CLEANING_COUNTS = {
    "dehyphenations": "_dehyphenations",
    "duplicate_pages_removed": "_duplicate_pages_removed",
    "headers_footers_removed": "_headers_footers_removed",
    "page_numbers_removed": "_page_numbers_removed",
    "stray_numbers_removed": "_stray_numbers_removed",
}

PASSTHROUGH = [
    "barcode_src",
    "title_src",
    "author_src",
    "date1_src",
    "date2_src",
    "page_count_src",
    "token_count_o200k_base_gen",
    "ocr_score_src",
    "ocr_score_gen",
    "language_src",
    "language_gen",
    "topic_or_subject_src",
    "topic_or_subject_gen",
    "topic_or_subject_score_gen",
]

SHARD_RE = re.compile(r"shard(\d+)\.complete\.jsonl$")


def extract_row(book: dict, shard: str) -> dict:
    """Flatten one book's JSON into a single dataframe row."""
    row = {"shard": shard}

    for col in PASSTHROUGH:
        row[col] = book.get(col)

    for col, key in CLEANING_COUNTS.items():
        row[col] = book.get(key, 0)

    # Derived structural counts
    row["n_frontmatter_pages"] = len(book.get("frontmatter") or [])
    row["n_backmatter_pages"] = len(book.get("backmatter") or [])
    row["n_middlematter_sentences"] = len(book.get("middlematter_sentences") or [])
    row["n_paragraphs"] = len(book.get("subtopic_paragraph_start_indices") or [])
    row["n_sections"] = len(book.get("subtopic_section_start_indices") or [])

    # Language distribution: number of detected languages + dominant proportion.
    dist = book.get("language_distribution_gen") or {}
    props = dist.get("proportion") or []
    row["n_languages_gen"] = len(dist.get("language") or [])
    row["dominant_language_proportion_gen"] = props[0] if props else None

    return row


def process_file(path: str) -> list[dict]:
    """Parse one shard file, returning one row dict per book."""
    m = SHARD_RE.search(os.path.basename(path))
    shard = m.group(1) if m else os.path.basename(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                book = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(extract_row(book, shard))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input-dir",
        default="DATA/Cluster/bak/final/processed",
        help="Directory of *.complete.jsonl shard files.",
    )
    ap.add_argument("--output", default="DATA/book_metadata.parquet", help="Output Parquet path.")
    ap.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel worker processes.",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.complete.jsonl")))
    if not files:
        raise SystemExit(f"No *.complete.jsonl files found in {args.input_dir}")
    print(f"Found {len(files)} shard files; using {args.workers} workers.")

    all_rows: list[dict] = []
    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_file, files), 1):
            all_rows.extend(rows)
            if i % 100 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files  ({len(all_rows):,} books)", flush=True)

    df = pd.DataFrame(all_rows)
    # fixed order
    ordered = (
        ["shard", "barcode_src", "title_src", "author_src", "date1_src", "date2_src"]
        + list(CLEANING_COUNTS)
        + [
            "page_count_src",
            "token_count_o200k_base_gen",
            "ocr_score_src",
            "ocr_score_gen",
            "language_src",
            "language_gen",
            "n_languages_gen",
            "dominant_language_proportion_gen",
            "topic_or_subject_src",
            "topic_or_subject_gen",
            "topic_or_subject_score_gen",
            "n_frontmatter_pages",
            "n_backmatter_pages",
            "n_middlematter_sentences",
            "n_paragraphs",
            "n_sections",
        ]
    )
    df = df[[c for c in ordered if c in df.columns]]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df):,} books × {len(df.columns)} columns -> {args.output}")
    print(f"Parquet size: {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
