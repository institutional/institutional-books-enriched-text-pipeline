#!/usr/bin/env python3
"""
Join per-book deduplication counts from ``clusters.json`` into the book-metadata
Parquet produced by ``collect_book_metadata.py``. This is used for technical
report statistics.

``clusters.json`` (≈4.6 GB) has the shape::

    {"clusters": {"<barcode>.<para_idx>": ["<barcode>.<para_idx>", ...], ...}}

The dict KEY of each cluster is the *representative* paragraph; every other
member of the list is a *duplicate* of it (this matches the semantics in
``commands/dedup_annotate.py``). For each book (barcode) we tally:

  - n_representative_paragraphs: cluster keys whose barcode is this book
  - n_duplicate_paragraphs:      non-key members whose barcode is this book

We also add a convenience column:

  - frac_duplicate_paragraphs:   n_duplicate_paragraphs / n_paragraphs

The clusters file is streamed once with ijson (yajl2_c backend) to keep memory
bounded.

Usage:
    python scripts/join_dedup_counts.py \
        --clusters DATA/Cluster/clusters.json \
        --parquet DATA/book_metadata.parquet \
        --output DATA/book_metadata.parquet
"""

import argparse
from collections import Counter

import ijson
import pandas as pd


def barcode_of(doc_id: str) -> str:
    """'32044000000018.107' -> '32044000000018'."""
    return doc_id.rsplit(".", 1)[0]


def tally_clusters(clusters_path: str) -> tuple[Counter, Counter]:
    """Stream clusters.json once, returning (rep_counts, dup_counts) by barcode."""
    rep_counts: Counter = Counter()
    dup_counts: Counter = Counter()
    n_clusters = 0
    with open(clusters_path, "rb") as f:
        for rep, members in ijson.kvitems(f, "clusters"):
            n_clusters += 1
            rep_counts[barcode_of(rep)] += 1
            for m in members:
                if m != rep:
                    dup_counts[barcode_of(m)] += 1
            if n_clusters % 5_000_000 == 0:
                print(f"  scanned {n_clusters:,} clusters", flush=True)
    print(f"  scanned {n_clusters:,} clusters total", flush=True)
    return rep_counts, dup_counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clusters", default="DATA/Cluster/clusters.json")
    ap.add_argument("--parquet", default="DATA/book_metadata.parquet")
    ap.add_argument("--output", default="DATA/book_metadata.parquet")
    args = ap.parse_args()

    print(f"Streaming {args.clusters} ...")
    rep_counts, dup_counts = tally_clusters(args.clusters)
    print(f"Books with a representative paragraph: {len(rep_counts):,}")
    print(f"Books with a duplicate paragraph:      {len(dup_counts):,}")

    df = pd.read_parquet(args.parquet)
    bc = df["barcode_src"]
    df["n_representative_paragraphs"] = bc.map(rep_counts).fillna(0).astype("int64")
    df["n_duplicate_paragraphs"] = bc.map(dup_counts).fillna(0).astype("int64")
    # Fraction of a book's paragraphs that are duplicates (0 when no paragraphs)
    npar = df["n_paragraphs"].replace(0, pd.NA)
    df["frac_duplicate_paragraphs"] = (
        (df["n_duplicate_paragraphs"] / npar).astype("float64").fillna(0.0)
    )

    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df):,} books × {len(df.columns)} columns -> {args.output}")

    # Sanity check: cluster barcodes that never appeared in the parquet
    known = set(bc)
    missing = (set(rep_counts) | set(dup_counts)) - known
    if missing:
        print(
            f"NOTE: {len(missing):,} barcodes in clusters.json are absent "
            f"from the parquet (not in the processed shards)."
        )


if __name__ == "__main__":
    main()
