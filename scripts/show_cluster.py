"""
Print the paragraphs in a specific duplicate cluster.

Given a cluster ID (e.g. "32044098672728.587"), looks it up in clusters.json,
retrieves the paragraph text from parquet files, and prints all members.

Requires a barcode index (from build_barcode_index.py) for fast lookups.

Usage:
    python scripts/show_cluster.py 32044098672728.587 \
        --clusters-file DATA/Cluster/clusters.json \
        --parquet-dir DATA/Cluster/parquet_shards \
        --barcode-index DATA/Cluster/barcode_index.json

    # Multiple clusters at once
    python scripts/show_cluster.py 32044098672728.587 32044000000018.19 \
        --clusters-file DATA/Cluster/clusters.json \
        --parquet-dir DATA/Cluster/parquet_shards \
        --barcode-index DATA/Cluster/barcode_index.json

    # Show full text (no truncation)
    python scripts/show_cluster.py 32044098672728.587 \
        --clusters-file DATA/Cluster/clusters.json \
        --parquet-dir DATA/Cluster/parquet_shards \
        --barcode-index DATA/Cluster/barcode_index.json \
        --max-text-len 0

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import ijson
import pyarrow.parquet as pq


BOOK_COLUMNS = ["barcode_src", "middlematter_sentences", "subtopic_paragraph_start_indices"]


def parse_cluster_id(doc_id: str) -> tuple[str, int]:
    """Split 'BARCODE.para_idx' into (barcode, para_idx)."""
    dot = doc_id.rfind(".")
    return doc_id[:dot], int(doc_id[dot + 1 :])


def find_clusters(clusters_path: Path, cluster_ids: set[str]) -> dict[str, list[str]]:
    """
    Find specific clusters by representative ID in clusters.json.

    A cluster can be looked up either by its representative (the key in the JSON)
    or by any member ID. Streams the file to avoid loading it all.
    """
    results: dict[str, list[str]] = {}
    remaining = set(cluster_ids)

    with open(clusters_path, "rb") as f:
        parser = ijson.parse(f)
        current_key = None
        current_members: list[str] = []

        for prefix, event, value in parser:
            if prefix == "clusters" and event == "map_key":
                if current_key is not None and (
                    current_key in remaining or remaining & set(current_members)
                ):
                    # Match by key or by any member
                    matched = {current_key} | (remaining & set(current_members))
                    for match_id in matched:
                        results[match_id] = current_members
                        remaining.discard(match_id)
                current_key = value
                current_members = []
            elif prefix.startswith("clusters.") and event == "string":
                current_members.append(value)

        # Check last cluster
        if current_key is not None and (
            current_key in remaining or remaining & set(current_members)
        ):
            matched = {current_key} | (remaining & set(current_members))
            for match_id in matched:
                results[match_id] = current_members
                remaining.discard(match_id)

    return results


def load_book(
    parquet_dir: Path,
    barcode_index: dict[str, str],
    barcode: str,
    book_cache: dict[str, tuple[list[str], list[int]]],
) -> None:
    """Load a single book's sentences and paragraph starts into cache."""
    if barcode in book_cache:
        return
    if barcode not in barcode_index:
        return

    parquet_path = parquet_dir / barcode_index[barcode]
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=BOOK_COLUMNS, batch_size=200):
        barcodes_list = batch.column("barcode_src").to_pylist()
        for i, bc in enumerate(barcodes_list):
            if bc == barcode:
                sents = batch.column("middlematter_sentences")[i].as_py()
                starts = batch.column("subtopic_paragraph_start_indices")[i].as_py()
                book_cache[barcode] = (sents or [], starts or [])
                return


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
    parser = argparse.ArgumentParser(description="Print paragraphs in a specific duplicate cluster")
    parser.add_argument(
        "cluster_ids",
        nargs="+",
        help="Cluster IDs to look up (e.g. 32044098672728.587)",
    )
    parser.add_argument("--clusters-file", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--barcode-index", type=Path, required=True)
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=0,
        help="Max characters per paragraph (0 = no limit, default: 0)",
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=20,
        help="Max members to display per cluster (0 = all, default: 20)",
    )
    args = parser.parse_args()

    # Load barcode index
    with open(args.barcode_index) as f:
        barcode_index: dict[str, str] = json.load(f)

    # Find the requested clusters
    requested = set(args.cluster_ids)
    print(f"Searching for {len(requested)} cluster(s) in {args.clusters_file} ...", file=sys.stderr)
    found = find_clusters(args.clusters_file, requested)

    not_found = requested - set(found.keys())
    if not_found:
        print(f"Warning: not found: {', '.join(sorted(not_found))}", file=sys.stderr)

    if not found:
        print("No clusters found.", file=sys.stderr)
        sys.exit(1)

    # Truncate large clusters before loading
    max_m = args.max_members
    for cluster_id in list(found.keys()):
        members = found[cluster_id]
        if max_m > 0 and len(members) > max_m:
            print(
                f"  Cluster {cluster_id}: showing {max_m} of {len(members)} members",
                file=sys.stderr,
            )
            found[cluster_id] = members[:max_m]

    # Load needed books
    book_cache: dict[str, tuple[list[str], list[int]]] = {}
    needed_barcodes: set[str] = set()
    for members in found.values():
        for m in members:
            needed_barcodes.add(parse_cluster_id(m)[0])

    print(f"Loading {len(needed_barcodes)} book(s)...", file=sys.stderr)
    for barcode in needed_barcodes:
        load_book(args.parquet_dir, barcode_index, barcode, book_cache)

    # Display
    wrapper = textwrap.TextWrapper(width=100, initial_indent="    ", subsequent_indent="    ")

    for cluster_id in args.cluster_ids:
        if cluster_id not in found:
            continue

        members = found[cluster_id]
        parsed = [parse_cluster_id(m) for m in members]
        books = {bc for bc, _ in parsed}
        scope = "within-book" if len(books) == 1 else f"cross-book ({len(books)} books)"

        print(f"\n{'=' * 80}")
        print(f"Cluster: {cluster_id}  ({len(members)} members, {scope})")
        print(f"{'=' * 80}")

        for i, (barcode, para_idx) in enumerate(parsed):
            print(f"\n  [{i + 1}] {barcode} paragraph {para_idx}")

            text = extract_paragraph(barcode, para_idx, book_cache)
            if text is None:
                if barcode not in barcode_index:
                    print("    (barcode not in index)")
                elif barcode not in book_cache:
                    print("    (book not loaded)")
                else:
                    print("    (paragraph index out of range)")
                continue

            if args.max_text_len > 0 and len(text) > args.max_text_len:
                text = text[: args.max_text_len] + "..."

            for line in wrapper.wrap(text):
                print(line)

    print()


if __name__ == "__main__":
    main()
