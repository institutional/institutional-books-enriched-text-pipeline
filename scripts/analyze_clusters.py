"""
Analyze duplicate clusters from dedup_find_duplicates output.

Streams the clusters.json file to avoid loading ~5GB into memory at once.
Produces summary statistics and detailed breakdowns.

Usage:
    python scripts/analyze_clusters.py DATA/Cluster/clusters.json
    python scripts/analyze_clusters.py DATA/Cluster/clusters.json --top-pairs 50
    python scripts/analyze_clusters.py DATA/Cluster/clusters.json \\
        --book-metadata DATA/Cluster/book_metadata.jsonl

Generate the book metadata file first:
    python scripts/extract_book_metadata.py \\
        --parquet-dir DATA/Cluster/parquet_shards \\
        --output-file DATA/Cluster/book_metadata.jsonl

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import ijson


def parse_cluster_id(doc_id: str) -> tuple[str, int]:
    """Split 'BARCODE.para_idx' into (barcode, para_idx)."""
    dot = doc_id.rfind(".")
    return doc_id[:dot], int(doc_id[dot + 1 :])


def stream_clusters(path: Path):
    """
    Stream clusters from clusters.json using ijson (SAX-style parsing).

    Yields (representative_id, [member_id, ...]) tuples.
    """
    with open(path, "rb") as f:
        # ijson iterates over "clusters.KEY.item" entries
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

        # Yield last cluster
        if current_key is not None:
            yield current_key, current_members


def load_book_metadata(path: Path) -> dict[str, dict]:
    """
    Load book metadata from JSONL file produced by extract_book_metadata.py.

    Returns dict: barcode -> {"language": str, "n_paragraphs": int}
    """
    meta: dict[str, dict] = {}
    print(f"Loading book metadata from {path} ...")
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            meta[d["barcode"]] = {
                "language": d["language"],
                "n_paragraphs": d["n_paragraphs"],
            }
    print(f"  Loaded metadata for {len(meta):,} books.")
    return meta


class BookUnionFind:
    """Simple string-keyed union-find for book-level connected components."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rep_id: dict[str, str] = {}  # representative cluster ID for each root

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rep_id[x] = ""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str, rep_id: str = "") -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            # Preserve the non-empty rep_id from the merged root
            if self.rep_id.get(ra) and rep_id:
                self.rep_id[rb] = f"{self.rep_id[ra]}|{rep_id}"
            elif rep_id:
                self.rep_id[rb] = rep_id
            elif self.rep_id.get(ra):
                self.rep_id[rb] = self.rep_id[ra]

    def get_components(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        components: dict[str, list[str]] = {}
        rep_ids: dict[str, str] = {}
        for x in self.parent:
            root = self.find(x)
            components.setdefault(root, []).append(x)
            rep_ids[root] = self.rep_id.get(root, "")
        return components, rep_ids


def main():
    parser = argparse.ArgumentParser(description="Analyze dedup clusters")
    parser.add_argument("clusters_file", type=Path)
    parser.add_argument("--top-pairs", type=int, default=30, help="Number of top book pairs to show")
    parser.add_argument("--top-books", type=int, default=30, help="Number of top books to show")
    parser.add_argument(
        "--book-metadata",
        type=Path,
        default=None,
        help="JSONL file from extract_book_metadata.py (enables language + density analysis)",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=int,
        default=1,
        help="Minimum shared duplicate paragraphs to link books in connected components (default: 1)",
    )
    args = parser.parse_args()

    if not args.clusters_file.exists():
        print(f"File not found: {args.clusters_file}", file=sys.stderr)
        sys.exit(1)

    # Optionally load book metadata (language, paragraph counts)
    book_meta: dict[str, dict] | None = None
    if args.book_metadata:
        book_meta = load_book_metadata(args.book_metadata)

    # ── Accumulators ──
    cluster_sizes: Counter[int] = Counter()  # size → count
    books_with_cross_dupes: set[str] = set()  # books in any cross-book cluster
    books_with_within_dupes: set[str] = set()  # books in any within-book cluster
    book_cross_dup_count: Counter[str] = Counter()  # barcode → paragraphs in cross-book clusters
    book_within_dup_count: Counter[str] = Counter()  # barcode → paragraphs in within-book clusters
    book_pair_shared: Counter[tuple[str, str]] = Counter()  # (book_a, book_b) → shared paragraphs
    book_pair_rep_ids: dict[tuple[str, str], list[str]] = {}  # (book_a, book_b) → cluster IDs
    cross_book_clusters = 0
    within_book_clusters = 0
    para_indices: list[int] = []  # sample of paragraph indices (capped)
    cluster_book_span: Counter[int] = Counter()  # num distinct books in cluster → count
    total_dup_paragraphs = 0
    max_cluster_size = 0
    max_cluster_id = ""

    PARA_INDEX_SAMPLE_LIMIT = 5_000_000  # cap to avoid memory issues

    print(f"Streaming clusters from {args.clusters_file} ...")
    n_clusters = 0

    for rep_id, members in stream_clusters(args.clusters_file):
        n_clusters += 1
        size = len(members)
        cluster_sizes[size] += 1
        total_dup_paragraphs += size

        if size > max_cluster_size:
            max_cluster_size = size
            max_cluster_id = rep_id

        # Parse all members
        parsed = [parse_cluster_id(m) for m in members]
        books_in_cluster: set[str] = set()

        for barcode, para_idx in parsed:
            books_in_cluster.add(barcode)
            if len(para_indices) < PARA_INDEX_SAMPLE_LIMIT:
                para_indices.append(para_idx)

        distinct_books = len(books_in_cluster)
        cluster_book_span[distinct_books] += 1

        if distinct_books == 1:
            within_book_clusters += 1
            for barcode, _ in parsed:
                books_with_within_dupes.add(barcode)
                book_within_dup_count[barcode] += 1
        else:
            cross_book_clusters += 1
            for barcode, _ in parsed:
                books_with_cross_dupes.add(barcode)
                book_cross_dup_count[barcode] += 1
            book_list = sorted(books_in_cluster)
            # Count book pairs (cap at 20 books to avoid combinatorial explosion)
            if distinct_books <= 20:
                for i in range(len(book_list)):
                    for j in range(i + 1, len(book_list)):
                        pair = (book_list[i], book_list[j])
                        book_pair_shared[pair] += 1
                        if pair not in book_pair_rep_ids:
                            book_pair_rep_ids[pair] = []
                        book_pair_rep_ids[pair].append(rep_id)

        if n_clusters % 5_000_000 == 0:
            print(f"  ... processed {n_clusters:,} clusters", file=sys.stderr)

    print(f"  Done. Processed {n_clusters:,} clusters.\n")

    # ── Report ──

    print("=" * 70)
    print("DUPLICATE CLUSTER ANALYSIS")
    print("=" * 70)

    # Derived: combined sets/counters
    books_with_any_dupes = books_with_cross_dupes | books_with_within_dupes
    books_with_only_within = books_with_within_dupes - books_with_cross_dupes
    book_total_dup_count: Counter[str] = Counter()
    book_total_dup_count.update(book_cross_dup_count)
    book_total_dup_count.update(book_within_dup_count)

    # 1. High-level stats
    print("\n--- Overview ---")
    print(f"  Total clusters:                        {n_clusters:,}")
    print(f"  Total duplicate paragraphs:            {total_dup_paragraphs:,}")
    print(f"  Books with any duplicates:             {len(books_with_any_dupes):,}")
    print(f"    - with cross-book duplicates:        {len(books_with_cross_dupes):,}")
    print(f"    - with within-book duplicates only:  {len(books_with_only_within):,}")

    # 4. Cluster size statistics
    print("\n--- Cluster Size Statistics ---")
    sizes = []
    for sz, count in cluster_sizes.items():
        sizes.extend([sz] * count)
    sizes.sort()

    if sizes:
        total = len(sizes)
        mean = sum(sizes) / total
        median = sizes[total // 2]
        p90 = sizes[int(total * 0.90)]
        p95 = sizes[int(total * 0.95)]
        p99 = sizes[int(total * 0.99)]
        mode_size, mode_count = cluster_sizes.most_common(1)[0]

        print(f"  Mean size:    {mean:.2f}")
        print(f"  Median size:  {median}")
        print(f"  Mode size:    {mode_size} (appears {mode_count:,} times)")
        print(f"  P90:          {p90}")
        print(f"  P95:          {p95}")
        print(f"  P99:          {p99}")
        print(f"  Max size:     {max_cluster_size} (cluster {max_cluster_id})")

    print("\n  Size distribution:")
    for sz in sorted(cluster_sizes.keys()):
        count = cluster_sizes[sz]
        if sz <= 10 or count >= 1000:
            pct = 100 * count / n_clusters
            print(f"    size {sz:>6}: {count:>12,} clusters ({pct:>5.1f}%)")
        elif sz == 11:
            print(f"    ...")

    # 5. Cross-book vs within-book
    print("\n--- Cross-book vs Within-book ---")
    print(f"  Within-book clusters (same book):    {within_book_clusters:,} ({100*within_book_clusters/n_clusters:.1f}%)")
    print(f"  Cross-book clusters (diff books):    {cross_book_clusters:,} ({100*cross_book_clusters/n_clusters:.1f}%)")

    print("\n  Books spanned per cluster:")
    for span in sorted(cluster_book_span.keys())[:20]:
        count = cluster_book_span[span]
        print(f"    {span:>3} books: {count:>12,} clusters")
    if max(cluster_book_span.keys()) > 20:
        print(f"    ... (max: {max(cluster_book_span.keys())} books)")

    # 3. Books with very many duplicates
    print(f"\n--- Top {args.top_books} Books by Cross-Book Duplicate Paragraph Count ---")
    for barcode, count in book_cross_dup_count.most_common(args.top_books):
        extra = ""
        if book_meta and barcode in book_meta:
            total_paras = book_meta[barcode]["n_paragraphs"]
            lang = book_meta[barcode]["language"] or "unknown"
            pct = 100 * count / total_paras if total_paras > 0 else 0
            extra = f" ({pct:.1f}% of {total_paras:,} paragraphs, {lang})"
        print(f"  {barcode}: {count:,} cross-book dup paragraphs{extra}")

    print(f"\n--- Top {args.top_books} Books by Within-Book Duplicate Paragraph Count ---")
    for barcode, count in book_within_dup_count.most_common(args.top_books):
        extra = ""
        if book_meta and barcode in book_meta:
            total_paras = book_meta[barcode]["n_paragraphs"]
            lang = book_meta[barcode]["language"] or "unknown"
            pct = 100 * count / total_paras if total_paras > 0 else 0
            extra = f" ({pct:.1f}% of {total_paras:,} paragraphs, {lang})"
        print(f"  {barcode}: {count:,} within-book dup paragraphs{extra}")

    # 4. Book-pair affinity (most shared duplicates)
    print(f"\n--- Top {args.top_pairs} Book Pairs by Shared Duplicates ---")
    for (a, b), count in book_pair_shared.most_common(args.top_pairs):
        extra = ""
        if book_meta:
            la = book_meta.get(a, {}).get("language", "?")
            lb = book_meta.get(b, {}).get("language", "?")
            extra = f" [{la}/{lb}]"
        print(f"  {a} <-> {b}: {count:,} shared duplicate paragraphs{extra}")

    # 5. Paragraph position bias
    if para_indices:
        para_indices.sort()
        n = len(para_indices)
        mean_idx = sum(para_indices) / n
        median_idx = para_indices[n // 2]
        p10_idx = para_indices[int(n * 0.10)]
        p25_idx = para_indices[int(n * 0.25)]
        p75_idx = para_indices[int(n * 0.75)]
        p90_idx = para_indices[int(n * 0.90)]

        print(f"\n--- Paragraph Position Bias (sampled {n:,} paragraphs) ---")
        print(f"  Mean paragraph index:    {mean_idx:.1f}")
        print(f"  Median paragraph index:  {median_idx}")
        print(f"  P10:                     {p10_idx}")
        print(f"  P25:                     {p25_idx}")
        print(f"  P75:                     {p75_idx}")
        print(f"  P90:                     {p90_idx}")

        buckets: Counter[str] = Counter()
        for idx in para_indices:
            if idx < 10:
                buckets["0-9"] += 1
            elif idx < 50:
                buckets["10-49"] += 1
            elif idx < 100:
                buckets["50-99"] += 1
            elif idx < 500:
                buckets["100-499"] += 1
            elif idx < 1000:
                buckets["500-999"] += 1
            else:
                buckets["1000+"] += 1

        print("  Position distribution:")
        for label in ["0-9", "10-49", "50-99", "100-499", "500-999", "1000+"]:
            c = buckets.get(label, 0)
            print(f"    para idx {label:>8}: {c:>12,} ({100*c/n:.1f}%)")

    # 6. Duplicate density per book
    if book_meta:
        print("\n--- Duplicate Density Distribution (cross-book) ---")
        densities: list[float] = []
        for barcode in books_with_cross_dupes:
            if barcode in book_meta:
                total_paras = book_meta[barcode]["n_paragraphs"]
                if total_paras > 0:
                    density = book_cross_dup_count[barcode] / total_paras
                    densities.append(density)
        densities.sort()
        if densities:
            n_d = len(densities)
            print(f"  Books with density data:  {n_d:,}")
            print(f"  Mean density:             {sum(densities)/n_d:.3f}")
            print(f"  Median density:           {densities[n_d//2]:.3f}")
            print(f"  P90 density:              {densities[int(n_d*0.90)]:.3f}")
            print(f"  P99 density:              {densities[int(n_d*0.99)]:.3f}")

            density_buckets: Counter[str] = Counter()
            for d in densities:
                if d < 0.01:
                    density_buckets["<1%"] += 1
                elif d < 0.05:
                    density_buckets["1-5%"] += 1
                elif d < 0.10:
                    density_buckets["5-10%"] += 1
                elif d < 0.25:
                    density_buckets["10-25%"] += 1
                elif d < 0.50:
                    density_buckets["25-50%"] += 1
                elif d < 0.75:
                    density_buckets["50-75%"] += 1
                else:
                    density_buckets["75-100%"] += 1

            print("  Density distribution:")
            for label in ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-100%"]:
                c = density_buckets.get(label, 0)
                print(f"    {label:>8} duplicated: {c:>10,} books ({100*c/n_d:.1f}%)")

        # Top books by cross-book duplicate percentage
        book_densities: list[tuple[float, str]] = []
        for barcode in books_with_cross_dupes:
            if barcode in book_meta:
                total_paras = book_meta[barcode]["n_paragraphs"]
                if total_paras > 0:
                    pct = book_cross_dup_count[barcode] / total_paras
                    book_densities.append((pct, barcode))
        book_densities.sort(reverse=True)

        print(f"\n--- Top {args.top_books} Books by Cross-Book Duplicate Percentage ---")
        for pct, barcode in book_densities[: args.top_books]:
            dup_count = book_cross_dup_count[barcode]
            total_paras = book_meta[barcode]["n_paragraphs"]
            lang = book_meta[barcode]["language"] or "unknown"
            print(
                f"  {barcode}: {100*pct:.1f}%"
                f" ({dup_count:,} / {total_paras:,} paragraphs, {lang})"
            )
    else:
        print("\n--- Duplicate Density ---")
        print("  Skipped (pass --book-metadata to enable)")

    # 7. Language distribution of books with duplicates
    if book_meta:
        print("\n--- Language Distribution (books with cross-book duplicates) ---")
        lang_counter: Counter[str] = Counter()
        lang_all: Counter[str] = Counter()
        for barcode in books_with_cross_dupes:
            lang = book_meta.get(barcode, {}).get("language") or "unknown"
            lang_counter[lang] += 1
        for meta in book_meta.values():
            lang_all[meta["language"] or "unknown"] += 1

        print(f"  {'Language':<12} {'With dupes':>12} {'Total books':>12} {'Dup rate':>10}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        for lang, dup_count in lang_counter.most_common(30):
            total_count = lang_all.get(lang, 0)
            rate = 100 * dup_count / total_count if total_count > 0 else 0
            print(f"  {lang:<12} {dup_count:>12,} {total_count:>12,} {rate:>9.1f}%")
    else:
        print("\n--- Language Distribution ---")
        print("  Skipped (pass --book-metadata to enable)")

    # 8. Book-level connected components
    min_w = args.min_edge_weight
    print(f"\n--- Book-Level Connected Components (min edge weight: {min_w}) ---")
    book_uf = BookUnionFind()
    for (a, b), weight in book_pair_shared.items():
        if weight >= min_w:
            book_uf.union(a, b, book_pair_rep_ids.get((a, b), [""])[0] if book_pair_rep_ids.get((a, b)) else "")
    # Add singleton books (have duplicates but only within-book, no qualifying edges)
    for barcode in books_with_any_dupes:
        book_uf.find(barcode)
    components, rep_ids = book_uf.get_components()
    comp_sizes = sorted([len(members) for members in components.values()], reverse=True)
    n_components = len(comp_sizes)
    singleton_components = sum(1 for s in comp_sizes if s == 1)
    multi_components = n_components - singleton_components

    print(f"  Total components:      {n_components:,}")
    print(f"  Singletons (1 book):   {singleton_components:,}")
    print(f"  Multi-book families:   {multi_components:,}")
    if comp_sizes:
        print(f"  Largest family:        {comp_sizes[0]:,} books")
        if len(comp_sizes) > 1:
            print(f"  2nd largest:           {comp_sizes[1]:,} books")
        if len(comp_sizes) > 2:
            print(f"  3rd largest:           {comp_sizes[2]:,} books")

    # Print largest families with cluster IDs
    if multi_components > 0:
        print("\n  Largest multi-book families:")
        sorted_components = sorted(components.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (root, members) in enumerate(sorted_components[:5]):
            if len(members) > 1:
                rep_id = rep_ids.get(root, "N/A")
                print(f"\n    Family #{i+1} (size: {len(members):,} books):")
                print(f"      Representative cluster ID: {rep_id}")
                print(f"      Books: {', '.join(sorted(members)[:10])}" + ("..." if len(members) > 10 else ""))


if __name__ == "__main__":
    main()
