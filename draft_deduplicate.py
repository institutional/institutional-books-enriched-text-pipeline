"""
This command performs cross-shard deduplication after all shards have
completed processing through steps 1-11.

Deduplication is a three-stage process:
1. Compute simhashes for all paragraphs across all shards
2. Find duplicates using LSH (requires all simhashes in memory)
3. Annotate books with duplicate information
"""

import glob
import json
from pathlib import Path
from typing import Any

import click
from loguru import logger


def compute_simhashes_for_shard(
    input_file: Path,
    output_file: Path,
    ngram_size: int = 9,
) -> int:
    """
    Compute simhashes for all paragraphs in a shard.

    Args:
        input_file: Input JSONL file with processed books
        output_file: Output JSONL file for simhash records
        ngram_size: N-gram size for simhash computation

    Returns:
        Number of records written
    """
    from utils.simhash import simhash128

    output_file.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            book_id = book.get("book_id", "")

            # Get paragraphs from chunked output
            sentences = book.get("middlematter_sentences", [])
            para_starts = book.get("subtopic_paragraph_start_indices", [])

            if not sentences or not para_starts:
                continue

            # Build paragraphs from sentence indices
            for i, start_idx in enumerate(para_starts):
                end_idx = para_starts[i + 1] if i + 1 < len(para_starts) else len(sentences)
                para_text = " ".join(sentences[start_idx:end_idx])

                if not para_text.strip():
                    continue

                # Compute simhash
                h = simhash128(para_text, ngram_size=ngram_size)
                doc_id = f"{book_id}.{i}"

                record = {
                    "doc_id": doc_id,
                    "simhash": f"0x{h:032x}",
                }
                f_out.write(json.dumps(record) + "\n")
                records_written += 1

    return records_written


def find_duplicates(
    simhash_files: list[Path],
    output_file: Path,
    threshold: int = 5,
) -> dict:
    """
    Find duplicate paragraphs across all shards using LSH.

    Args:
        simhash_files: List of simhash JSONL files
        output_file: Output file for cluster information
        threshold: Hamming distance threshold for duplicates

    Returns:
        Statistics dictionary
    """
    from utils.simhash import hamming_distance, extract_bands

    # Load all simhash records
    logger.info(f"Loading simhash records from {len(simhash_files)} files...")
    records: list[tuple[str, int]] = []

    for path in simhash_files:
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                doc_id = data["doc_id"]
                h = int(data["simhash"], 16)
                records.append((doc_id, h))

    logger.info(f"Loaded {len(records)} records")

    # Build LSH index (band -> set of doc_ids)
    logger.info("Building LSH index...")
    band_index: dict[tuple[int, int], set[str]] = {}

    for doc_id, h in records:
        bands = extract_bands(h)
        for band_idx, band_value in enumerate(bands):
            key = (band_idx, band_value)
            if key not in band_index:
                band_index[key] = set()
            band_index[key].add(doc_id)

    # Find candidate pairs
    logger.info("Finding candidate pairs...")
    hash_lookup = {doc_id: h for doc_id, h in records}
    candidates: set[tuple[str, str]] = set()

    for doc_ids in band_index.values():
        if len(doc_ids) < 2:
            continue
        doc_list = sorted(doc_ids)
        for i, d1 in enumerate(doc_list):
            for d2 in doc_list[i + 1 :]:
                candidates.add((d1, d2))

    logger.info(f"Found {len(candidates)} candidate pairs")

    # Verify candidates and build clusters
    logger.info("Verifying candidates...")
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    duplicates_found = 0
    for d1, d2 in candidates:
        h1, h2 = hash_lookup[d1], hash_lookup[d2]
        if hamming_distance(h1, h2) <= threshold:
            union(d1, d2)
            duplicates_found += 1

    logger.info(f"Found {duplicates_found} duplicate pairs")

    # Build clusters
    clusters: dict[str, set[str]] = {}
    for doc_id in hash_lookup:
        rep = find(doc_id)
        if rep not in clusters:
            clusters[rep] = set()
        clusters[rep].add(doc_id)

    # Filter to clusters with multiple members
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    # Determine which docs to keep (representatives)
    to_keep = set(hash_lookup.keys())
    for rep, members in clusters.items():
        for m in members:
            if m != rep:
                to_keep.discard(m)

    # Write output
    output_data = {
        "clusters": {k: sorted(v) for k, v in clusters.items()},
        "to_keep": sorted(to_keep),
        "statistics": {
            "total_records": len(records),
            "duplicate_pairs": duplicates_found,
            "clusters": len(clusters),
            "to_keep": len(to_keep),
            "to_remove": len(records) - len(to_keep),
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Clusters written to: {output_file}")
    return output_data["statistics"]


def annotate_books(
    input_file: Path,
    clusters_file: Path,
    output_file: Path,
) -> dict:
    """
    Annotate books with duplicate information.

    Args:
        input_file: Input JSONL file with processed books
        clusters_file: Clusters JSON file from find_duplicates
        output_file: Output JSONL file with annotated books

    Returns:
        Statistics dictionary
    """
    # Load clusters
    with open(clusters_file) as f:
        cluster_data = json.load(f)

    clusters = cluster_data["clusters"]
    to_keep = set(cluster_data["to_keep"])

    # Build doc_to_rep mapping
    doc_to_rep: dict[str, str] = {}
    for rep, members in clusters.items():
        for m in members:
            doc_to_rep[m] = rep

    stats = {
        "books_processed": 0,
        "paragraphs_processed": 0,
        "duplicates_marked": 0,
        "representatives_marked": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            book_id = book.get("book_id", "")

            para_starts = book.get("subtopic_paragraph_start_indices", [])
            duplicate_annotations: dict[str, str] = {}
            representative_annotations: dict[str, bool] = {}

            for i in range(len(para_starts)):
                doc_id = f"{book_id}.{i}"
                stats["paragraphs_processed"] += 1

                if doc_id in doc_to_rep:
                    rep = doc_to_rep[doc_id]
                    if doc_id == rep:
                        # This paragraph is a representative
                        representative_annotations[str(i)] = True
                        stats["representatives_marked"] += 1
                    else:
                        # This paragraph is a duplicate
                        rep_book, rep_para = rep.rsplit(".", 1)
                        duplicate_annotations[str(i)] = f"{rep_book}:p:{rep_para}"
                        stats["duplicates_marked"] += 1

            if duplicate_annotations:
                book["duplicate_paragraphs"] = duplicate_annotations
            if representative_annotations:
                book["representative_paragraphs"] = representative_annotations

            f_out.write(json.dumps(book, ensure_ascii=False) + "\n")
            stats["books_processed"] += 1

    return stats


def run_deduplicate(
    processed_dir: Path,
    output_dir: Path,
    threshold: int = 5,
    ngram_size: int = 9,
) -> dict:
    """
    Run the full deduplication pipeline.

    Args:
        processed_dir: Directory containing processed shard files
        output_dir: Output directory for results
        threshold: Hamming distance threshold
        ngram_size: N-gram size for simhash

    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    simhash_dir = output_dir / "simhashes"
    simhash_dir.mkdir(exist_ok=True)

    # Stage 1: Compute simhashes for all shards
    logger.info("Stage 1: Computing simhashes...")
    processed_files = sorted(processed_dir.glob("*.jsonl"))

    if not processed_files:
        logger.error(f"No processed shard files found in {processed_dir}")
        return {"error": "No processed files found"}

    simhash_files = []
    total_records = 0

    for proc_file in processed_files:
        simhash_file = simhash_dir / f"{proc_file.stem}_simhash.jsonl"
        simhash_files.append(simhash_file)

        if simhash_file.exists():
            logger.info(f"Simhash file exists, skipping: {simhash_file}")
            # Count existing records
            with open(simhash_file) as f:
                total_records += sum(1 for _ in f)
        else:
            logger.info(f"Computing simhashes: {proc_file.name}")
            count = compute_simhashes_for_shard(proc_file, simhash_file, ngram_size)
            total_records += count
            logger.info(f"  Wrote {count} records")

    logger.info(f"Total simhash records: {total_records}")

    # Stage 2: Find duplicates
    logger.info("\nStage 2: Finding duplicates...")
    clusters_file = output_dir / "clusters.json"
    find_stats = find_duplicates(simhash_files, clusters_file, threshold)

    # Stage 3: Annotate all books
    logger.info("\nStage 3: Annotating books...")
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    annotate_stats = {
        "books_processed": 0,
        "paragraphs_processed": 0,
        "duplicates_marked": 0,
        "representatives_marked": 0,
    }

    for proc_file in processed_files:
        annotated_file = annotated_dir / f"{proc_file.stem}_annotated.jsonl"
        logger.info(f"Annotating: {proc_file.name}")
        stats = annotate_books(proc_file, clusters_file, annotated_file)

        for k, v in stats.items():
            annotate_stats[k] += v

    return {
        "simhash_records": total_records,
        "find_duplicates": find_stats,
        "annotation": annotate_stats,
    }


@click.command()
@click.option(
    "--processed-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing processed shard files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for deduplication results",
)
@click.option(
    "--threshold",
    type=int,
    default=5,
    help="Hamming distance threshold for duplicates (default: 5)",
)
@click.option(
    "--ngram-size",
    type=int,
    default=9,
    help="N-gram size for simhash computation (default: 9)",
)
def main(
    processed_dir: Path,
    output_dir: Path,
    threshold: int,
    ngram_size: int,
):
    """
    Run cross-shard deduplication on processed books.

    This command performs three stages:
    1. Compute simhashes for all paragraphs
    2. Find duplicates using LSH
    3. Annotate books with duplicate information

    Example:
        python commands/deduplicate.py \\
            --processed-dir ./DATA/shards/processed \\
            --output-dir ./DATA/dedup
    """
    click.echo(f"Running deduplication on: {processed_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Threshold: {threshold}")
    click.echo(f"N-gram size: {ngram_size}")

    stats = run_deduplicate(
        processed_dir=processed_dir,
        output_dir=output_dir,
        threshold=threshold,
        ngram_size=ngram_size,
    )

    click.echo("\nResults:")
    click.echo(f"  Simhash records: {stats.get('simhash_records', 0)}")

    if "find_duplicates" in stats:
        fd = stats["find_duplicates"]
        click.echo(f"  Duplicate pairs: {fd.get('duplicate_pairs', 0)}")
        click.echo(f"  Clusters: {fd.get('clusters', 0)}")
        click.echo(f"  To keep: {fd.get('to_keep', 0)}")
        click.echo(f"  To remove: {fd.get('to_remove', 0)}")

    if "annotation" in stats:
        ann = stats["annotation"]
        click.echo(f"  Books annotated: {ann.get('books_processed', 0)}")
        click.echo(f"  Duplicates marked: {ann.get('duplicates_marked', 0)}")
        click.echo(f"  Representatives marked: {ann.get('representatives_marked', 0)}")

    # Write stats
    stats_path = output_dir / "dedup_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    click.echo(f"\nStats written to: {stats_path}")


if __name__ == "__main__":
    main()
