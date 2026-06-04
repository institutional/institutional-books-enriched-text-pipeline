"""
dedup_mark_removed.py - Mark paragraphs from specified clusters for removal.

This is an optional phase after dedup_annotate. Unlike other steps, this
specifically requires manual investigation.

Reads a JSON file listing cluster IDs to remove (e.g., library metadata),
then marks all paragraphs belonging to those clusters (both representatives
and duplicates) in the shard's books.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

import click
from loguru import logger

from const.types import BookJSON
from utils.atomic_write import atomic_write_jsonl


def mark_removed_in_book(
    book: BookJSON,
    removed_clusters: set[str],
) -> tuple[BookJSON, int]:
    """
    Mark paragraphs belonging to removed clusters.

    `removed_clusters` should consist of cluster IDs to remove (e.g., {"bookA:5", "bookB:12"}).

    Returns (updated book, num pars marked for removal).
    """
    book_id = book.get("barcode_src", "")
    duplicate_paras = book.get("duplicate_paragraphs", {})
    representative_paras = book.get("representative_paragraphs", {})

    removed: dict[str, str] = {}

    # Check duplicates
    for para_idx, cluster_ref in duplicate_paras.items():
        if cluster_ref in removed_clusters:
            removed[para_idx] = "removed_cluster"

    # Check representatives
    for para_idx in representative_paras:
        cluster_id = f"{book_id}:{para_idx}"
        if cluster_id in removed_clusters:
            removed[para_idx] = "removed_cluster"

    if removed:
        book["removed_paragraphs"] = removed

    return book, len(removed)


@click.command()
@click.option(
    "--shard-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Shard JSONL file to update (will be overwritten)",
)
@click.option(
    "--removal-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="JSON file listing cluster IDs to remove",
)
def main(shard_file: Path, removal_file: Path):
    """
    Mark paragraphs from specified clusters for removal.

    The removal file should be a JSON array of cluster IDs, e.g.:
    ["bookA:5", "bookB:12"]

    All paragraphs that are members of those clusters (both the representative
    and all duplicates) will be marked in removed_paragraphs.

    Example:
        python -m commands.dedup_mark_removed \\
            --shard-file DATA/shards/processed/shard0001.complete.jsonl \\
            --removal-file DATA/dedup/removed_clusters.json
    """
    with open(removal_file) as f:
        removed_clusters = set(json.load(f))

    logger.info(f"Loaded {len(removed_clusters)} cluster IDs for removal")

    books: list[BookJSON] = []
    total_removed = 0
    books_affected = 0

    with open(shard_file) as f:
        for line in f:
            book = json.loads(line)
            book, n_removed = mark_removed_in_book(book, removed_clusters)
            books.append(book)
            total_removed += n_removed
            if n_removed > 0:
                books_affected += 1

    atomic_write_jsonl(iter(books), shard_file)

    logger.info(
        f"Marked {total_removed} paragraphs for removal "
        + f"across {books_affected} books in {shard_file}"
    )


if __name__ == "__main__":
    main()
