"""
dedup_annotate.py - Annotate books with duplicate information.

This is phase 3 of the deduplication workflow:
1. Compute simhashes
2. Find duplicates
3. Annotate (this step)
"""

import json
from pathlib import Path

import click
import ijson
from loguru import logger

from const.types import BookJSON
from utils.atomic_write import atomic_write_jsonl


def annotate_book(
    book: BookJSON,
    doc_to_rep: dict[str, str],
) -> BookJSON:
    """
    Annotate a single book with duplicate information.

    Adds:
    - duplicate_paragraphs: {"5": "other_book:3"} - paragraph 5 is dup of other_book paragraph 3
    - representative_paragraphs: {"3": true} - paragraph 3 is a cluster representative
    """
    book_id = book.get("barcode_src", "")
    para_starts = book.get("subtopic_paragraph_start_indices", [])

    duplicate_annotations: dict[str, str] = {}
    representative_annotations: dict[str, bool] = {}

    for i in range(len(para_starts)):
        doc_id = f"{book_id}.{i}"

        if doc_id in doc_to_rep:
            rep = doc_to_rep[doc_id]
            if doc_id == rep:
                # is a cluster representative
                representative_annotations[str(i)] = True
            else:
                # is a duplicate of rep
                rep_book, rep_para = rep.rsplit(".", 1)
                duplicate_annotations[str(i)] = f"{rep_book}:{rep_para}"

    if duplicate_annotations:
        book["duplicate_paragraphs"] = duplicate_annotations
    if representative_annotations:
        book["representative_paragraphs"] = representative_annotations

    return book


@click.command()
@click.option(
    "--shard-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Shard JSONL file to annotate (will be overwritten)",
)
@click.option(
    "--clusters-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Clusters JSON file from dedup_find_duplicates",
)
def main(shard_file: Path, clusters_file: Path):
    """
    Annotate books in a shard with duplicate information.

    Errors go to .incomplete.jsonl file.

    Example:
        python -m commands.dedup_annotate \\
            --shard-file DATA/shards/processed/shard0001.complete.jsonl \\
            --clusters-file DATA/dedup/clusters.json
    """
    ## Low Memory Strategy ##
    #
    # In practice, the clusters.json file may be large and there may be many
    # shards we want to process in parallel. For each shard, we identify
    # barcodes we care about. Then we stream the clusters.json file and keep
    # only the relevant portions. And then we process the shard in place.

    # Identify relevant barcodes
    shard_barcodes: set[str] = set()
    with open(shard_file) as f:
        for line in f:
            barcode = json.loads(line).get("barcode_src", "")
            if barcode:
                shard_barcodes.add(barcode)
    logger.info(f"Found {len(shard_barcodes)} barcodes in shard")

    # Extract relevant portions from clusters file
    doc_to_rep: dict[str, str] = {}
    clusters_scanned = 0
    with open(clusters_file, "rb") as f:
        for rep, members in ijson.kvitems(f, "clusters"):
            clusters_scanned += 1
            # Check if any member belongs to this shard
            relevant = [m for m in members if m.rsplit(".", 1)[0] in shard_barcodes]
            if relevant:
                for m in relevant:
                    doc_to_rep[m] = rep
            if clusters_scanned % 5_000_000 == 0:
                logger.info(f"Scanned {clusters_scanned} clusters, kept {len(doc_to_rep)} entries")

    logger.info(
        f"Scanned {clusters_scanned} clusters, "
        + f"built lookup with {len(doc_to_rep)} entries for this shard"
    )

    # Process books
    complete_books: list[BookJSON] = []
    incomplete_books: list[BookJSON] = []
    stats = {
        "books_processed": 0,
        "books_failed": 0,
        "duplicates_marked": 0,
        "representatives_marked": 0,
    }

    with open(shard_file) as f:
        for line in f:
            book = json.loads(line)
            book_id = book.get("barcode_src", "UNKNOWN")
            if book_id == "UNKNOWN":
                raise ValueError("Unknown book detected.")

            try:
                annotated = annotate_book(book, doc_to_rep)
                complete_books.append(annotated)
                stats["books_processed"] += 1

                # Count annotations
                stats["duplicates_marked"] += len(annotated.get("duplicate_paragraphs", {}))
                stats["representatives_marked"] += len(
                    annotated.get("representative_paragraphs", {})
                )
            except Exception as e:
                logger.error(f"Failed to annotate {book_id}: {e}")
                book["annotation_error"] = str(e)
                incomplete_books.append(book)
                stats["books_failed"] += 1

            if (stats["books_processed"] + stats["books_failed"]) % 1000 == 0:
                logger.info(
                    f"Progress: {stats['books_processed'] + stats['books_failed']} books processed"
                )

    logger.info(f"Writing {len(complete_books)} annotated books to {shard_file}")
    atomic_write_jsonl(iter(complete_books), shard_file)

    if incomplete_books:
        incomplete_file = shard_file.with_suffix(".incomplete.jsonl")
        # Append to existing incomplete file
        with open(incomplete_file, "a") as f:
            for book in incomplete_books:
                f.write(json.dumps(book, ensure_ascii=False) + "\n")
        logger.warning(f"Wrote {len(incomplete_books)} failures to {incomplete_file}")

    logger.info(
        f"Annotated {stats['books_processed']} books in {shard_file}: "
        + f"{stats['duplicates_marked']} duplicates, "
        + f"{stats['representatives_marked']} representatives"
    )
    if stats["books_failed"]:
        logger.warning(f"{stats['books_failed']} books failed")


if __name__ == "__main__":
    main()
