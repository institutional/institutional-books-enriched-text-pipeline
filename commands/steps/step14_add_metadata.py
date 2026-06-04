"""
step14_add_metadata.py - Compute and add per-book metadata statistics.

Computes for middlematter only:
- Basic text stats: token_count, char_count, word_count, sentence_count, etc.
- N-gram stats: bigram/trigram counts (total and unique)
- Tokenizability: o200k_base tokenizability ratio
- BPB stats: min, max, median, avg, and percentiles (if computed)

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

import click
from loguru import logger

from const.types import BookJSON
from library.metadata.bpb_stats import compute_bpb_stats
from library.metadata.text_stats import compute_text_stats
from utils.jsonl_io import load_bpb_map


def add_metadata(
    book: BookJSON,
    bpb_map: dict[str, list[float]],
) -> BookJSON:
    """
    Compute and add metadata statistics to a book.

    Args:
        book: Book dictionary with middlematter_sentences.
        bpb_map: Map of book_id to BPB values.

    Returns:
        Book with metadata field added.
    """
    book_id = book.get("barcode_src", "UNKNOWN")

    text_stats = compute_text_stats(book)

    bpb_values = bpb_map.get(book_id, [])
    bpb_stats = compute_bpb_stats(bpb_values) if bpb_values else {}

    lang_dist = book.get("language_distribution_gen", {"languages": [], "proportion": []})

    metadata = {**text_stats, **bpb_stats, **lang_dist}

    book["metadata"] = metadata
    return book


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file with annotated books",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file with metadata added",
)
@click.option(
    "--bpb-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional .bpb.jsonl file with bits-per-byte values",
)
def main(
    input_file: Path,
    output_file: Path,
    bpb_file: Path | None,
):
    """
    Compute and add metadata statistics to books.

    Reads annotated books and adds a 'metadata' field with text statistics,
    n-gram counts, tokenizability, and BPB statistics.

    Example:
        python -m commands.step14_add_metadata \\
            --input-file DATA/shards/annotated/shard0001.annotated.jsonl \\
            --output-file DATA/shards/metadata/shard0001.metadata.jsonl \\
            --bpb-file DATA/bpb/shard0001.bpb.jsonl
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    bpb_map = load_bpb_map(bpb_file)
    if bpb_map:
        logger.info(f"Loaded BPB values for {len(bpb_map)} books")

    books_processed = 0
    books_failed = 0

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            book_id = book.get("barcode_src", "UNKNOWN")

            try:
                with_metadata = add_metadata(book, bpb_map)
                f_out.write(json.dumps(with_metadata, ensure_ascii=False) + "\n")
                books_processed += 1
            except Exception as e:
                logger.error(f"Failed to compute metadata for {book_id}: {e}")
                book["metadata_error"] = str(e)
                f_out.write(json.dumps(book, ensure_ascii=False) + "\n")
                books_failed += 1

    logger.info(f"Added metadata to {books_processed} books in {output_file}")
    if books_failed:
        logger.warning(f"{books_failed} books failed metadata computation")


if __name__ == "__main__":
    main()
