"""
step14_add_metadata.py - Compute and add per-book metadata statistics.

Compute similar metadata

Computes for middlematter only:
- Basic text stats: token_count, char_count, word_count, sentence_count, etc.
- N-gram stats: bigram/trigram counts (total and unique)
- Tokenizability: o200k_base tokenizability ratio
- Perplexity stats: min, max, median, avg, and percentiles (if computed)
"""

import json
from pathlib import Path

import click
from loguru import logger

from const.types import BookJSON
from library.metadata.perplexity_stats import compute_perplexity_stats
from library.metadata.text_stats import compute_text_stats


def load_perplexity_map(perplexity_file: Path | None) -> dict[str, list[float]]:
    """
    Load perplexity values from a .perplexity.jsonl file.

    Args:
        perplexity_file: Path to perplexity file, or None.

    Returns:
        Dict mapping book_id to list of perplexities.
    """
    if perplexity_file is None or not perplexity_file.exists():
        return {}

    perp_map: dict[str, list[float]] = {}
    with open(perplexity_file) as f:
        for line in f:
            record = json.loads(line)
            perp_map[record["book_id"]] = record["perplexities"]

    return perp_map


def add_metadata(
    book: BookJSON,
    perp_map: dict[str, list[float]],
) -> BookJSON:
    """
    Compute and add metadata statistics to a book.

    Args:
        book: Book dictionary with middlematter_sentences.
        perp_map: Map of book_id to perplexity values.

    Returns:
        Book with metadata field added.
    """
    book_id = book.get("barcode_src", "UNKNOWN")

    # Compute text statistics
    text_stats = compute_text_stats(book)

    # Get perplexity statistics if available
    perplexities = perp_map.get(book_id, [])
    perp_stats = compute_perplexity_stats(perplexities) if perplexities else {}

    # Combine all metadata
    metadata = {**text_stats, **perp_stats}

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
    "--perplexity-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional .perplexity.jsonl file with perplexity values",
)
def main(
    input_file: Path,
    output_file: Path,
    perplexity_file: Path | None,
):
    """
    Compute and add metadata statistics to books.

    Reads annotated books and adds a 'metadata' field with text statistics,
    n-gram counts, tokenizability, and perplexity statistics.

    Example:
        python -m commands.step14_add_metadata \\
            --input-file DATA/shards/annotated/shard0001.annotated.jsonl \\
            --output-file DATA/shards/metadata/shard0001.metadata.jsonl \\
            --perplexity-file DATA/perplexity/shard0001.perplexity.jsonl
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load perplexity map
    perp_map = load_perplexity_map(perplexity_file)
    if perp_map:
        logger.info(f"Loaded perplexities for {len(perp_map)} books")

    books_processed = 0
    books_failed = 0

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            book_id = book.get("barcode_src", "UNKNOWN")

            try:
                with_metadata = add_metadata(book, perp_map)
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
