"""
dedup_compute_simhashes.py - Compute simhashes for all paragraphs in a shard.

This is phase 1 of the deduplication workflow:

1. Compute simhashes (this step)
2. Find duplicates
3. Build lookups
4. Annotate - parallelizable per shard

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

import click
from loguru import logger

from library.deduplicate.compute_simhashes import compute_simhashes_in_book


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file with chunked books",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file for simhash records",
)
@click.option(
    "--ngram-size",
    type=int,
    default=9,
    help="N-gram size for simhash computation (default: 9)",
)
def main(input_file: Path, output_file: Path, ngram_size: int):
    """
    Compute simhashes for all paragraphs in a shard.

    Reads books with chunked paragraphs and outputs compact simhash records:
        {"book_id": "barcode123", "simhashes": ["0a1b2c...", "3d4e5f...", ...]}

    Example:
        python -m commands.dedup_compute_simhashes \\
            --input-file DATA/shards/processed/shard0001.complete.jsonl \\
            --output-file DATA/dedup/simhashes/shard0001.simhashes.jsonl
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    books_processed = 0
    books_skipped = 0
    total_hashes = 0

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            book_id = book.get("barcode_src", "")

            try:
                result = compute_simhashes_in_book(book, ngram_size=ngram_size)
                # Store hashes as hex strings for portability
                record = {
                    "book_id": result["book_id"],
                    "simhashes": [f"{h:032x}" for h in result["simhashes"]],
                }
                f_out.write(json.dumps(record) + "\n")
                books_processed += 1
                total_hashes += len(result["simhashes"])
            except ValueError as e:
                logger.warning(f"Skipping {book_id}: {e}")
                books_skipped += 1

    logger.info(
        f"Processed {books_processed} books, skipped {books_skipped}, "
        f"wrote {total_hashes} simhashes to {output_file}"
    )


if __name__ == "__main__":
    main()
