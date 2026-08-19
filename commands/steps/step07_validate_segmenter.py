"""
step07_validate_segmenter.py - Validate segmenter assignment

Validate that the book is assigned to the correct segmenter shard based on its language.

This step does not modify the book data, but raises an error if the book
is in the wrong shard for its language.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path
from typing import Any

import click
from loguru import logger

from const.config import PipelineConfig, load_config
from const.languages import is_nupunkt_language
from const.types import BookJSON


class SegmenterMismatchError(ValueError):
    """Raised when a book's language doesn't match its assigned segmenter."""

    pass


def validate_segmenter_for_book(
    book: BookJSON,
    segmenter: str,
) -> None:
    """
    Validate that the book's language matches the shard's segmenter.
    """
    lang = book.get("language_gen")
    if not lang:
        book_id = book.get("barcode_src", book.get("book_id", "unknown"))
        lang = "unknown"
        logger.info(f"No 'language_gen' found in book {book_id}")

    if segmenter not in ("nupunkt", "sat"):
        raise ValueError(f"Invalid segmenter: {segmenter}. Must be 'nupunkt' or 'sat'.")

    should_use_nupunkt = is_nupunkt_language(lang)
    using_nupunkt = segmenter == "nupunkt"

    if should_use_nupunkt != using_nupunkt:
        book_id = book.get("barcode_src", book.get("book_id", "unknown"))
        expected = "nupunkt" if should_use_nupunkt else "sat"
        raise SegmenterMismatchError(
            f"Book {book_id} (language={lang}) is in {segmenter} shard "
            + f"but should be in {expected} shard"
        )


def process_book(
    book: dict[str, Any],
    config: PipelineConfig,
    segmenter: str | None = None,
) -> dict[str, Any]:
    if segmenter is None:
        raise ValueError("Segmenter must be provided for validation step")
    validate_segmenter_for_book(book, segmenter)
    return book


@click.command()
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-file", type=click.Path(path_type=Path), required=True)
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--segmenter", type=click.Choice(["nupunkt", "sat"]), required=True)
def main(input_file: Path, output_file: Path, config_file: Path | None, segmenter: str):
    """Run step 07 (segmenter validation) on a JSONL file."""
    config = load_config(config_file) if config_file else PipelineConfig()

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            result = process_book(book, config, segmenter=segmenter)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
