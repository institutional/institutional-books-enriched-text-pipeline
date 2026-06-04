"""
step15_clean.py - Remove processing artifacts and produce final output.

This step removes intermediate fields while preserving:
- Core identifiers: barcode_src, language_gen
- Annotated output: annotated_frontmatter, annotated_middlematter, annotated_backmatter
- Metadata: metadata field
- Structural indices: subtopic_paragraph_start_indices, subtopic_section_start_indices
- Sentences: middlematter_sentences (for downstream use)

Removes:
- Intermediate text fields: uniformized_text, frontmatter, middlematter, backmatter
- Processing artifacts: duplicate_paragraphs, representative_paragraphs
- Error fields: annotation_error, metadata_error
"""

import json
from pathlib import Path

import click
from loguru import logger

from const.types import BookJSON

# Fields to keep in final output
KEEP_FIELDS = {
    ## Core identifiers
    "barcode_src",
    "language_gen",
    ## Annotated output
    "annotated_frontmatter",
    "annotated_middlematter",
    "annotated_backmatter",
    ## Metadata
    "metadata",
    ## Structural indices
    # "subtopic_paragraph_start_indices",
    # "subtopic_section_start_indices",
    ## Sentences
    # "middlematter_sentences",
}


def clean_book(book: BookJSON) -> BookJSON:
    """
    Remove processing artifacts from a book.

    Args:
        book: Book dictionary with all fields.

    Returns:
        Cleaned book with only essential fields.
    """
    return {k: v for k, v in book.items() if k in KEEP_FIELDS}


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file with books to clean",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file for cleaned books",
)
@click.option(
    "--keep-sentences/--no-keep-sentences",
    default=True,
    help="Keep middlematter_sentences in output (default: keep)",
)
@click.option(
    "--keep-indices/--no-keep-indices",
    default=True,
    help="Keep paragraph/section indices in output (default: keep)",
)
def main(
    input_file: Path,
    output_file: Path,
    keep_sentences: bool,
    keep_indices: bool,
):
    """
    Remove processing artifacts and produce final output.

    Removes intermediate fields like uniformized_text, frontmatter, etc.
    and keeps only the annotated output, metadata, and core identifiers.

    Example:
        python -m commands.step15_clean \\
            --input-file DATA/shards/metadata/shard0001.metadata.jsonl \\
            --output-file DATA/shards/final/shard0001.final.jsonl
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Adjust keep fields based on options
    keep = KEEP_FIELDS.copy()
    if not keep_sentences:
        keep.discard("middlematter_sentences")
    if not keep_indices:
        keep.discard("subtopic_paragraph_start_indices")
        keep.discard("subtopic_section_start_indices")

    books_processed = 0

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)

            # Clean book using current keep set
            cleaned = {k: v for k, v in book.items() if k in keep}
            f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            books_processed += 1

    logger.info(f"Cleaned {books_processed} books to {output_file}")


if __name__ == "__main__":
    main()
