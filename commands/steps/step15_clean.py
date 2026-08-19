"""
step15_clean.py - Remove processing artifacts and produce final output.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

import click
from loguru import logger

# Base fields always kept in final output.
# Intermediate fields (sentences, paragraph/section indices) are added on top of
# this set only when the corresponding --keep-* flag is passed.
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
}


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
    default=False,
    help="Keep middlematter_sentences in output (default: drop)",
)
@click.option(
    "--keep-indices/--no-keep-indices",
    default=False,
    help="Keep paragraph/section indices in output (default: drop)",
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

    # Optionally retain intermediate fields on top of the minimal keep set
    keep = KEEP_FIELDS.copy()
    if keep_sentences:
        keep.add("middlematter_sentences")
    if keep_indices:
        keep.add("subtopic_paragraph_start_indices")
        keep.add("subtopic_section_start_indices")

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
