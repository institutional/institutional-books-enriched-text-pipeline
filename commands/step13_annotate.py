"""
step13_annotate.py - build annotated strings with semantic tags

This step produces annotations from earlier computation:

1. Annotates frontmatter/backmatter pages with endmatter classification tags
2. Annotates middlematter with section/paragraph/duplicate tags
3. Includes perplexity scores from optional .perplexity.jsonl files

Produces:
- annotated_frontmatter: List of tagged page strings
- annotated_middlematter: Single annotated string with all tags
- annotated_backmatter: List of tagged page strings
"""

import json
from pathlib import Path

import click
from loguru import logger

from const.config import PipelineConfig, load_config
from const.types import BookJSON
from library.annotate.endmatter import (
    annotate_backmatter,
    annotate_frontmatter,
    load_em_subclassifier,
)
from library.annotate.middlematter import annotate_middlematter
from utils.jsonl_io import load_perplexity_map


def annotate_book(
    book: BookJSON,
    em_classifier,
    perp_map: dict[str, list[float]],
) -> BookJSON:
    """
    Annotate a single book with semantic tags.

    Args:
        book: Book dictionary with frontmatter, middlematter, backmatter.
        em_classifier: Endmatter subclassifier model.
        perp_map: Map of book_id to perplexity values.

    Returns:
        Book with annotated_frontmatter, annotated_middlematter, annotated_backmatter.
    """
    book_id = book.get("barcode_src", "UNKNOWN")
    if book_id == "UNKNOWN":
        raise ValueError("Unknown book in annotation")

    # Get perplexities for this book if available
    perplexities = perp_map.get(book_id)

    # Annotate frontmatter
    frontmatter = book.get("frontmatter", [])
    book["annotated_frontmatter"] = annotate_frontmatter(frontmatter, em_classifier)

    # Annotate middlematter
    book["annotated_middlematter"] = annotate_middlematter(book, perplexities)

    # Annotate backmatter
    backmatter = book.get("backmatter", [])
    book["annotated_backmatter"] = annotate_backmatter(backmatter, em_classifier)

    return book


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file with books to annotate",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file for annotated books",
)
@click.option(
    "--perplexity-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional .perplexity.jsonl file with perplexity values",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional config file (YAML)",
)
def main(
    input_file: Path,
    output_file: Path,
    perplexity_file: Path | None,
    config_file: Path | None,
):
    """
    Annotate books with semantic tags.

    Reads books with frontmatter/middlematter/backmatter and produces
    annotated versions with HTML semantic tags.

    Example:
        python -m commands.step13_annotate \\
            --input-file DATA/shards/processed/shard0001.complete.jsonl \\
            --output-file DATA/shards/annotated/shard0001.annotated.jsonl \\
            --perplexity-file DATA/perplexity/shard0001.perplexity.jsonl \\
            --config-file config.yaml
    """
    config = load_config(config_file) if config_file else PipelineConfig()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load models
    em_classifier = load_em_subclassifier(config.model_paths.em_subclassifier)
    logger.info(f"Loaded em_subclassifier from {config.model_paths.em_subclassifier}")

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
                annotated = annotate_book(book, em_classifier, perp_map)
                f_out.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                books_processed += 1
            except Exception as e:
                logger.error(f"Failed to annotate {book_id}: {e}")
                book["annotation_error"] = str(e)
                f_out.write(json.dumps(book, ensure_ascii=False) + "\n")
                books_failed += 1

    logger.info(f"Annotated {books_processed} books to {output_file}")
    if books_failed:
        logger.warning(f"{books_failed} books failed annotation")


if __name__ == "__main__":
    main()
