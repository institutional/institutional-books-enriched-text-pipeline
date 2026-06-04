"""
step13_annotate.py - build annotated strings with semantic tags

This step produces annotations from earlier computation:

1. Annotates frontmatter/backmatter pages with endmatter classification tags
2. Annotates middlematter with section/paragraph/duplicate tags
3. Includes BPB scores from optional .bpb.jsonl files

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
from utils.jsonl_io import load_bpb_map


def annotate_book(
    book: BookJSON,
    em_classifier,
    bpb_map: dict[str, list[float]],
) -> BookJSON:
    """
    Annotate a single book with semantic tags.

    Args:
        book: Book dictionary with frontmatter, middlematter, backmatter.
        em_classifier: Endmatter subclassifier model.
        bpb_map: Map of book_id to BPB values.

    Returns:
        Book with annotated_frontmatter, annotated_middlematter, annotated_backmatter.
    """
    book_id = book.get("barcode_src", "UNKNOWN")
    if book_id == "UNKNOWN":
        raise ValueError("Unknown book in annotation")

    bpb_values = bpb_map.get(book_id)

    frontmatter = book.get("frontmatter", [])
    book["annotated_frontmatter"] = annotate_frontmatter(frontmatter, em_classifier)

    annotated_mm, lang_dist = annotate_middlematter(book, bpb_values)
    book["annotated_middlematter"] = annotated_mm
    book["language_distribution_gen"] = lang_dist

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
    "--bpb-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional .bpb.jsonl file with bits-per-byte values",
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
    bpb_file: Path | None,
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
            --bpb-file DATA/bpb/shard0001.bpb.jsonl \\
            --config-file config.yaml
    """
    config = load_config(config_file) if config_file else PipelineConfig()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    em_classifier = load_em_subclassifier(config.model_paths.em_subclassifier)
    logger.info(f"Loaded em_subclassifier from {config.model_paths.em_subclassifier}")

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
                annotated = annotate_book(book, em_classifier, bpb_map)
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
