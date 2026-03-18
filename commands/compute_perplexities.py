"""
compute_perplexities.py - Compute per-paragraph perplexity scores.

Computes perplexity for each paragraph using a causal language model.
Outputs to .perplexity.jsonl files for use by postprocess_shard.py.

This is designed to run on GPU nodes separately from the main pipeline.
"""

import json
from pathlib import Path

import click
import torch
from loguru import logger

from const.config import PipelineConfig, load_config
from library.perplexity.compute_perplexity import (
    compute_perplexities_in_book,
    load_perplexity_model,
)
from utils.jsonl_io import open_jsonl


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file with chunked books (after step 10)",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output .perplexity.jsonl file",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional config file (YAML)",
)
def main(input_file: Path, output_file: Path, config_file: Path | None):
    """
    Compute perplexity for all paragraphs in a shard.

    Reads books with chunked paragraphs and outputs perplexity records:
        {"book_id": "barcode123", "perplexities": [12.5, 45.2, 8.7, ...]}

    The nth perplexity corresponds to the nth paragraph
    (from subtopic_paragraph_start_indices).

    Example:
        python -m commands.compute_perplexities \\
            --input-file DATA/shards/processed/shard0001.complete.jsonl \\
            --output-file DATA/perplexity/shard0001.perplexity.jsonl
    """
    config = load_config(config_file) if config_file else PipelineConfig()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Loading perplexity model on {device}...")
    model, tokenizer = load_perplexity_model(config.perplexity.model_name, device)

    books_processed = 0
    books_skipped = 0
    total_paragraphs = 0

    with open_jsonl(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            book_id = book.get("barcode_src", "")

            try:
                result = compute_perplexities_in_book(book, model, tokenizer, device)
                record = {
                    "book_id": result["book_id"],
                    "perplexities": result["perplexities"],
                }
                f_out.write(json.dumps(record) + "\n")
                books_processed += 1
                total_paragraphs += len(result["perplexities"])
            except ValueError as e:
                logger.warning(f"Skipping {book_id}: {e}")
                books_skipped += 1

    logger.info(
        f"Processed {books_processed} books, skipped {books_skipped}, "
        + f"computed {total_paragraphs} perplexities to {output_file}"
    )


if __name__ == "__main__":
    main()
