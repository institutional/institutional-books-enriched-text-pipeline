"""
compute_bpb.py - Compute per-paragraph bits-per-byte scores.

Computes BPB for each paragraph using a causal language model.
Outputs to .bpb.jsonl files for use by postprocess_shard.py.

This is designed to run on GPU nodes separately from the main pipeline.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

import click
import torch
from loguru import logger

from const.config import PipelineConfig, load_config
from library.bpb.compute_bpb import (
    compute_bpb_in_book,
    load_bpb_model,
)
from utils.jsonl_io import open_jsonl


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_processed_book_ids(progress_path: Path) -> set[str]:
    """Load set of already-processed book IDs from a progress file."""
    processed = set()
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        book_id = record.get("book_id")
                        if book_id:
                            processed.add(book_id)
                    except json.JSONDecodeError:
                        continue
    return processed


def finalize_output(progress_path: Path, final_path: Path) -> int:
    """
    Finalize output by renaming progress file to final destination.

    Returns the number of records.
    """
    if not progress_path.exists():
        return 0

    # Count lines
    with open(progress_path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())

    progress_path.rename(final_path)
    return count


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
    help="Output .bpb.jsonl file",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional config file (YAML)",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous progress (skip already-processed books)",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    show_default=True,
    help="Batch size for paragraph inference (tune per GPU memory)",
)
def main(
    input_file: Path,
    output_file: Path,
    config_file: Path | None,
    resume: bool,
    batch_size: int,
):
    """
    Compute bits-per-byte for all paragraphs in a shard.

    Reads books with chunked paragraphs and outputs BPB records:
        {"book_id": "barcode123", "bpb_values": [0.82, 1.45, 0.67, ...]}

    The nth value corresponds to the nth paragraph
    (from subtopic_paragraph_start_indices).

    Use --resume to continue from a previous interrupted run.

    Example:
        python -m commands.compute_bpb \\
            --input-file DATA/shards/processed/shard0001.complete.jsonl \\
            --output-file DATA/bpb/shard0001.bpb.jsonl
    """
    config = load_config(config_file) if config_file else PipelineConfig()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    progress_path = output_file.with_suffix(".progress.jsonl")
    incomplete_path = output_file.with_name(
        output_file.stem + ".incomplete.jsonl"
    )

    processed_book_ids: set[str] = set()
    if resume:
        processed_book_ids = load_processed_book_ids(progress_path)
        if processed_book_ids:
            logger.info(
                f"Resuming: found {len(processed_book_ids)} previously processed books"
            )
        else:
            logger.info("Resuming: no previous progress found, starting fresh")
    else:
        if progress_path.exists():
            progress_path.unlink()

    device = get_device()
    logger.info(f"Loading BPB model on {device}...")
    model, tokenizer = load_bpb_model(config.bpb.model_name, device)

    books_processed = 0
    books_skipped = 0
    books_failed = 0
    total_paragraphs = 0

    progress_file = open(progress_path, "a", encoding="utf-8")
    incomplete_file = open(incomplete_path, "a", encoding="utf-8")

    try:
        with open_jsonl(input_file, "r") as f_in:
            for line in f_in:
                book = json.loads(line)
                book_id = book.get("barcode_src", "")

                if book_id in processed_book_ids:
                    books_skipped += 1
                    continue

                try:
                    result = compute_bpb_in_book(
                        book, model, tokenizer, device, batch_size=batch_size
                    )
                    record = {
                        "book_id": result["book_id"],
                        "bpb_values": result["bpb_values"],
                    }
                    progress_file.write(json.dumps(record) + "\n")
                    progress_file.flush()
                    books_processed += 1
                    total_paragraphs += len(result["bpb_values"])
                except ValueError as e:
                    logger.warning(f"Skipping {book_id}: {e}")
                    books_skipped += 1
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    logger.warning(f"Failed {book_id}: {e}")
                    incomplete_file.write(
                        json.dumps({"book_id": book_id, "error": str(e)}) + "\n"
                    )
                    incomplete_file.flush()
                    books_failed += 1
                    if device == "cuda":
                        torch.cuda.empty_cache()

    finally:
        progress_file.close()
        incomplete_file.close()

    if resume and books_skipped > 0:
        resumed_count = len(processed_book_ids)
        skip_other = books_skipped - resumed_count
        if resumed_count > 0:
            logger.info(f"Skipped {resumed_count} previously processed books")
        if skip_other > 0:
            logger.info(f"Skipped {skip_other} books due to validation errors")

    if books_failed > 0:
        logger.warning(
            f"{books_failed} books failed (OOM/runtime error), logged to {incomplete_path}"
        )

    total_books = finalize_output(progress_path, output_file)

    logger.info(
        f"Processed {books_processed} books this run, "
        f"{total_books} total books, "
        f"computed BPB for {total_paragraphs} paragraphs to {output_file}"
    )


if __name__ == "__main__":
    main()
