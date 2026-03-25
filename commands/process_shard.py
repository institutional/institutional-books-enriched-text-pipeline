"""
process_shard.py - main shard processing orchestrator

Handles steps 1-10.
"""

import gzip
import importlib
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import click
from loguru import logger

from commands.steps import MAIN_STEPS, get_step_range
from const.config import PipelineConfig, load_config
from const.types import (
    BookJSON,
    ErrorMessage,
    LastCompletedStep,
    ProcessStats,
    StepFunction,
)
from utils.jsonl_io import open_jsonl


def setup_logging(log_file: Path | None):
    """
    Per-shard logging.
    """
    if log_file:
        logger.add(log_file)
    # In principle, set up other logging details. But defaults are acceptable.


def import_step_function(step_name: str) -> StepFunction:
    """
    Dynamically import a step's process_book function.
    """
    module_name = f"commands.{step_name}"
    module = importlib.import_module(module_name)
    return module.process_book


def process_book_through_steps(
    book: BookJSON,
    config: PipelineConfig,
    segmenter: str,
    steps: list[str],
) -> tuple[BookJSON, LastCompletedStep, ErrorMessage | None]:
    """
    Process a single book through a series of steps.

    Returns:
        Tuple of (result_book, last_completed_step, error_message)
        - On success: (processed_book, last_step, None)
        - On failure: (partial_book, last_successful_step, error_message)
    """
    book_id = book.get("barcode_src", "UNKNOWN")
    if book_id == "UNKNOWN":
        logger.warning("Current book is UNKNOWN. Check pipeline for leaks.")
    current_book = book
    last_completed_step: LastCompletedStep = "step0_prestart"

    for step_name in steps:
        try:
            logger.debug(f"Starting {step_name} on {book_id}.")
            step_fn = import_step_function(step_name)
            current_book = step_fn(current_book, config, segmenter)
            last_completed_step = step_name
        except Exception as e:
            error_msg = f"{step_name}: {type(e).__name__}: {str(e)}"
            logger.error(f"Book {book_id} failed at {step_name}: {error_msg}")
            logger.debug(traceback.format_exc())
            return current_book, last_completed_step, error_msg
    logger.debug(f"Steps complete on {book_id}")

    return current_book, last_completed_step, None


def load_processed_barcodes(progress_path: Path) -> set[str]:
    """Load set of already-processed barcodes from a progress file."""
    processed = set()
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        barcode = record.get("barcode_src")
                        if barcode:
                            processed.add(barcode)
                    except json.JSONDecodeError:
                        continue
    return processed


def finalize_output(
    progress_path: Path,
    final_path: Path,
    compress: bool,
) -> int:
    """
    Finalize output by moving progress file to final destination.

    If compress=True, reads progress and writes compressed.
    Otherwise, renames progress to final.

    Returns the number of records.
    """
    if not progress_path.exists():
        return 0

    count = 0
    if compress:
        with (
            open(progress_path, "r", encoding="utf-8") as f_in,
            gzip.open(final_path, "wt", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                f_out.write(line)
                count += 1
        progress_path.unlink()
    else:
        # Count lines before rename
        with open(progress_path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        progress_path.rename(final_path)

    return count


def process_shard(
    input_file: Path,
    output_dir: Path,
    shard_id: str,
    segmenter: str,
    config: PipelineConfig,
    start_step: str | None = None,
    end_step: str | None = None,
    resume: bool = False,
) -> ProcessStats:
    """
    Process all books in a shard through the pipeline steps.

    Args:
        input_file: Input JSONL file
        output_dir: Output directory for results
        shard_id: Shard identifier
        segmenter: Segmenter type ('nupunkt' or 'sat')
        config: Pipeline configuration
        start_step: First step to run (default: first in MAIN_STEPS)
        end_step: Last step to run (default: last in MAIN_STEPS)
        resume: If True, resume from previous progress

    Returns:
        Statistics dict with counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = get_step_range(start_step, end_step, MAIN_STEPS)
    logger.info(f"Running steps: {steps[0]} -> {steps[-1]} on shard {shard_id}.")

    # Set up output paths
    use_gzip = config.use_gzip
    if use_gzip:
        complete_path = output_dir / f"shard{shard_id}.complete.jsonl.gz"
    else:
        complete_path = output_dir / f"shard{shard_id}.complete.jsonl"
    incomplete_path = output_dir / f"shard{shard_id}.incomplete.jsonl"

    # Progress file paths (always uncompressed for append efficiency)
    complete_progress = output_dir / f"shard{shard_id}.complete.progress.jsonl"
    incomplete_progress = output_dir / f"shard{shard_id}.incomplete.progress.jsonl"

    # Load previously processed barcodes if resuming
    processed_barcodes: set[str] = set()
    if resume:
        processed_barcodes = load_processed_barcodes(complete_progress)
        processed_barcodes |= load_processed_barcodes(incomplete_progress)
        if processed_barcodes:
            logger.info(
                f"Resuming: found {len(processed_barcodes)} previously processed books"
            )
        else:
            logger.info("Resuming: no previous progress found, starting fresh")
    else:
        # Fresh start - remove any existing progress files
        if complete_progress.exists():
            complete_progress.unlink()
        if incomplete_progress.exists():
            incomplete_progress.unlink()

    # Open progress files for appending
    complete_file = open(complete_progress, "a", encoding="utf-8")
    incomplete_file = open(incomplete_progress, "a", encoding="utf-8")

    complete_count = 0
    incomplete_count = 0
    skipped_count = 0

    try:
        with open_jsonl(input_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    book = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON: {e}")
                    continue

                book_id = book.get("barcode_src", f"shard_{shard_id}_line_{line_num}")

                # Skip if already processed
                if book_id in processed_barcodes:
                    skipped_count += 1
                    continue

                logger.info(f"Processing book {book_id}  (book #{line_num} in shard)")

                result_book, last_step, error_msg = process_book_through_steps(
                    book, config, segmenter, steps
                )

                if error_msg is None:
                    result_book["_processing_complete"] = True
                    result_book["_last_completed_step"] = last_step
                    result_book["_processed_at"] = datetime.now(UTC).isoformat()
                    complete_file.write(json.dumps(result_book, ensure_ascii=False))
                    complete_file.write("\n")
                    complete_file.flush()
                    complete_count += 1
                    logger.info(f"Book {book_id} completed successfully")
                else:
                    result_book["_processing_complete"] = False
                    result_book["_last_completed_step"] = last_step
                    result_book["_error_message"] = error_msg
                    result_book["_processed_at"] = datetime.now(UTC).isoformat()
                    incomplete_file.write(json.dumps(result_book, ensure_ascii=False))
                    incomplete_file.write("\n")
                    incomplete_file.flush()
                    incomplete_count += 1
                    logger.warning(f"Book {book_id} marked incomplete: {error_msg}")

    finally:
        complete_file.close()
        incomplete_file.close()

    # Account for previously processed books in totals
    if resume and skipped_count > 0:
        logger.info(f"Skipped {skipped_count} previously processed books")

    # Finalize outputs: move progress files to final destinations
    prev_complete = len(load_processed_barcodes(complete_progress)) if resume else 0
    prev_incomplete = len(load_processed_barcodes(incomplete_progress)) if resume else 0

    # For totals, we need to count what's actually in the progress files
    total_complete = finalize_output(complete_progress, complete_path, use_gzip)
    total_incomplete = finalize_output(incomplete_progress, incomplete_path, False)

    logger.info(f"Complete books: {total_complete}, Incomplete: {total_incomplete}")

    return {
        "total": total_complete + total_incomplete,
        "complete": total_complete,
        "incomplete": total_incomplete,
    }


@click.command()
@click.option(
    "--shard-id",
    required=True,
    help="Shard identifier (e.g., '0001')",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("./DATA/shards/raw"),
    help="Input directory containing raw shards",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./DATA/shards/processed"),
    help="Output directory for processed shards",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=Path("./DATA/logs/shards"),
    help="Log directory",
)
@click.option(
    "--segmenter",
    type=click.Choice(["nupunkt", "sat"]),
    required=True,
    help="Segmenter type for this shard",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Pipeline configuration file (YAML)",
)
@click.option(
    "--start-step",
    default=None,
    help="First step to run (default: step01_uniformize_unicode)",
)
@click.option(
    "--end-step",
    default=None,
    help="Last step to run (default: last of main steps)",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous progress (skip already-processed books)",
)
def main(
    shard_id: str,
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    segmenter: str,
    config_file: Path | None,
    start_step: str | None,
    end_step: str | None,
    resume: bool,
):
    """
    Process a shard through the main pipeline (steps 1-10).

    Reads books from input shard, processes each through the configured
    steps, and writes results to complete/incomplete output files.

    Use --resume to continue from a previous interrupted run.
    """
    # Try gzipped first, then fall back to uncompressed
    input_file = input_dir / f"shard{shard_id}_{segmenter}.jsonl.gz"
    if not input_file.exists():
        input_file = input_dir / f"shard{shard_id}_{segmenter}.jsonl"
    if not input_file.exists():
        input_file = input_dir / f"shard{shard_id}.jsonl.gz"
    if not input_file.exists():
        input_file = input_dir / f"shard{shard_id}.jsonl"
    if not input_file.exists():
        raise click.ClickException(f"Input file not found for shard {shard_id}")

    # Load config
    config = load_config(config_file) if config_file else PipelineConfig()

    # Setup logging
    log_file = log_dir / f"shard{shard_id}.log"
    setup_logging(log_file)

    logger.info(f"Processing shard {shard_id}")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Segmenter: {segmenter}")
    if resume:
        logger.info("Resume mode enabled")

    stats = process_shard(
        input_file=input_file,
        output_dir=output_dir,
        shard_id=shard_id,
        segmenter=segmenter,
        config=config,
        start_step=start_step,
        end_step=end_step,
        resume=resume,
    )

    # Info
    click.echo(f"\nShard {shard_id} processing complete:")
    click.echo(f"  Total books: {stats['total']}")
    click.echo(f"  Complete: {stats['complete']}")
    click.echo(f"  Incomplete: {stats['incomplete']}")


if __name__ == "__main__":
    main()
