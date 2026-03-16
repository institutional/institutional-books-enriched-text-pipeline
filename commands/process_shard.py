"""
process_shard.py - main shard processing orchestrator

Handles steps 1-11.
"""

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
from utils.atomic_write import atomic_write_jsonl


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


def process_shard(
    input_file: Path,
    output_dir: Path,
    shard_id: str,
    segmenter: str,
    config: PipelineConfig,
    start_step: str | None = None,
    end_step: str | None = None,
) -> ProcessStats:
    """
    Process all books in a shard through the pipeline steps.

    Args:
        input_file: Input JSONL file
        output_dir: Output directory for results
        shard_id: Shard identifier
        segmenter: Segmenter type ('nupunkt' or 'sat')
        config: Pipeline configuration
        logger: Logger instance
        start_step: First step to run (default: first in MAIN_STEPS)
        end_step: Last step to run (default: last in MAIN_STEPS)

    Returns:
        Statistics dict with counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = get_step_range(start_step, end_step, MAIN_STEPS)
    logger.info(f"Running steps: {steps[0]} -> {steps[-1]} on shard {shard_id}.")

    complete_books: list[BookJSON] = []
    incomplete_books: list[BookJSON] = []

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                book = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON: {e}")
                continue

            book_id = book.get("barcode_src", f"shard_{shard_id}_line_{line_num}")
            logger.info(f"Processing book {book_id}")

            result_book, last_step, error_msg = process_book_through_steps(
                book, config, segmenter, steps
            )

            if error_msg is None:
                result_book["_processing_complete"] = True
                result_book["_last_completed_step"] = last_step
                result_book["_processed_at"] = datetime.now(UTC).isoformat()
                complete_books.append(result_book)
                logger.info(f"Book {book_id} completed successfully")
            else:
                result_book["_processing_complete"] = False
                result_book["_last_completed_step"] = last_step
                result_book["_error_message"] = error_msg
                result_book["_processed_at"] = datetime.now(UTC).isoformat()
                incomplete_books.append(result_book)
                logger.warning(f"Book {book_id} marked incomplete: {error_msg}")

    # Write outputs atomically
    complete_path = output_dir / f"shard{shard_id}.complete.jsonl"
    incomplete_path = output_dir / f"shard{shard_id}.incomplete.jsonl"
    complete_count = atomic_write_jsonl(iter(complete_books), complete_path)
    incomplete_count = atomic_write_jsonl(iter(incomplete_books), incomplete_path)

    logger.info(f"Complete books: {complete_count}, Incomplete: {incomplete_count}")

    return {
        "total": complete_count + incomplete_count,
        "complete": complete_count,
        "incomplete": incomplete_count,
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
def main(
    shard_id: str,
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    segmenter: str,
    config_file: Path | None,
    start_step: str | None,
    end_step: str | None,
):
    """
    Process a shard through the main pipeline (steps 1-11).

    Reads books from input shard, processes each through the configured
    steps, and writes results to complete/incomplete output files.
    """
    input_file = input_dir / f"shard{shard_id}_{segmenter}.jsonl"
    if not input_file.exists():
        # Try without segmenter suffix
        input_file = input_dir / f"shard{shard_id}.jsonl"
        if not input_file.exists():
            raise click.ClickException(f"Input file not found: {input_file}")

    # Load config
    config = load_config(config_file) if config_file else PipelineConfig()

    # Setup logging
    log_file = log_dir / f"shard{shard_id}.log"
    setup_logging(log_file)

    logger.info(f"Processing shard {shard_id}")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Segmenter: {segmenter}")

    stats = process_shard(
        input_file=input_file,
        output_dir=output_dir,
        shard_id=shard_id,
        segmenter=segmenter,
        config=config,
        start_step=start_step,
        end_step=end_step,
    )

    # Info
    click.echo(f"\nShard {shard_id} processing complete:")
    click.echo(f"  Total books: {stats['total']}")
    click.echo(f"  Complete: {stats['complete']}")
    click.echo(f"  Incomplete: {stats['incomplete']}")


if __name__ == "__main__":
    main()
