"""
postprocess_shard.py - post-processing orchestrator for steps 13-15.

Handles annotation, metadata computation, and cleanup after deduplication.
"""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import click
from loguru import logger

from commands.steps import POST_STEPS, get_step_range
from const.config import PipelineConfig, load_config
from const.types import BookJSON, ProcessStats
from library.annotate.endmatter import load_em_subclassifier
from library.annotate.middlematter import annotate_middlematter
from library.metadata.perplexity_stats import compute_perplexity_stats
from library.metadata.text_stats import compute_text_stats
from utils.jsonl_io import load_perplexity_map, open_jsonl

# Fields to keep in final output (step15)
KEEP_FIELDS = {
    "barcode_src",
    "language_gen",
    "annotated_frontmatter",
    "annotated_middlematter",
    "annotated_backmatter",
    "metadata",
    "subtopic_paragraph_start_indices",
    "subtopic_section_start_indices",
    "middlematter_sentences",
}


def step13_annotate_book(
    book: BookJSON,
    em_classifier,
    perp_map: dict[str, list[float]],
) -> BookJSON:
    """Step 13: Annotate book with semantic tags."""
    from library.annotate.endmatter import annotate_backmatter, annotate_frontmatter

    book_id = book.get("barcode_src", "UNKNOWN")
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


def step14_add_metadata_book(
    book: BookJSON,
    perp_map: dict[str, list[float]],
) -> BookJSON:
    """Step 14: Compute and add metadata statistics."""
    book_id = book.get("barcode_src", "UNKNOWN")

    # Compute text statistics
    text_stats = compute_text_stats(book)

    # Get perplexity statistics if available
    perplexities = perp_map.get(book_id, [])
    perp_stats = compute_perplexity_stats(perplexities) if perplexities else {}

    # Combine all metadata
    book["metadata"] = {**text_stats, **perp_stats}
    return book


def step15_clean_book(
    book: BookJSON,
    keep_sentences: bool = True,
    keep_indices: bool = True,
) -> BookJSON:
    """Step 15: Remove processing artifacts."""
    keep = KEEP_FIELDS.copy()
    if not keep_sentences:
        keep.discard("middlematter_sentences")
    if not keep_indices:
        keep.discard("subtopic_paragraph_start_indices")
        keep.discard("subtopic_section_start_indices")

    return {k: v for k, v in book.items() if k in keep}


def postprocess_book(
    book: BookJSON,
    steps: list[str],
    em_classifier,
    perp_map: dict[str, list[float]],
    keep_sentences: bool = True,
    keep_indices: bool = True,
) -> tuple[BookJSON, str | None]:
    """
    Process a single book through post-processing steps.

    Returns:
        Tuple of (result_book, error_message or None)
    """
    book_id = book.get("barcode_src", "UNKNOWN")
    current_book = book

    try:
        for step_name in steps:
            if step_name == "step13_annotate":
                current_book = step13_annotate_book(current_book, em_classifier, perp_map)
            elif step_name == "step14_add_metadata":
                current_book = step14_add_metadata_book(current_book, perp_map)
            elif step_name == "step15_clean":
                current_book = step15_clean_book(current_book, keep_sentences, keep_indices)
            else:
                raise ValueError(f"Unknown post-processing step: {step_name}")

        return current_book, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Book {book_id} failed: {error_msg}")
        return current_book, error_msg


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


def postprocess_shard(
    input_file: Path,
    output_file: Path,
    config: PipelineConfig,
    perplexity_file: Path | None = None,
    start_step: str | None = None,
    end_step: str | None = None,
    keep_sentences: bool = True,
    keep_indices: bool = True,
    resume: bool = False,
) -> ProcessStats:
    """
    Post-process all books in a shard through steps 13-15.

    Streams output to avoid holding all books in memory.

    Args:
        input_file: Input JSONL file (after deduplication)
        output_file: Output JSONL file
        config: Pipeline configuration
        perplexity_file: Optional perplexity file
        start_step: First step to run (default: step13_annotate)
        end_step: Last step to run (default: step15_clean)
        keep_sentences: Keep middlematter_sentences in final output
        keep_indices: Keep paragraph/section indices in final output
        resume: If True, resume from previous progress

    Returns:
        Statistics dict with counts
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    steps = get_step_range(start_step, end_step, POST_STEPS)
    logger.info(f"Running steps: {steps[0]} -> {steps[-1]}")

    # Load models if needed
    em_classifier = None
    if "step13_annotate" in steps:
        em_classifier = load_em_subclassifier(config.model_paths.em_subclassifier)
        logger.info(f"Loaded em_subclassifier from {config.model_paths.em_subclassifier}")

    # Load perplexity map
    perp_map = load_perplexity_map(perplexity_file)
    if perp_map:
        logger.info(f"Loaded perplexities for {len(perp_map)} books")

    # Determine if output should be compressed
    use_gzip = output_file.suffix == ".gz" or output_file.name.endswith(".jsonl.gz")

    # Progress file paths (always uncompressed for append efficiency)
    complete_progress = output_file.with_name(
        output_file.name.replace(".jsonl.gz", ".progress.jsonl").replace(
            ".jsonl", ".progress.jsonl"
        )
    )
    incomplete_path = output_file.with_suffix(".incomplete.jsonl")
    incomplete_progress = incomplete_path.with_suffix(".progress.jsonl")

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

                book_id = book.get("barcode_src", f"line_{line_num}")

                # Skip if already processed
                if book_id in processed_barcodes:
                    skipped_count += 1
                    continue

                logger.debug(f"Post-processing book {book_id}")

                result_book, error_msg = postprocess_book(
                    book, steps, em_classifier, perp_map, keep_sentences, keep_indices
                )

                if error_msg is None:
                    result_book["_postprocessing_complete"] = True
                    result_book["_postprocessed_at"] = datetime.now(UTC).isoformat()
                    complete_file.write(json.dumps(result_book, ensure_ascii=False))
                    complete_file.write("\n")
                    complete_file.flush()
                    complete_count += 1
                else:
                    result_book["_postprocessing_complete"] = False
                    result_book["_postprocessing_error"] = error_msg
                    result_book["_postprocessed_at"] = datetime.now(UTC).isoformat()
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
    total_complete = finalize_output(complete_progress, output_file, use_gzip)
    total_incomplete = finalize_output(incomplete_progress, incomplete_path, False)

    if total_incomplete > 0:
        logger.warning(f"Wrote {total_incomplete} failures to {incomplete_path}")

    logger.info(f"Complete: {total_complete}, Incomplete: {total_incomplete}")

    return {
        "total": total_complete + total_incomplete,
        "complete": total_complete,
        "incomplete": total_incomplete,
    }


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file (after deduplication)",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file for final books",
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
    help="Pipeline configuration file (YAML)",
)
@click.option(
    "--start-step",
    type=click.Choice(POST_STEPS),
    default=None,
    help="First step to run (default: step13_annotate)",
)
@click.option(
    "--end-step",
    type=click.Choice(POST_STEPS),
    default=None,
    help="Last step to run (default: step15_clean)",
)
@click.option(
    "--keep-sentences/--no-keep-sentences",
    default=True,
    help="Keep middlematter_sentences in final output (default: keep)",
)
@click.option(
    "--keep-indices/--no-keep-indices",
    default=True,
    help="Keep paragraph/section indices in final output (default: keep)",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous progress (skip already-processed books)",
)
def main(
    input_file: Path,
    output_file: Path,
    perplexity_file: Path | None,
    config_file: Path | None,
    start_step: str | None,
    end_step: str | None,
    keep_sentences: bool,
    keep_indices: bool,
    resume: bool,
):
    """
    Post-process a shard through steps 13-15 (annotate, metadata, clean).

    Reads books after deduplication and produces final annotated output
    with metadata statistics.

    Use --resume to continue from a previous interrupted run.

    Example:
        python -m commands.postprocess_shard \\
            --input-file DATA/shards/processed/shard0001.complete.jsonl \\
            --output-file DATA/shards/final/shard0001.final.jsonl \\
            --perplexity-file DATA/perplexity/shard0001.perplexity.jsonl \\
            --config-file config.yaml
    """
    config = load_config(config_file) if config_file else PipelineConfig()

    logger.info(f"Post-processing {input_file}")
    logger.info(f"Output: {output_file}")
    if resume:
        logger.info("Resume mode enabled")

    stats = postprocess_shard(
        input_file=input_file,
        output_file=output_file,
        config=config,
        perplexity_file=perplexity_file,
        start_step=start_step,
        end_step=end_step,
        keep_sentences=keep_sentences,
        keep_indices=keep_indices,
        resume=resume,
    )

    click.echo("\nPost-processing complete:")
    click.echo(f"  Total books: {stats['total']}")
    click.echo(f"  Complete: {stats['complete']}")
    click.echo(f"  Incomplete: {stats['incomplete']}")


if __name__ == "__main__":
    main()
