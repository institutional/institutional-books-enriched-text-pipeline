"""
prepare_shards.py - download shards from HuggingFace dataset

Download books from institutional/institutional-books-1.0 dataset and partitions
them into shards depending on primary language. There are two sets of shards:
Nupunkt shards (roughly having punctuation similar to English) and SaT shards
(different).

Supports interruption and resumption via a progress file.

Institutional Books - Enriched Text - 2026
"""

import csv
import json
from pathlib import Path
from typing import Any, TypedDict

import click
from datasets import IterableDataset, load_dataset
from loguru import logger
from tqdm import tqdm

from const.languages import is_nupunkt_language
from const.types import BookJSON, ManifestStats, ShardStats
from utils.atomic_write import atomic_write_jsonl


class ProgressState(TypedDict):
    """State saved for resumption."""

    hf_state_dict: dict[str, Any]
    books_processed: int
    next_shard_id: int
    nupunkt_books: int
    sat_books: int
    nupunkt_queue: list[BookJSON]
    sat_queue: list[BookJSON]
    manifest_entries: list[ManifestStats]


def load_progress(progress_path: Path) -> ProgressState | None:
    """Load progress state from file if it exists."""
    if not progress_path.exists():
        return None
    try:
        with open(progress_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load progress file: {e}")
        return None


def save_progress(progress_path: Path, state: ProgressState) -> None:
    """Save progress state to file atomically."""
    tmp_path = progress_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    tmp_path.rename(progress_path)


def delete_progress(progress_path: Path) -> None:
    """Delete progress file on successful completion."""
    if progress_path.exists():
        progress_path.unlink()
        logger.info(f"Deleted progress file: {progress_path}")


def determine_segmenter(book: BookJSON) -> str:
    """
    If 'language_gen' is Nupunkt-compatible, use Nupunkt.
    """
    lang: str
    lang = book.get("language_gen", "")
    return "nupunkt" if is_nupunkt_language(lang) else "sat"


def make_shard_filename(shard_id: int, segmenter: str, use_gzip: bool = False) -> str:
    ext = ".jsonl.gz" if use_gzip else ".jsonl"
    return f"shard{shard_id:04d}_{segmenter}{ext}"


def prepare_shards(
    output_dir: Path,
    shard_size: int = 1000,
    dataset_name: str = "institutional/institutional-books-1.0",
    split: str = "train",
    max_books: int | None = None,
    use_gzip: bool = False,
) -> ShardStats:
    """
    Stream HF dataset into shards and return statistics dict with counts.

    Instead of using HF shards, this streams into two types of shards based on later sentence
    segmentation preference.

    Supports interruption and resumption via a progress file.
    """
    logger.info(f"Preparing shards for HF:{dataset_name}:{split} to {output_dir}.")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / "progress.json"

    # Load or initialize state
    progress = load_progress(progress_path)
    if progress:
        logger.info(f"Resuming from progress file: {progress['books_processed']} books processed")
        nupunkt_queue = progress["nupunkt_queue"]
        sat_queue = progress["sat_queue"]
        manifest_entries = progress["manifest_entries"]
        next_shard_id = progress["next_shard_id"]
        total_books = progress["books_processed"]
        nupunkt_books = progress["nupunkt_books"]
        sat_books = progress["sat_books"]
        hf_state_dict = progress["hf_state_dict"]
    else:
        nupunkt_queue = []
        sat_queue = []
        manifest_entries = []
        next_shard_id = 1
        total_books = 0
        nupunkt_books = 0
        sat_books = 0
        hf_state_dict = None

    # Load dataset
    dataset: IterableDataset = load_dataset(dataset_name, split=split, streaming=True)
    if hf_state_dict:
        dataset.load_state_dict(hf_state_dict)

    def flush_queue(queue: list[BookJSON], segmenter: str) -> None:
        """Flush queue to shard file."""
        nonlocal next_shard_id
        if not queue:
            return

        shard_id = next_shard_id
        next_shard_id += 1

        filename = make_shard_filename(shard_id, segmenter, use_gzip)
        shard_path = raw_dir / filename
        logger.info(f"Writing shard to {shard_path}...")
        count = atomic_write_jsonl(iter(queue), shard_path)
        manifest_entries.append(
            {
                "shard_id": f"{shard_id:04d}",
                "filename": filename,
                "segmenter": segmenter,
                "book_count": count,
            }
        )
        queue.clear()

    def save_current_progress() -> None:
        """Save current state to progress file."""
        state: ProgressState = {
            "hf_state_dict": dataset.state_dict(),
            "books_processed": total_books,
            "next_shard_id": next_shard_id,
            "nupunkt_books": nupunkt_books,
            "sat_books": sat_books,
            "nupunkt_queue": nupunkt_queue,
            "sat_queue": sat_queue,
            "manifest_entries": manifest_entries,
        }
        save_progress(progress_path, state)

    logger.debug("Starting to stream books...")
    IB1_SIZE = 983004
    total = IB1_SIZE
    if max_books and max_books < IB1_SIZE:
        total = max_books

    with tqdm(total=total, initial=total_books) as pbar:
        for book in dataset:
            book = dict(book)
            segmenter = determine_segmenter(book)
            if segmenter == "nupunkt":
                nupunkt_queue.append(book)
                nupunkt_books += 1
                if len(nupunkt_queue) >= shard_size:
                    flush_queue(nupunkt_queue, "nupunkt")
                    save_current_progress()
            else:
                sat_queue.append(book)
                sat_books += 1
                if len(sat_queue) >= shard_size:
                    flush_queue(sat_queue, "sat")
                    save_current_progress()

            total_books += 1
            pbar.update(1)
            if max_books and total_books >= max_books:
                break

    logger.info("Book streaming complete.")
    # flush remaining books
    flush_queue(nupunkt_queue, "nupunkt")
    flush_queue(sat_queue, "sat")

    # write manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shard_id", "filename", "segmenter", "book_count"])
        writer.writeheader()
        writer.writerows(manifest_entries)

    # Clean up progress file on successful completion
    delete_progress(progress_path)

    stats: ShardStats = {
        "total_books": total_books,
        "nupunkt_books": nupunkt_books,
        "sat_books": sat_books,
        "nupunkt_shards": sum(1 for e in manifest_entries if e["segmenter"] == "nupunkt"),
        "sat_shards": sum(1 for e in manifest_entries if e["segmenter"] == "sat"),
        "total_shards": len(manifest_entries),
    }
    logger.info(f"Shard statistics: {stats}")
    return stats


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./DATA/shards"),
    help="Output directory for shards (default './DATA/shards')",
)
@click.option(
    "--shard-size",
    type=int,
    default=1000,
    help="Number of books per shard (default 1000)",
)
@click.option(
    "--dataset",
    default="institutional/institutional-books-1.0",
    help="HuggingFace dataset identifier",
)
@click.option(
    "--split",
    default="train",
    help="Dataset split to use",
)
@click.option(
    "--max-books",
    type=int,
    default=None,
    help="Maximum number of books to process (default None, i.e. no limit)",
)
@click.option(
    "--gzip/--no-gzip",
    default=False,
    help="Compress shard files with gzip (default: no compression)",
)
def main(
    output_dir: Path, shard_size: int, dataset: str, split: str, max_books: int | None, gzip: bool
):
    """
    Downloads books and partitions them into shards based on Nupunkt compatibility.
    """
    click.echo(f"Preparing shards from {dataset} ({split})")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Shard size: {shard_size}")

    if max_books:
        click.echo(f"Max books: {max_books}")
    if gzip:
        click.echo("Compression: gzip enabled")

    stats = prepare_shards(
        output_dir=output_dir,
        shard_size=shard_size,
        dataset_name=dataset,
        split=split,
        max_books=max_books,
        use_gzip=gzip,
    )

    click.echo("\nResults:")
    click.echo(f"  Total books: {stats['total_books']}")
    click.echo(f"  Nupunkt books: {stats['nupunkt_books']} ({stats['nupunkt_shards']} shards)")
    click.echo(f"  SAT books: {stats['sat_books']} ({stats['sat_shards']} shards)")
    click.echo(f"  Total shards: {stats['total_shards']}")
    click.echo(f"\nManifest written to: {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
