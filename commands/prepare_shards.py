"""
prepare_shards.py - download shards from HuggingFace dataset

Download books from institutional/institutional-books-1.0 dataset and partitions
them into shards depending on primary language. There are two sets of shards:
Nupunkt shards (roughly having punctuation similar to English) and SaT shards
(different).
"""

import csv
from pathlib import Path
from typing import Iterator

import click
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from const.languages import is_nupunkt_language
from const.types import BookJSON, ManifestStats, ShardStats
from utils.atomic_write import atomic_write_jsonl


def stream_books_from_hf(
    dataset_name: str = "institutional/institutional-books-1.0", split: str = "train"
) -> Iterator[BookJSON]:
    """
    Stream book records
    """
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for record in dataset:  # type: ignore
        yield dict(record)  # type: ignore


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
    """
    # TODO: add basic validation on max_books and shard_size
    logger.info(f"Preparing shards for HF:{dataset_name}:{split} to {output_dir}.")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    nupunkt_queue: list[BookJSON] = []
    sat_queue: list[BookJSON] = []
    manifest_entries: list[ManifestStats] = []

    next_shard_id: int = 1

    def flush_queue(queue: list[BookJSON], segmenter: str) -> int:
        """Flush queue to shard file and return new shard id"""
        nonlocal next_shard_id
        if not queue:
            return next_shard_id

        shard_id = next_shard_id
        next_shard_id += 1

        filename = make_shard_filename(shard_id, segmenter, use_gzip)
        shard_path = raw_dir / filename
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
        logger.info(f"Writing shard to {shard_path}...")
        return next_shard_id

    total_books = 0
    nupunkt_books = 0
    sat_books = 0

    logger.debug("Starting to stream books...")
    IB1_SIZE = 983004
    total = IB1_SIZE
    if max_books and max_books < IB1_SIZE:
        total = max_books
    with tqdm(total=total) as pbar:
        for book in stream_books_from_hf(dataset_name, split):
            segmenter = determine_segmenter(book)
            if segmenter == "nupunkt":
                nupunkt_queue.append(book)
                nupunkt_books += 1
                if len(nupunkt_queue) >= shard_size:
                    flush_queue(nupunkt_queue, "nupunkt")
            else:
                sat_queue.append(book)
                sat_books += 1
                if len(sat_queue) >= shard_size:
                    flush_queue(sat_queue, "sat")

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
    help="Maximum number of books to process (default None, i.e. no limit))",
)
@click.option(
    "--gzip/--no-gzip",
    default=False,
    help="Compress shard files with gzip (default: no compression)",
)
def main(output_dir: Path, shard_size: int, dataset: str, split: str, max_books: int | None, gzip: bool):
    """
    Downloads books and partitions them into shards based on Nupunkt compatability.
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
