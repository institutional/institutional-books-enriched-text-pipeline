"""
dedup_build_lookup.py - Invert clusters.json into per-shard lookup sidecars.

This is phase 3 of the deduplication workflow, between find_duplicates and
annotate:
1. Compute simhashes
2. Find duplicates
3. Build lookups    (this step)  -> <shard>.lookup.jsonl per shard
4. Annotate

clusters.json is parsed once and per-shard lookup files are written
with lines of the form

    <member_doc_id>\\t<representative_doc_id>

Representatives are included automatically (a representative is a member of its
own cluster, producing the line "<rep>\\t<rep>").

Institutional Books - Enriched Text - 2026
"""

import os
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import click
import ijson
from loguru import logger
from tqdm import tqdm

COMPLETE_SUFFIX = ".complete.jsonl"
LOOKUP_SUFFIX = ".lookup.jsonl"

# Buffered TSV lines held across all shards before flushing to disk. Caps memory
# (~80 MB at 2M lines) while keeping the number of file-reopen rounds small.
FLUSH_THRESHOLD = 2_000_000

# barcode_src is the first field of every book line, so this regex matches near
# the start of the line. Much cheaper than full json parse.
BARCODE_RE = re.compile(r'"barcode_src"\s*:\s*"([^"]+)"')


def shard_stem(path: Path) -> str:
    """'shard0001.complete.jsonl' -> 'shard0001'."""
    name = path.name
    if name.endswith(COMPLETE_SUFFIX):
        return name[: -len(COMPLETE_SUFFIX)]
    return path.stem


def barcode_of(doc_id: str) -> str:
    """'32044000000018.107' -> '32044000000018'."""
    return doc_id.rsplit(".", 1)[0]


def index_shard_file(path: Path) -> tuple[str, list[str]]:
    """Return (shard_stem, [barcodes]) for one shard.

    Extracts only barcode_src, avoiding a full JSON parse of every book.
    """
    barcodes: list[str] = []
    with open(path) as f:
        for line in f:
            match = BARCODE_RE.search(line)
            if match:
                barcodes.append(match.group(1))
    return shard_stem(path), barcodes


@click.command()
@click.option(
    "--shard-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory of *.complete.jsonl shard files.",
)
@click.option(
    "--clusters-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Clusters JSON file from dedup_find_duplicates.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write per-shard <stem>.lookup.jsonl sidecars.",
)
@click.option(
    "--shard-glob",
    default="*.complete.jsonl",
    show_default=True,
    help="Glob for shard files within --shard-dir.",
)
@click.option(
    "--workers",
    type=int,
    default=min(12, os.cpu_count() or 1),
    show_default=True,
    help="Parallel workers for the barcode-indexing phase.",
)
def main(
    shard_dir: Path,
    clusters_file: Path,
    output_dir: Path,
    shard_glob: str,
    workers: int,
):
    """
    Invert clusters.json into one small lookup sidecar per shard.

    Enables extremely fast and parallel annotations.

    Example:
        python -m commands.dedup_build_lookup \\
            --shard-dir DATA/shards/processed \\
            --clusters-file DATA/dedup/clusters.json \\
            --output-dir DATA/dedup/lookups
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(shard_dir.glob(shard_glob))
    if not shard_files:
        raise click.ClickException(f"No files matching {shard_glob!r} in {shard_dir}")

    logger.info(
        f"Building per-shard lookups for {len(shard_files)} shards from "
        f"{clusters_file} -> {output_dir}"
    )

    # Phase 1: build barcode -> shard map. Create an (empty) lookup file for every
    # shard up front so each shard has a sidecar, even one with no clustered
    # paragraphs. Indexing is parallelized across files since reading ~all shard
    # bytes is the dominant cost.
    workers = max(1, min(workers, len(shard_files)))
    logger.info(f"Phase 1/2: indexing shard barcodes with {workers} worker(s)...")
    for sf in shard_files:
        (output_dir / f"{shard_stem(sf)}{LOOKUP_SUFFIX}").write_text("")

    barcode_to_shard: dict[str, str] = {}

    def consume(results) -> None:
        for stem, barcodes in tqdm(
            results, total=len(shard_files), desc="Indexing shards", unit="shard"
        ):
            for barcode in barcodes:
                barcode_to_shard[barcode] = stem

    if workers == 1:
        consume(index_shard_file(sf) for sf in shard_files)
    else:
        with Pool(workers) as pool:
            consume(pool.imap_unordered(index_shard_file, shard_files))

    logger.info(f"Indexed {len(barcode_to_shard):,} barcodes across {len(shard_files)} shards")

    # Phase 2: single streaming pass over clusters.json, routing each member to
    # its shard's buffer; flush periodically to bound memory.
    buffers: dict[str, list[str]] = defaultdict(list)
    buffered = 0
    written = 0
    missing = 0
    n_clusters = 0

    def flush() -> None:
        nonlocal buffered, written
        for stem, lines in buffers.items():
            if lines:
                with open(output_dir / f"{stem}{LOOKUP_SUFFIX}", "a") as out:
                    out.write("".join(lines))
                written += len(lines)
        buffers.clear()
        buffered = 0

    logger.info(f"Phase 2/2: streaming {clusters_file} and routing cluster members...")
    with open(clusters_file, "rb") as f:
        for rep, members in tqdm(
            ijson.kvitems(f, "clusters"), desc="Routing clusters", unit="cluster"
        ):
            n_clusters += 1
            for m in members:
                stem = barcode_to_shard.get(barcode_of(m))
                if stem is None:
                    missing += 1
                    continue
                buffers[stem].append(f"{m}\t{rep}\n")
                buffered += 1
            if buffered >= FLUSH_THRESHOLD:
                flush()
    flush()

    logger.info(
        f"Done: scanned {n_clusters:,} clusters, wrote {written:,} lookup entries "
        f"across {len(shard_files)} shards into {output_dir}"
    )
    if missing:
        logger.warning(
            f"{missing:,} cluster members had a barcode not present in any shard (skipped)"
        )


if __name__ == "__main__":
    main()
