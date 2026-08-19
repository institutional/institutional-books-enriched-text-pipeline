"""
Build parquet files for HuggingFace upload from final JSONL shards.

Produces files following HuggingFace auto-discovery conventions:
    data/train-00000-of-04916.parquet
    data/train-00001-of-04916.parquet
    ...

Writes one output parquet file per input shard (1-to-1). Flattens metadata
into top-level columns with naming convention:
    _src  = from original book
    _gen  = computed by pipeline

Computes language distribution on-the-fly by parsing data-language
attributes from middlematter. Also produces an opinionated
processed_middlematter_gen column (non-duplicate paragraphs in the
[p10, p90] bits-per-byte band).

I wrote this for actual use. When transferring data to and from the cluster, I
used zstd compression. This is an optionally allowed input format here.

Usage:
    python scripts/build_hf_parquet.py \
        --input-dir DATA/Cluster/output \
        --output-dir DATA/hf_upload
"""

import json
import math
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from library.annotate.language import UNKNOWN

MIN_PARAGRAPHS = 5
LANG_ATTR_RE = re.compile(r'<p\b[^>]*\bdata-language="([^"]+)"')

SHARDS_PER_FILE = 1

SCHEMA = pa.schema(
    [
        ("barcode_src", pa.string()),
        ("primary_language_gen", pa.string()),
        (
            "language_distribution_gen",
            pa.list_(
                pa.struct(
                    [
                        ("language", pa.string()),
                        ("proportion", pa.float64()),
                    ]
                )
            ),
        ),
        ("token_count_gen", pa.int64()),
        ("char_count_gen", pa.int64()),
        ("word_count_gen", pa.int64()),
        ("sentence_count_gen", pa.int64()),
        ("paragraph_count_gen", pa.int64()),
        ("section_count_gen", pa.int64()),
        ("bigram_count_gen", pa.int64()),
        ("bigram_count_unique_gen", pa.int64()),
        ("trigram_count_gen", pa.int64()),
        ("trigram_count_unique_gen", pa.int64()),
        ("tokenizability_ratio_gen", pa.float64()),
        ("bpb_min_gen", pa.float64()),
        ("bpb_max_gen", pa.float64()),
        ("bpb_median_gen", pa.float64()),
        ("bpb_avg_gen", pa.float64()),
        ("bpb_p10_gen", pa.float64()),
        ("bpb_p30_gen", pa.float64()),
        ("bpb_p70_gen", pa.float64()),
        ("bpb_p90_gen", pa.float64()),
        ("processed_middlematter_gen", pa.string()),
        ("frontmatter_gen", pa.string()),
        ("middlematter_gen", pa.string()),
        ("backmatter_gen", pa.string()),
    ]
)


def compute_language_distribution(
    annotated_middlematter: str,
) -> list[dict[str, str | float]]:
    """Compute language distribution by parsing data-language attributes.

    Returns list of {"language": ..., "proportion": ...} dicts sorted by frequency.
    """
    languages = LANG_ATTR_RE.findall(annotated_middlematter)

    if not languages:
        return []

    counts = Counter(lang for lang in languages if lang != UNKNOWN)

    total_known = sum(counts.values())
    if total_known == 0:
        return []

    filtered = {lang: count for lang, count in counts.items() if count >= MIN_PARAGRAPHS}

    if not filtered:
        return []

    sorted_langs = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    return [
        {"language": lang, "proportion": round(count / total_known, 4)}
        for lang, count in sorted_langs
    ]


class FilteredExtractor(HTMLParser):
    """Extract non-duplicate paragraphs within the [p10, p90] bpb band."""

    def __init__(self, p10: float, p90: float):
        super().__init__()
        self.p10 = p10
        self.p90 = p90
        self.paragraphs: list[str] = []
        self._current_text: list[str] = []
        self._in_duplicate = False
        self._in_p = False
        self._current_bpb: float | None = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "aside":
            self._in_duplicate = True
        elif tag == "p":
            self._in_p = True
            self._current_text = []
            bpb = attrs_dict.get("data-bpb")
            self._current_bpb = float(bpb) if bpb else None

    def handle_endtag(self, tag):
        if tag == "aside":
            self._in_duplicate = False
        elif tag == "p":
            if (
                not self._in_duplicate
                and self._current_bpb is not None
                and self.p10 <= self._current_bpb <= self.p90
            ):
                self.paragraphs.append("".join(self._current_text))
            self._in_p = False
            self._current_bpb = None

    def handle_data(self, data):
        if self._in_p:
            self._current_text.append(data)


def extract_processed_middlematter(
    annotated_middlematter: str, p10: float | None, p90: float | None
) -> str:
    """Extract filtered middlematter text from annotated HTML."""
    if not annotated_middlematter or p10 is None or p90 is None:
        return ""

    parser = FilteredExtractor(p10, p90)
    parser.feed(annotated_middlematter)
    return "\n\n".join(parser.paragraphs)


def transform_book(book: dict) -> dict:
    """Transform a final JSONL book into the HuggingFace column format."""
    metadata = book.get("metadata", {})
    annotated_mm = book.get("annotated_middlematter", "")

    lang_dist = compute_language_distribution(annotated_mm)

    p10 = metadata.get("bpb_p10")
    p90 = metadata.get("bpb_p90")
    processed_mm = extract_processed_middlematter(annotated_mm, p10, p90)

    return {
        "barcode_src": book.get("barcode_src", ""),
        "primary_language_gen": book.get("language_gen", ""),
        "language_distribution_gen": lang_dist,
        "token_count_gen": metadata.get("token_count", 0),
        "char_count_gen": metadata.get("char_count", 0),
        "word_count_gen": metadata.get("word_count", 0),
        # metadata["sentence_count"] is the pipeline segmenter's count (len of
        # middlematter_sentences at step 14), so it stays consistent with the
        # paragraph/section structure and survives even when the raw sentence
        # list is dropped from the final output.
        "sentence_count_gen": metadata.get("sentence_count", 0),
        "paragraph_count_gen": metadata.get("paragraph_count", 0),
        "section_count_gen": metadata.get("section_count", 0),
        "bigram_count_gen": metadata.get("bigram_count", 0),
        "bigram_count_unique_gen": metadata.get("bigram_count_unique", 0),
        "trigram_count_gen": metadata.get("trigram_count", 0),
        "trigram_count_unique_gen": metadata.get("trigram_count_unique", 0),
        "tokenizability_ratio_gen": metadata.get("tokenizability_o200k_base_ratio", 0.0),
        "bpb_min_gen": metadata.get("bpb_min"),
        "bpb_max_gen": metadata.get("bpb_max"),
        "bpb_median_gen": metadata.get("bpb_median"),
        "bpb_avg_gen": metadata.get("bpb_avg"),
        "bpb_p10_gen": metadata.get("bpb_p10"),
        "bpb_p30_gen": metadata.get("bpb_p30"),
        "bpb_p70_gen": metadata.get("bpb_p70"),
        "bpb_p90_gen": metadata.get("bpb_p90"),
        "processed_middlematter_gen": processed_mm,
        "frontmatter_gen": book.get("annotated_frontmatter", ""),
        "middlematter_gen": book.get("annotated_middlematter", ""),
        "backmatter_gen": book.get("annotated_backmatter", ""),
    }


def load_shard(path: Path) -> list[dict]:
    """Load a (.jsonl or .jsonl.zst) shard and transform each book.

    .zst shards are decompressed by streaming through the `zstd` binary, so no
    Python zstd library and no on-disk decompressed copy is needed.
    """
    rows = []
    if path.suffix == ".zst":
        proc = subprocess.Popen(
            ["zstd", "-dc", str(path)], stdout=subprocess.PIPE, text=True, encoding="utf-8"
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.strip():
                rows.append(transform_book(json.loads(line)))
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"zstd failed to decompress {path} (exit {proc.returncode})")
    else:
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(transform_book(json.loads(line)))
    return rows


def process_batch(
    file_idx: int,
    batch_files: list[Path],
    data_dir: Path,
    num_output_files: int,
) -> tuple[int, list[str]]:
    """Process a batch of shards into one parquet file.

    Writes to a temp file and atomically renames on success, so a partial
    file from a crash will never be mistaken for a completed one.

    Returns (number of books written, list of failed shard names).
    """
    out_name = f"train-{file_idx:05d}-of-{num_output_files:05d}.parquet"
    out_path = data_dir / out_name
    tmp_path = data_dir / f".{out_name}.tmp"

    rows = []
    failed_shards = []
    for shard_file in batch_files:
        try:
            rows.extend(load_shard(shard_file))
        except Exception as e:
            failed_shards.append(f"{shard_file.name}: {e}")

    if not rows:
        return 0, failed_shards

    table = pa.Table.from_pylist(rows, schema=SCHEMA)

    pq.write_table(
        table,
        tmp_path,
        row_group_size=len(table),
        compression="snappy",
        write_page_index=True,
        use_content_defined_chunking=True,
    )

    tmp_path.rename(out_path)

    return len(rows), failed_shards


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing *.final.jsonl files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory (will contain data/ subdirectory)",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of parallel workers (default: 1)",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Skip batches whose output parquet already exists (default: resume)",
)
def main(input_dir: Path, output_dir: Path, workers: int, resume: bool):
    """
    Build HuggingFace-ready parquet files from final JSONL shards.

    Groups SHARDS_PER_FILE input shards into each output parquet file.
    Output follows HuggingFace naming convention for auto-discovery.
    Supports resuming after crashes by skipping already-written files.
    """
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Prefer compressed shards if present (decompressed on the fly via the zstd
    # binary); fall back to plain .jsonl.
    input_files = sorted(input_dir.glob("*.final.jsonl.zst")) or sorted(
        input_dir.glob("*.final.jsonl")
    )
    if not input_files:
        logger.error(f"No *.final.jsonl[.zst] files found in {input_dir}")
        return

    num_output_files = math.ceil(len(input_files) / SHARDS_PER_FILE)

    # Build batch list, skipping already-completed batches if resuming
    batches = []
    skipped = 0
    for file_idx in range(num_output_files):
        out_name = f"train-{file_idx:05d}-of-{num_output_files:05d}.parquet"
        out_path = data_dir / out_name

        if resume and out_path.exists():
            skipped += 1
            continue

        start = file_idx * SHARDS_PER_FILE
        end = min(start + SHARDS_PER_FILE, len(input_files))
        batches.append((file_idx, input_files[start:end]))

    workers = min(workers, max(len(batches), 1), os.cpu_count() or 1)

    if skipped:
        logger.info(f"Resuming: skipping {skipped} already-completed parquet files")

    if not batches:
        logger.info("All parquet files already exist, nothing to do")
        return

    logger.info(
        f"Converting {len(batches)} batches ({len(batches) * SHARDS_PER_FILE} shards) "
        f"into parquet files with {workers} workers"
    )

    total_books = 0
    all_failed_shards = []

    if workers == 1:
        for file_idx, batch_files in tqdm(batches, desc="Writing parquet files"):
            try:
                count, failed = process_batch(file_idx, batch_files, data_dir, num_output_files)
                total_books += count
                all_failed_shards.extend(failed)
            except Exception as e:
                shard_names = [f.name for f in batch_files]
                logger.error(f"Batch {file_idx} crashed: {e} (shards: {shard_names})")
                all_failed_shards.extend(f"{name}: batch crash - {e}" for name in shard_names)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, file_idx, batch_files, data_dir, num_output_files): (
                    file_idx,
                    batch_files,
                )
                for file_idx, batch_files in batches
            }
            with tqdm(total=len(future_to_batch), desc="Writing parquet files") as pbar:
                for future in as_completed(future_to_batch):
                    file_idx, batch_files = future_to_batch[future]
                    try:
                        count, failed = future.result()
                        total_books += count
                        all_failed_shards.extend(failed)
                    except Exception as e:
                        shard_names = [f.name for f in batch_files]
                        logger.error(f"Batch {file_idx} crashed: {e} (shards: {shard_names})")
                        all_failed_shards.extend(
                            f"{name}: batch crash - {e}" for name in shard_names
                        )
                    pbar.update(1)

    logger.info(f"Wrote {total_books} books to parquet files in {data_dir}")

    if all_failed_shards:
        failed_log = output_dir / "failed_shards.log"
        with open(failed_log, "w") as f:
            f.write("\n".join(all_failed_shards) + "\n")
        logger.warning(f"{len(all_failed_shards)} shards failed. See {failed_log}")


if __name__ == "__main__":
    main()
