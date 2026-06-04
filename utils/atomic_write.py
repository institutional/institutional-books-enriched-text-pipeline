"""
atomic_write.py - guarantee atomic writes for final files

This is a naive atomic strategy: write to a temporary file first, then rename
the file. On Linux, renaming is atomic. This prevents partial writes from
corrupting output files.

Institutional Books - Enriched Text - 2026
"""

import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator


@contextmanager
def atomic_jsonl_writer(output_path: Path, compress: bool | None = None):
    """
    Context manager for streaming atomic JSONL writes.

    Writes to a temp file, renames on successful exit. Cleans up on failure.

    Usage:
        with atomic_jsonl_writer(path) as writer:
            writer.write_record({"key": "value"})
            writer.write_record({"key": "value2"})
        # File is atomically renamed on exit

    Yields:
        _AtomicJsonlWriter with write_record() method and count property
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress is None:
        compress = output_path.suffix == ".gz" or output_path.name.endswith(".jsonl.gz")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = _AtomicJsonlWriter(tmp_path, compress)

    try:
        writer.open()
        yield writer
        writer.close()
        tmp_path.rename(output_path)
    except Exception:
        writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class _AtomicJsonlWriter:
    """Internal writer class for atomic_jsonl_writer context manager."""

    def __init__(self, path: Path, compress: bool):
        self._path = path
        self._compress = compress
        self._file: IO[str] | None = None
        self._count = 0

    def open(self):
        if self._compress:
            self._file = gzip.open(self._path, "wt", encoding="utf-8")
        else:
            self._file = open(self._path, "w", encoding="utf-8")

    def write_record(self, record: dict[str, Any]) -> None:
        """Write a single record to the file."""
        if self._file is None:
            raise RuntimeError("Writer not open")
        self._file.write(json.dumps(record, ensure_ascii=False))
        self._file.write("\n")
        self._count += 1

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def count(self) -> int:
        return self._count


def atomic_write_jsonl(
    records: Iterator[dict[str, Any]],
    output_path: Path,
    compress: bool | None = None,
) -> int:
    """
    Atomically write records to a JSONL file. Returns the number of records written.

    Args:
        records: Iterator of dictionaries to write
        output_path: Output file path
        compress: If True, write gzipped. If None, auto-detect from .gz extension.

    Returns:
        Number of records written
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect compression from extension if not specified
    if compress is None:
        compress = output_path.suffix == ".gz" or output_path.name.endswith(".jsonl.gz")

    # write to temporary file first
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    count = 0
    try:
        if compress:
            with gzip.open(tmp_path, "wt", encoding="utf8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                    count += 1
        else:
            with open(tmp_path, "w", encoding="utf8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                    count += 1
        # atomic rename
        tmp_path.rename(output_path)
    except Exception:
        # clean up tmp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return count


def atomic_write_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Atomically write a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # write to temporary file first
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        # atmoic rename
        tmp_path.rename(output_path)
    except Exception:
        # clean up tmp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise
