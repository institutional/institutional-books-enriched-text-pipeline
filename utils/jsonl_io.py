"""
jsonl_io.py - utilities for reading/writing JSONL files with optional gzip compression

Provides transparent handling of .jsonl and .jsonl.gz files.
"""

import gzip
import json
from pathlib import Path
from typing import IO, Any, Iterator


def open_jsonl(path: Path | str, mode: str = "r") -> IO[str]:
    """
    Open a JSONL file, automatically handling gzip compression.

    Detects .gz extension and uses gzip.open() accordingly.
    For reading, use mode='r' or 'rt'.
    For writing, use mode='w' or 'wt'.

    Args:
        path: Path to the file
        mode: File mode ('r', 'rt', 'w', 'wt')

    Returns:
        File handle (text mode)
    """
    path = Path(path)

    # Normalize mode to text mode
    if mode in ("r", "rt"):
        text_mode = "rt"
    elif mode in ("w", "wt"):
        text_mode = "wt"
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'r' or 'w'.")

    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, text_mode, encoding="utf-8")  # type: ignore[return-value]
    else:
        return open(path, text_mode, encoding="utf-8")


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """
    Iterate over records in a JSONL file.

    Automatically handles gzip compression based on file extension.

    Args:
        path: Path to the JSONL file

    Yields:
        Parsed JSON records as dictionaries
    """
    with open_jsonl(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_gzipped(path: Path | str) -> bool:
    """Check if a path indicates a gzipped file."""
    path = Path(path)
    return path.suffix == ".gz" or path.name.endswith(".jsonl.gz")


def ensure_gz_suffix(path: Path | str) -> Path:
    """
    Ensure path has .gz suffix for gzipped JSONL files.

    Converts:
        foo.jsonl -> foo.jsonl.gz
        foo.jsonl.gz -> foo.jsonl.gz (unchanged)
    """
    path = Path(path)
    if path.name.endswith(".jsonl.gz"):
        return path
    elif path.suffix == ".jsonl":
        return path.with_suffix(".jsonl.gz")
    else:
        return Path(str(path) + ".gz")


def strip_gz_suffix(path: Path | str) -> Path:
    """
    Remove .gz suffix from path if present.

    Converts:
        foo.jsonl.gz -> foo.jsonl
        foo.jsonl -> foo.jsonl (unchanged)
    """
    path = Path(path)
    if path.name.endswith(".jsonl.gz"):
        return Path(str(path)[:-3])  # Remove .gz
    return path
