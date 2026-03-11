"""
atomic_write.py - guarantee atomic writes for final files

This is a naive atomic strategy: write to a temporary file first, then rename
the file. On Linux, renaming is atomic. This prevents partial writes from
corrupting output files.
"""

import json
from pathlib import Path
from typing import Any, Iterator


def atomic_write_jsonl(
    records: Iterator[dict[str, Any]],
    output_path: Path,
) -> int:
    """
    Atomically write records to a JSONL file. Returns the number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # write to temporary file first
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    count = 0
    try:
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
