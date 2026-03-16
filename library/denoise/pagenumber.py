"""
pagenumber.py - detect stray page numbers
"""

import unicodedata

from loguru import logger

from const.config import PipelineConfig
from const.types import BookJSON, NormPage
from utils.text import numeric_count


def remove_page_numbers_book(
    book: BookJSON,
    config: PipelineConfig | None = None,
    header_lines: int = 5,
    footer_lines: int = 5,
) -> BookJSON:
    """
    Remove page numbers from the first `header_lines` lines and last `footer_lines`
    lines from a book's middlematter.
    """
    middlematter = book.get("middlematter", [])
    if not middlematter:
        book_id = book.get("barcode_src", "UNKNOWN")
        logger.warning(f"{book_id} has no middlematter.")
        raise ValueError(f"{book_id} has no middlematter.")

    cleaned_middlematter = detect_and_remove_page_numbers(
        middlematter,
        header_lines=header_lines,
        footer_lines=footer_lines,
    )

    result = book
    result["middlematter"] = cleaned_middlematter
    return result


def is_probable_page_number_line(
    line: str,
    max_chars: int = 8,
) -> bool:
    """
    Returns True if the given line is likely to be a standalone page number.

    Heuristic:
    - <= max_chars long (excluding whitespace)
    - Consists of all numeric characters, with at most one exception
    """
    s = unicodedata.normalize("NFKC", line).strip()
    if not s:
        return False

    if len(s) > max_chars:
        return False

    numeric, nonnumeric = numeric_count(s)

    # At most one non-numeric character
    if nonnumeric > 1:
        return False
    # At least one numeric character
    if numeric > 0:
        return True
    return False


def detect_and_remove_page_numbers(
    pages: list[NormPage],
    header_lines: int = 5,
    footer_lines: int = 5,
) -> list[NormPage]:
    """
    Detect and remove page numbers from the start and end of pages.
    """
    cleaned_pages: list[str] = []
    for page in pages:
        lines = page.splitlines()
        header_indices = range(min(header_lines, len(lines)))
        footer_indices = range(max(0, len(lines) - footer_lines), len(lines))

        lines_to_remove: set[int] = set()
        for idx in header_indices:
            if is_probable_page_number_line(lines[idx]):
                lines_to_remove.add(idx)
        for idx in footer_indices:
            if is_probable_page_number_line(lines[idx]):
                lines_to_remove.add(idx)

        cleaned_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]
        cleaned_page = "\n".join(cleaned_lines)
        cleaned_pages.append(cleaned_page)
    return cleaned_pages  # type: ignore
