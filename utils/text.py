"""
text.py - text processing utilities
"""

import unicodedata


def numeric_count(segment: str) -> tuple[int, int]:
    """
    Count numeric and non-numeric characters in a string.

    Whitespace is ignored. Uses Unicode category to detect numeric characters,
    so this works with non-ASCII numerals

    Returns:
        A tuple of (numeric_count, non_numeric_count)
    """
    numeric = 0
    nonnumeric = 0
    for ch in segment:
        if unicodedata.category(ch).startswith("N"):
            numeric += 1
        elif not ch.isspace():
            nonnumeric += 1
    return numeric, nonnumeric
