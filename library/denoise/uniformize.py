"""
uniformize.py - unicode normalization
"""

import re
import unicodedata
from typing import cast

from const.types import BookJSON, CharTokens, NormPage, NormText, RawPage, RawText


def uniformize_book(book: BookJSON) -> BookJSON:
    """
    Uniformize unicode characters in a book's text.

    Reads from 'text_by_page_src' and creates 'uniformized_text'.
    """
    book_text: list[RawPage] | None = book.get("text_by_page_src")

    if not book_text:
        book_id = book.get("barcode_src", "UNKNOWN_BARCODE")
        raise ValueError(f"No 'text_by_page_src' found in book {book_id}")

    uniformized_pages = [normalize_unicode_in_page(raw_page) for raw_page in book_text]
    result = book
    result["uniformized_text"] = uniformized_pages

    # remove original text fields
    result.pop("text_by_page_src", None)
    result.pop("text_by_page_gen", None)
    result.pop("text_analysis_gen", None)

    return result


def normalize_unicode_in_page(raw_page: RawPage) -> NormPage:
    """
    Normalize each line of text in a page.

    Preserves line breaks.
    """
    normalized_lines = [normalize_text(RawText(line)) for line in raw_page.splitlines()]
    joined = "\n".join(cast(list[str], normalized_lines))  # python type warts
    return cast(NormPage, joined)


# All will be mapped to ASCII hyphen-minus ("-")
HYPHEN_LIKE = {
    "\u2010",  # HYPHEN
    "\u2011",  # NON-BREAKING HYPHEN
    "\u2012",  # FIGURE DASH
    "\u2013",  # EN DASH
    "\u2014",  # EM DASH
    "\u2015",  # HORIZONTAL BAR
    "\u2212",  # MINUS SIGN
    "\ufe58",  # SMALL EM DASH
    "\ufe63",  # SMALL HYPHEN-MINUS
    "\uff0d",  # FULLWIDTH HYPHEN-MINUS
}
SOFT_HYPHEN = "\u00ad"  # SOFT HYPHEN  (will be removed outright)


def normalize_hyphens(text: str) -> str:
    text = text.replace(SOFT_HYPHEN, "")
    return "".join("-" if ch in HYPHEN_LIKE else ch for ch in text)


# All will be mapped to ASCII space (" ")
SPACE_LIKE = {
    "\u00a0",  # NO-BREAK SPACE
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",  # various spaces
    "\u202f",  # NARROW NO-BREAK SPACE
    "\u205f",  # MEDIUM MATHEMATICAL SPACE
    "\u3000",  # IDEOGRAPHIC SPACE
}


def normalize_spaces(text: str) -> str:
    return "".join(" " if ch in SPACE_LIKE else ch for ch in text)


# All will be completely removed
ZERO_WIDTH = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
}


def remove_zero_width(text: str) -> str:
    return "".join(ch for ch in text if ch not in ZERO_WIDTH)


QUOTE_MAP = {
    ord("\u2018"): "'",  # LEFT SINGLE QUOTATION MARK
    ord("\u2019"): "'",  # RIGHT SINGLE QUOTATION MARK
    ord("\u201a"): "'",  # SINGLE LOW-9 QUOTATION MARK
    ord("\u201b"): "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    ord("`"): "'",
    ord("\u201c"): '"',  # LEFT DOUBLE QUOTATION MARK
    ord("\u201d"): '"',  # RIGHT DOUBLE QUOTATION MARK
    ord("\u201e"): '"',  # DOUBLE LOW-9 QUOTATION MARK
    ord("\u201f"): '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
}


def normalize_quotes(text: str) -> str:
    return text.translate(QUOTE_MAP)


def normalize_text(raw: RawText) -> NormText:
    """
    Transform raw text into normalized form.
    """
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", raw)
    # 2. Remove zero-width characters
    text = remove_zero_width(text)
    # 3. Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 4. Normalize hyphens/dashes
    text = normalize_hyphens(text)
    # 5. Normalize spaces
    text = normalize_spaces(text)
    # 6. Normalize quotes
    text = normalize_quotes(text)
    # 7. Normalize whitespace: tabs/newlines -> space
    text = text.replace("\t", " ").replace("\n", " ")
    # 8. Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    # 9. Strip leading/trailing spaces
    text = text.strip()

    return NormText(text)


def to_char_tokens(text: NormText) -> CharTokens:
    """
    Convert text to space-separated character tokens (e.g. for KenLM).

    Spaces are represented as the special token "<sp>".
    """
    ret = " ".join("<sp>" if ch == " " else ch for ch in text)
    return cast(CharTokens, ret)
