"""
stray_numbers.py

The logic is similar to page number removal, except that this is expected to be
used after sentence segmentation. This looks for sentences consisting almost
entirely of numeric characters and removes them.

Institutional Books - Enriched Text - 2026
"""

from const.config import PipelineConfig
from const.types import BookJSON
from utils.text import numeric_count


def remove_stray_numbers_book(
    book: BookJSON,
    config: PipelineConfig,
    threshold: float = 0.9,
    min_length: int = 1,
) -> BookJSON:
    """
    Remove stray number sentences from a book's middlematter_sentences.

    Args:
        book: Book dictionary with 'middlematter_sentences' field
        config: Pipeline configuration (unused for this step)
        threshold: Fraction of numeric chars to trigger removal (default: 0.9)
        min_length: Minimum sentence length to consider (default: 1)
    """
    sentences = book.get("middlematter_sentences", [])
    if not sentences:
        book_id = book.get("barcode_src", "UNKNOWN")
        raise ValueError(f"No 'sentences' found in book {book_id}")

    cleaned_sentences, num_removed = detect_and_remove_stray_number_fragments(
        sentences,
        threshold=threshold,
        min_length=min_length,
    )

    result = book
    result["middlematter_sentences"] = cleaned_sentences
    if num_removed > 0:
        result["_stray_numbers_removed"] = num_removed
    return result


def is_probable_stray_number_fragment(
    fragment: str,
    threshold: float = 0.9,
    min_length: int = 2,
    max_length: int = 10,
) -> bool:
    """
    Returns True if the given fragment is likely to be a stray number fragment.

    The fragment is expected to be a whole sentence.

    Heuristic: if the fraction of numeric characters is >= threshold,
    then it's probably a stray number fragment. Alternately, if there is only
    one nonnumeric character, it's probably a stray number fragment.
    """
    if len(fragment) < min_length:
        return False
    if len(fragment) > max_length:
        return False
    numeric, nonnumeric = numeric_count(fragment)
    total = numeric + nonnumeric
    if total == 0:
        return False
    if nonnumeric <= 1:
        return True
    return (numeric / total) >= threshold


def detect_and_remove_stray_number_fragments(
    sentence_list: list[str],
    threshold: float = 0.9,
    min_length: int = 2,
) -> tuple[list[str], int]:
    """
    Detect and remove stray number fragments from the sentence list.

    Returns:
        A tuple (cleaned_sentences, num_removed).
    """
    cleaned_sentences = [
        sentence
        for sentence in sentence_list
        if not is_probable_stray_number_fragment(sentence, threshold, min_length)
    ]
    return cleaned_sentences, len(sentence_list) - len(cleaned_sentences)
