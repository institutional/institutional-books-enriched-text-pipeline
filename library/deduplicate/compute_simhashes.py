"""
compute_simhashes.py
"""

from const.types import BookJSON, BookSimhashes
from library.chunk.utils import segments_from_starts
from utils.simhash_fast import simhash128_batch


def compute_simhashes_in_book(book: BookJSON, ngram_size: int = 9) -> BookSimhashes:
    """
    Compute simhashes for all subtopic pararagraph chunks in book.
    """
    book_id = book.get("barcode_src", "UNKNOWN")
    if book_id == "UNKNOWN":
        raise ValueError("Unknown book encountered")
    sentences = book.get("middlematter_sentences", [])
    if not sentences:
        raise ValueError(f"No sentences found in {book_id}.")
    para_starts = book.get("subtopic_paragraph_start_indices", [])
    if not para_starts:
        raise ValueError(f"No paragraph indices in {book_id}. Run chunking steps first.")

    paras: list[str]
    paras = [" ".join(segments) for segments in segments_from_starts(sentences, para_starts)]
    hashes = simhash128_batch(paras, ngram_size=ngram_size)
    return BookSimhashes(book_id=book_id, simhashes=hashes)
