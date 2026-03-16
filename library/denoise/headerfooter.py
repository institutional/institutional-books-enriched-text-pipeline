"""
headerfooter.py - detect and remove running headers and footers

By default, this considers the top `header_size=5` and bottom `footer_size=5`
lines on each page and detects high similarity pairs within small clusters of
pages. The default removes duplicate lines with at least 3 instances within a
span of 5 pages.

If OCR incorrectly identifies text appearing at the top or bottom of the page,
or if repeated content occurs somewhere else, then it won't be removed.
"""

from itertools import islice

from datasketch import MinHash, MinHashLSH
from loguru import logger

from const.config import PipelineConfig
from const.types import BookJSON, NormPage, NormText, PageIdx

# type-based documentation
type Line = NormText
type LineIdx = int
type PageLines = list[Line]
type PageLocation = tuple[PageIdx, LineIdx]


def remove_headers_footers_book(
    book: BookJSON,
    config: PipelineConfig | None = None,
    header_size: int = 5,
    footer_size: int = 5,
    sim_threshold: float = 0.85,
) -> BookJSON:
    """
    Remove headers and footers from book['middlematter'].

    Checks the first `header_size` and last `footer_size` lines for duplicate
    entries, where duplicates must be at least `sim_threshold` similar.
    """
    middlematter = book.get("middlematter", [])
    if not middlematter:
        book_id = book.get("barcode_src", "UNKNOWN")
        logger.warning(f"{book_id} has no middlematter.")
        raise ValueError(f"{book_id} has no middlematter.")

    cleaned_pages, _ = detect_and_clean_headers_footers(
        middlematter,
        header_size=header_size,
        footer_size=footer_size,
        sim_threshold=sim_threshold,
    )

    cleaned_middlematter = ["\n".join(page) for page in cleaned_pages]

    result = book
    result["middlematter"] = cleaned_middlematter
    return result


def detect_and_clean_headers_footers(
    pages: list[NormPage],
    header_size: int = 5,
    footer_size: int = 5,
    sim_threshold: float = 0.85,
) -> tuple[list[PageLines], set[PageLocation]]:
    """
    Detect and clean headers and footers.

    Returns cleaned pages and the PageLocations of removed lines.
    """
    pages_by_lines: list[PageLines]
    pages_by_lines = [page.splitlines() for page in pages]  # type: ignore
    locations_to_remove = detect_hf_duplicates(
        pages_by_lines,
        header_size=header_size,
        footer_size=footer_size,
        sim_threshold=sim_threshold,
    )
    cleaned_pages = remove_hf_lines(pages_by_lines, locations_to_remove)
    return cleaned_pages, locations_to_remove


def remove_hf_lines(
    pages_by_lines: list[PageLines],
    lines_to_remove: set[PageLocation],
) -> list[PageLines]:
    """Remove specified lines from pages."""
    new_pages_by_lines: list[PageLines] = []
    for page_idx, page_by_lines in enumerate(pages_by_lines):
        new_page_lines: PageLines = []
        for line_idx, line in enumerate(page_by_lines):
            if (page_idx, line_idx) not in lines_to_remove:
                new_page_lines.append(line)
        new_pages_by_lines.append(new_page_lines)
    return new_pages_by_lines


def detect_hf_duplicates(
    pages_by_lines: list[PageLines],
    header_size: int = 5,
    footer_size: int = 5,
    sim_threshold: float = 0.9,
) -> set[PageLocation]:
    """
    Detect header/footer candidates based on n-gram multiset similarity.

    Returns a set of (page_idx, line_idx) tuples indicating lines to be removed.
    """
    # Initialize minhash setup for duplicate detection in book in memory
    groups: set[tuple[PageLocation, ...]] = set()
    header_lsh = MinHashLSH(threshold=sim_threshold)
    header_hashes: list[MinHash] = []
    footer_lsh = MinHashLSH(threshold=sim_threshold)
    footer_hashes: list[MinHash] = []

    # For every page and every line in the first header_size and last
    # footer_size, compute the minhashes and add to LSH.
    for pidx, page_by_lines in enumerate(pages_by_lines):
        for lidx, line in enumerate(page_by_lines):
            # Also remove spaces to not deal with odd OCR
            line = "".join(ch for ch in line if not ch.isspace())
            # Lines that are too trivial are never removed
            if len(line) < 6:
                continue
            key: PageLocation
            if lidx < header_size:
                mh = minhash_from_tokens(line)
                key = (pidx, lidx)
                header_lsh.insert(key, mh)
                header_hashes.append(mh)
            elif len(page_by_lines) - lidx <= footer_size:
                mh = minhash_from_tokens(line)
                key = (pidx, lidx)
                footer_lsh.insert(key, mh)
                footer_hashes.append(mh)

    # Use MinhashLSH to identify groups naively, checking every hash against
    # every hash.
    for mh in header_hashes:
        if len(header_lsh.query(mh)) >= 3:  # type: ignore
            groups.add(tuple(sorted(header_lsh.query(mh))))  # type: ignore
    for mh in footer_hashes:
        if len(footer_lsh.query(mh)) >= 3:  # type: ignore
            groups.add(tuple(sorted(footer_lsh.query(mh))))  # type: ignore

    lines_to_remove: set[PageLocation] = set()
    for group in groups:
        if not clustered_group(group):
            continue
        for key in group:
            lines_to_remove.add(key)
    return lines_to_remove


def clustered_group(group: tuple[PageLocation, ...]) -> bool:
    """
    Check if a group of PageLocations is a localized header/footer group.

    A valid group has at least 3 instances from a span of 5 pages.
    """
    if len(group) < 3:
        return False
    page_indices = {pidx for pidx, _ in group}
    sorted_pages = sorted(page_indices)
    # this is so leetcode-coded
    for i in range(len(sorted_pages) - 2):
        if sorted_pages[i + 2] - sorted_pages[i] <= 4:
            return True
    return False


def ngrams(tokens: str, n: int = 5):
    """Generate character n-grams from a string."""
    iters = [islice(tokens, i, None) for i in range(n)]
    return zip(*iters)


def minhash_from_tokens(tokens: str, n: int = 5, num_perm: int = 128) -> MinHash:
    """Create a MinHash from character n-grams."""
    m = MinHash(num_perm=num_perm)
    for ng in ngrams(tokens, n):
        gram_str = " ".join(ng)
        m.update(gram_str.encode("utf-8"))  # type: ignore
    return m
