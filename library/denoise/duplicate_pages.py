"""
duplicate_pages.py - remove duplicate pages within a book using simhash

Institutional Books - Enriched Text - 2026
"""

from typing import Any

from const.types import BookJSON, NormPage, PageIdx, SimhashHash
from library.denoise.uniformize import hard_normalize_unicode
from utils.simhash_fast import are_near_duplicates, simhash128
from utils.unionfind import UnionFind


def remove_duplicate_pages_from_book(
    book: BookJSON,
    config: dict[str, Any] | None = None,
    hamming_threshold: int = 6,
    ngram_size: int = 9,
    min_length: int = 50,
) -> BookJSON:
    """
    Remove duplicate pages from a book's uniformized_text.

    Uses simhash-based near-duplicate detection to identify and remove
    duplicate pages within the book. The first occurrence of each
    duplicate group is kept.

    Args:
        book: Book dictionary with 'uniformized_text' field
        config: Pipeline configuration (not used in current version)
        threshold: Maximum Hamming distance for near-duplicates (default: 6)
        ngram_size: N-gram size for simhash computation (default: 9)
        min_length: Minimum page length to consider for deduplication (default: 50)

    Returns:
        Book dictionary with duplicate pages removed from 'uniformized_text'
    """
    pages = book.get("uniformized_text", [])
    if not pages:
        return book
    cleaned_pages, num_removed = remove_duplicate_pages(
        pages,
        threshold=hamming_threshold,
        ngram_size=ngram_size,
        min_length=min_length,
    )
    result = dict(book)
    result["uniformized_text"] = cleaned_pages
    if num_removed > 0:
        result["_duplicate_pages_removed"] = num_removed
    return result


type PageIndicesToKeep = list[int]

# page_representative --> list of matching pages
type ClusterRecord = dict[int, list[int]]


def detect_duplicate_pages(
    pages: list[NormPage],
    threshold: int = 6,
    ngram_size: int = 9,
    min_length: int = 50,
) -> tuple[PageIndicesToKeep, ClusterRecord]:
    """
    Detect duplicate pages within a book using simhash.

    Args:
        pages: List of page strings
        threshold: Maximum Hamming distance for near-duplicates (default: 6)
        ngram_size: N-gram size for simhash computation (default: 9)
        min_length: Minimum page length to consider for deduplication.
            Shorter pages are always kept (default: 50)

    Returns:
        A tuple (pages_to_keep, clusters) where:
        - pages_to_keep: Sorted list of page indices to keep
        - clusters: Dict mapping representative page index to list of duplicate indices
    """
    # Compute simhashes for all pages
    page_hashes: list[tuple[PageIdx, SimhashHash]] = []
    short_pages: set[PageIdx] = set()

    for idx, page in enumerate(pages):
        # Never recognize short pages as duplicates
        if len(page.strip()) < min_length:
            short_pages.add(idx)
            continue
        normalized = hard_normalize_unicode(page)
        h = simhash128(normalized, ngram_size=ngram_size)
        page_hashes.append((idx, h))

    if len(page_hashes) < 2:
        # Not enough pages to compare
        return list(range(len(pages))), {}

    # Find duplicate pairs using pairwise comparison
    # For single-book deduplication, O(n^2) is acceptable since n is small
    uf = UnionFind()
    for i, (idx_i, hash_i) in enumerate(page_hashes):
        for j in range(i + 1, len(page_hashes)):
            idx_j, hash_j = page_hashes[j]
            if are_near_duplicates(hash_i, hash_j, threshold):
                uf.union(idx_i, idx_j)

    clusters = uf.get_clusters()

    # Determine pages to keep: all short pages + first occurrence in each cluster
    pages_to_keep: set[PageIdx] = set(short_pages)

    # Add non-clustered pages
    clustered_pages: set[PageIdx] = set()
    for members in clusters.values():
        clustered_pages.update(members)
    for idx, _ in page_hashes:
        if idx not in clustered_pages:
            pages_to_keep.add(idx)

    # Add representative (first) from each cluster
    for members in clusters.values():
        representative = min(members)
        pages_to_keep.add(representative)

    return sorted(pages_to_keep), clusters


def remove_duplicate_pages(
    pages: list[NormPage],
    threshold: int = 6,
    ngram_size: int = 9,
    min_length: int = 50,
) -> tuple[list[NormPage], int]:
    """
    Remove duplicate pages from a list of pages.

    Args:
        pages: List of page strings
        threshold: Maximum Hamming distance for near-duplicates (default: 6)
        ngram_size: N-gram size for simhash computation (default: 9)
        min_length: Minimum page length to consider for deduplication (default: 50)

    Returns:
        A tuple (cleaned_pages, num_removed) where:
        - cleaned_pages: List of pages with duplicates removed
        - num_removed: Number of pages removed
    """
    pages_to_keep, _ = detect_duplicate_pages(
        pages,
        threshold=threshold,
        ngram_size=ngram_size,
        min_length=min_length,
    )

    cleaned_pages = [pages[idx] for idx in pages_to_keep]
    num_removed = len(pages) - len(cleaned_pages)

    return cleaned_pages, num_removed
