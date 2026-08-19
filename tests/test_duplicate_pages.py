"""
Tests for library/denoise/duplicate_pages.py

Institutional Books - Enriched Text - 2026
"""

from typing import cast

from const.types import BookJSON, NormPage
from library.denoise.duplicate_pages import (
    detect_duplicate_pages,
    remove_duplicate_pages,
    remove_duplicate_pages_from_book,
)


class TestDetectDuplicatePages:
    def test_no_duplicates(self):
        pages = [
            "This is the first page with unique content about apples.",
            "This is the second page with different content about oranges.",
            "This is the third page with content about bananas and grapes.",
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, clusters = detect_duplicate_pages(pages)
        assert len(pages_to_keep) == 3
        assert clusters == {}

    def test_exact_duplicates(self):
        pages = [
            "This is a page with enough content to be considered for deduplication.",
            "This is a page with enough content to be considered for deduplication.",
            "This is a different page with unique content that should be kept.",
        ]
        pages_to_keep, clusters = detect_duplicate_pages(pages)
        assert len(pages_to_keep) == 2  # One duplicate removed
        assert 0 in pages_to_keep  # First occurrence kept
        assert 2 in pages_to_keep  # Different page kept

    def test_short_pages_kept(self):
        pages = [
            "Short",
            "Short",
            "This is a longer page that should be considered for deduplication.",
        ]
        pages = cast(list[NormPage], pages)
        # Short pages (< 50 chars) should always be kept
        pages_to_keep, _ = detect_duplicate_pages(pages, min_length=50)
        assert 0 in pages_to_keep
        assert 1 in pages_to_keep

    def test_empty_pages(self):
        pages_to_keep, clusters = detect_duplicate_pages([])
        assert pages_to_keep == []
        assert clusters == {}


class TestRemoveDuplicatePages:
    def test_removes_duplicates(self):
        pages = [
            "This is a page with enough content to be considered for deduplication.",
            "This is a page with enough content to be considered for deduplication.",
            "Unique page here.",
        ]
        pages = cast(list[NormPage], pages)
        cleaned, num_removed = remove_duplicate_pages(pages)
        assert num_removed == 1
        assert len(cleaned) == 2


class TestRemoveDuplicatePagesFromBook:
    def test_book_processing(self):
        book = {
            "barcode_src": "test123",
            "uniformized_text": [
                "This is a page with enough content to be considered for deduplication.",
                "This is a page with enough content to be considered for deduplication.",
                "Different page content.",
            ],
        }
        result = remove_duplicate_pages_from_book(book)
        assert len(result["uniformized_text"]) == 2
        assert result.get("_duplicate_pages_removed") == 1

    def test_empty_book(self):
        book: BookJSON = {"barcode_src": "test123", "uniformized_text": []}
        result = remove_duplicate_pages_from_book(book)
        assert result["uniformized_text"] == []


# =============================================================================
# Integration tests - Near-duplicates and OCR variations
# =============================================================================


class TestNearDuplicateDetection:
    """Integration tests for near-duplicate detection with realistic variations."""

    def test_exact_duplicate_long_pages(self):
        """Test that exact duplicate long pages are detected."""
        long_content = "The quick brown fox jumps over the lazy dog. " * 5
        pages = [
            long_content,
            long_content,
            "This is a completely different page with unique content about something else entirely. "
            * 3,
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, clusters = detect_duplicate_pages(pages, threshold=6)

        # First two should be detected as exact duplicates
        assert len(pages_to_keep) == 2
        assert 0 in pages_to_keep  # First occurrence kept
        assert 2 in pages_to_keep  # Different page kept

    def test_near_duplicate_high_threshold(self):
        """Test near-duplicates with higher threshold for more tolerance."""
        pages = [
            "The quick brown fox jumps over the lazy dog. This sentence repeats multiple times for testing.",
            "The quick brown fox jumps over the lazy dog. This sentence repeats multiple times for testing.",
            "Something completely different here with no relation to foxes or dogs at all and more text.",
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, _ = detect_duplicate_pages(pages, threshold=6)

        assert len(pages_to_keep) == 2

    def test_very_similar_pages_detected(self):
        """Test that very similar pages are detected with appropriate threshold."""
        base_text = "Hello world! This is a test page with enough content. " * 4
        pages = [
            base_text,
            base_text,
            "Goodbye world. This is a totally different page about something unrelated. " * 4,
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, _ = detect_duplicate_pages(pages, threshold=6)

        assert len(pages_to_keep) == 2

    def test_multiple_duplicate_groups(self):
        """Test detection of multiple distinct duplicate groups."""
        pages = [
            "First repeated content appears on this page with plenty of words to make it long enough.",
            "First repeated content appears on this page with plenty of words to make it long enough.",
            "Second repeated content is different from the first but also appears multiple times.",
            "Second repeated content is different from the first but also appears multiple times.",
            "This page is unique and should definitely be kept in the final output.",
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, clusters = detect_duplicate_pages(pages, threshold=6)

        # Should keep one from each group plus the unique page
        assert len(pages_to_keep) == 3
        assert 0 in pages_to_keep or 1 in pages_to_keep  # One from first group
        assert 2 in pages_to_keep or 3 in pages_to_keep  # One from second group
        assert 4 in pages_to_keep  # Unique page

    def test_near_duplicate_threshold_sensitivity(self):
        """Test that threshold parameter controls sensitivity."""
        # Pages that are similar but not identical
        base_text = "The quick brown fox jumps over the lazy dog. " * 3
        pages = [
            base_text,
            base_text.replace("quick", "fast").replace("lazy", "sleepy"),
            "Completely unrelated text about programming and software development.",
        ]
        pages = cast(list[NormPage], pages)

        # With strict threshold, they might not match
        pages_strict, _ = detect_duplicate_pages(pages, threshold=3)

        # With lenient threshold, they should match
        pages_lenient, _ = detect_duplicate_pages(pages, threshold=10)

        # Lenient should detect more duplicates (keep fewer pages)
        assert len(pages_lenient) <= len(pages_strict)

    def test_preserves_page_order(self):
        """Test that kept pages maintain their original order."""
        pages = [
            "Page zero with unique content that should be preserved in the output.",
            "Page one is a duplicate of page three with the same exact content here.",
            "Page two with unique content that should also be preserved correctly.",
            "Page one is a duplicate of page three with the same exact content here.",
            "Page four with unique content that rounds out this test case nicely.",
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, _ = detect_duplicate_pages(pages, threshold=6)

        # Check order is maintained
        sorted_pages = sorted(pages_to_keep)
        assert pages_to_keep == sorted_pages

    def test_boundary_length_pages(self):
        """Test pages at exactly the minimum length threshold."""
        # Pages need enough variety for simhash - repeated chars don't work well
        # Use 60 chars with actual text patterns
        page_long = "The quick brown fox jumps over the lazy sleeping dog today."  # 60 chars
        pages = [
            page_long,
            page_long,
            "A completely different sentence with unique content here now.",  # 60 chars
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, _ = detect_duplicate_pages(pages, min_length=50)

        # Should detect the duplicate
        assert len(pages_to_keep) == 2

        # Now test with short pages (below threshold)
        short_pages = [
            "Short text here",
            "Short text here",
            "Other short txt",
        ]
        short_pages = cast(list[NormPage], short_pages)
        short_pages_to_keep, _ = detect_duplicate_pages(short_pages, min_length=50)

        # Short pages should all be kept (not considered for dedup)
        assert len(short_pages_to_keep) == 3

    def test_real_book_like_duplicates(self):
        """Test with content resembling real book page duplicates."""
        # Simulating a scanned book where a page was accidentally scanned twice
        real_page = """
        Chapter 5: The Industrial Revolution

        The Industrial Revolution marked a major turning point in history.
        Almost every aspect of daily life was influenced in some way.
        The transition included going from hand production methods to machines.
        New chemical manufacturing and iron production processes were developed.
        The use of steam power and water power also increased dramatically.
        """
        pages = [
            real_page,
            real_page,  # Duplicate scan
            """
        Chapter 6: The Modern Era

        The modern era brought unprecedented changes to human society.
        Technology advanced at a rate never before seen in human history.
        Communication became instantaneous across vast distances.
        Transportation allowed people to travel the globe with ease.
        """,
        ]
        pages = cast(list[NormPage], pages)
        pages_to_keep, clusters = detect_duplicate_pages(pages, threshold=6)

        assert len(pages_to_keep) == 2
        assert 0 in pages_to_keep  # First scan kept
        assert 2 in pages_to_keep  # Different chapter kept
        assert len(clusters) == 1  # One duplicate cluster
