"""Tests for library/denoise/duplicate_pages.py"""

from library.denoise.duplicate_pages import (
    remove_duplicate_pages_from_book,
    detect_duplicate_pages,
    remove_duplicate_pages,
)


class TestDetectDuplicatePages:
    def test_no_duplicates(self):
        pages = [
            "This is the first page with unique content about apples.",
            "This is the second page with different content about oranges.",
            "This is the third page with content about bananas and grapes.",
        ]
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
        book = {"barcode_src": "test123", "uniformized_text": []}
        result = remove_duplicate_pages_from_book(book)
        assert result["uniformized_text"] == []
