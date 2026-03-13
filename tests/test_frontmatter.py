"""Tests for library/denoise/frontmatter.py"""

import pytest

from library.denoise.frontmatter import (
    identify_frontmatter_backmatter,
    separate_endmatter_book,
)


class MockClassifier:
    """Mock classifier returning predefined labels."""

    def __init__(self, labels: list[str]):
        self.labels = labels

    def predict(self, texts: list[str]) -> list[str]:
        return self.labels[: len(texts)]


class TestIdentifyFrontmatterBackmatter:
    def test_typical_book_structure(self):
        """Test a book with frontmatter, middlematter, and backmatter."""
        pages = ["preface", "ch1", "ch2", "ch3", "index"]
        labels = ["ENDMATTER", "MIDDLEMATTER", "MIDDLEMATTER", "MIDDLEMATTER", "ENDMATTER"]
        classifier = MockClassifier(labels)

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        assert front_end == 1
        assert main_end == 4
        assert total == 5

    def test_no_frontmatter(self):
        """Test a book with no frontmatter."""
        pages = ["ch1", "ch2", "ch3", "index"]
        labels = ["MIDDLEMATTER", "MIDDLEMATTER", "MIDDLEMATTER", "ENDMATTER"]
        classifier = MockClassifier(labels)

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        assert front_end == 0  # no frontmatter
        assert main_end == 3
        assert total == 4

    def test_no_backmatter(self):
        """Test a book with no backmatter."""
        pages = ["preface", "ch1", "ch2", "ch3"]
        labels = ["ENDMATTER", "MIDDLEMATTER", "MIDDLEMATTER", "MIDDLEMATTER"]
        classifier = MockClassifier(labels)

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        assert front_end == 1
        assert main_end == 4  # middlematter goes to end
        assert total == 4

    def test_all_middlematter(self):
        """Test a book with only middlematter."""
        pages = ["ch1", "ch2", "ch3"]
        labels = ["MIDDLEMATTER", "MIDDLEMATTER", "MIDDLEMATTER"]
        classifier = MockClassifier(labels)

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        assert front_end == 0
        assert main_end == 3
        assert total == 3

    def test_insufficient_middlematter_returns_all_content(self):
        """Test that books with <2 middlematter pages return all as main content."""
        pages = ["preface", "single_chapter", "index"]
        labels = ["ENDMATTER", "MIDDLEMATTER", "ENDMATTER"]
        classifier = MockClassifier(labels)

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        # With only 1 middlematter page, we can't reliably identify boundaries
        assert front_end == 0
        assert main_end == total
        assert total == 3

    def test_empty_pages_treated_as_endmatter(self):
        """Test that empty pages classified as MIDDLEMATTER become ENDMATTER."""
        pages = ["", "ch1", "ch2", ""]
        labels = ["MIDDLEMATTER", "MIDDLEMATTER", "MIDDLEMATTER", "MIDDLEMATTER"]
        classifier = MockClassifier(labels)

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        # Empty first page should be treated as frontmatter
        assert front_end == 1
        # Empty last page should be treated as backmatter
        assert main_end == 3


class TestSeparateEndmatterBook:
    def test_separates_book_correctly(self):
        """Test that a book is correctly separated into three parts."""
        book = {
            "barcode_src": "test123",
            "uniformized_text": ["preface", "ch1", "ch2", "index"],
        }
        labels = ["ENDMATTER", "MIDDLEMATTER", "MIDDLEMATTER", "ENDMATTER"]
        classifier = MockClassifier(labels)

        result = separate_endmatter_book(book, classifier)

        assert result["frontmatter"] == ["preface"]
        assert result["middlematter"] == ["ch1", "ch2"]
        assert result["backmatter"] == ["index"]
        assert "uniformized_text" not in result

    def test_preserves_other_fields(self):
        """Test that other book fields are preserved."""
        book = {
            "barcode_src": "test123",
            "language_gen": "en",
            "uniformized_text": ["ch1", "ch2"],
        }
        labels = ["MIDDLEMATTER", "MIDDLEMATTER"]
        classifier = MockClassifier(labels)

        result = separate_endmatter_book(book, classifier)

        assert result["barcode_src"] == "test123"
        assert result["language_gen"] == "en"

    def test_raises_on_missing_uniformized_text(self):
        """Test that missing uniformized_text raises ValueError."""
        book = {"barcode_src": "test123"}
        classifier = MockClassifier([])

        with pytest.raises(ValueError, match="No 'uniformized_text' found"):
            separate_endmatter_book(book, classifier)

    def test_raises_on_empty_uniformized_text(self):
        """Test that empty uniformized_text raises ValueError."""
        book = {"barcode_src": "test123", "uniformized_text": []}
        classifier = MockClassifier([])

        with pytest.raises(ValueError, match="No 'uniformized_text' found"):
            separate_endmatter_book(book, classifier)
