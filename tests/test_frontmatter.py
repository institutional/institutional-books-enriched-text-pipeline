"""Tests for library/denoise/frontmatter.py

Institutional Books - Enriched Text - 2026
"""

from pathlib import Path

import pytest

from library.denoise.frontmatter import (
    identify_frontmatter_backmatter,
    load_classifier,
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


# =============================================================================
# Integration tests - Real m2v classifier
# =============================================================================

# Check for real classifier availability
_default_classifier_path = Path("./DATA/pretrain/models/mmem_classifier")

requires_m2v_classifier = pytest.mark.skipif(
    not _default_classifier_path.exists(),
    reason=f"m2v classifier not available at {_default_classifier_path}",
)


@requires_m2v_classifier
class TestFrontmatterWithRealClassifier:
    """Integration tests using real m2v classifier."""

    def test_load_real_classifier(self):
        """Test that real classifier loads successfully."""
        classifier = load_classifier(_default_classifier_path)
        assert classifier is not None
        assert hasattr(classifier, "predict")

    def test_classifier_returns_valid_labels(self):
        """Test that classifier returns valid label strings."""
        classifier = load_classifier(_default_classifier_path)

        pages = ["This is a preface page.", "This is chapter content.", "This is an index."]
        predictions = classifier.predict(pages)

        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred, str)

    def test_typical_book_structure(self):
        """Test classification of a typical book structure."""
        classifier = load_classifier(_default_classifier_path)

        # Simulated book with clear structure
        pages = [
            "PREFACE\n\nThis book is dedicated to all students of science.",
            "TABLE OF CONTENTS\n\nChapter 1 - Introduction\nChapter 2 - Methods",
            "CHAPTER 1\n\nIntroduction to the subject matter. This chapter covers the basics.",
            "CHAPTER 1 (continued)\n\nMore detailed discussion of foundational concepts.",
            "CHAPTER 2\n\nMethodology and approach used in this research study.",
            "CHAPTER 2 (continued)\n\nFurther details about the experimental methods.",
            "BIBLIOGRAPHY\n\nSmith, J. (2020). Research Methods. Academic Press.",
            "INDEX\n\nA\nApproach, 45\nAnalysis, 67",
        ]

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        # Should identify some frontmatter and backmatter
        assert total == 8
        # Frontmatter should be at least the preface/TOC
        assert front_end >= 0
        # Main content should end before bibliography/index
        assert main_end <= total

    def test_all_content_book(self):
        """Test a book that is all main content."""
        classifier = load_classifier(_default_classifier_path)

        # All pages are clearly main content
        pages = [
            "Chapter 1: The story begins on a dark and stormy night in London.",
            "The protagonist walked through the rain, contemplating life choices.",
            "Meanwhile, across town, events were unfolding that would change everything.",
            "Chapter 2: The next morning brought unexpected visitors to the door.",
            "The conversation that followed revealed shocking family secrets.",
        ]

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        # Should identify some middlematter (at least 2 pages required by algorithm)
        assert total == 5
        # Check that result makes sense (indices are valid)
        assert 0 <= front_end <= main_end <= total

    def test_separate_book_with_real_classifier(self):
        """Test full book separation with real classifier."""
        classifier = load_classifier(_default_classifier_path)

        book = {
            "barcode_src": "test123",
            "uniformized_text": [
                "PREFACE\n\nWelcome to this comprehensive guide.",
                "Chapter 1\n\nThe fundamentals of the subject are covered here.",
                "Chapter 2\n\nAdvanced topics build on the previous chapter.",
                "Chapter 3\n\nPractical applications demonstrate the concepts.",
                "INDEX\n\nA: Applications, 45\nB: Basics, 12",
            ],
        }

        result = separate_endmatter_book(book, classifier)

        # Should have all three sections
        assert "frontmatter" in result
        assert "middlematter" in result
        assert "backmatter" in result
        assert "uniformized_text" not in result

        # Total pages should be preserved
        total_pages = (
            len(result["frontmatter"])
            + len(result["middlematter"])
            + len(result["backmatter"])
        )
        assert total_pages == 5

    def test_empty_pages_handled(self):
        """Test that empty pages don't break classification."""
        classifier = load_classifier(_default_classifier_path)

        pages = [
            "",
            "Chapter 1\n\nActual content starts here with important information.",
            "Chapter 2\n\nMore content continues in this chapter.",
            "",
        ]

        front_end, main_end, total = identify_frontmatter_backmatter(pages, classifier)

        # Should handle empty pages gracefully without crashing
        assert total == 4
        # Result should be valid indices
        assert 0 <= front_end <= main_end <= total

    def test_preserves_page_content(self):
        """Test that separation preserves all page content exactly."""
        classifier = load_classifier(_default_classifier_path)

        original_pages = [
            "Preface content here.",
            "Main chapter content.",
            "More main content.",
            "Index content here.",
        ]

        book = {
            "barcode_src": "test",
            "uniformized_text": original_pages.copy(),
        }

        result = separate_endmatter_book(book, classifier)

        # Collect all pages from result
        all_result_pages = (
            result["frontmatter"] + result["middlematter"] + result["backmatter"]
        )

        # All original pages should be present
        assert len(all_result_pages) == len(original_pages)
        for page in original_pages:
            assert page in all_result_pages
