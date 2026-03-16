"""
Tests for library/denoise/dehyphenate.py
"""

import pytest

from library.denoise.dehyphenate import (
    char_join_hyphen_break,
    dehyphenate_book,
    dehyphenate_middlematter,
    dehyphenate_page,
    ends_with_hyphen,
    has_hyphenated_lines,
)
from library.denoise.ngrams import NGramScorer, build_ngram_stats


class TestEndsWithHyphen:
    """Tests for ends_with_hyphen function."""

    def test_basic_hyphen(self):
        """Test detection of basic ASCII hyphen."""
        assert ends_with_hyphen("word-") is True
        assert ends_with_hyphen("word") is False

    def test_hyphen_with_trailing_spaces(self):
        """Test that trailing spaces are handled."""
        assert ends_with_hyphen("word-  ") is True
        assert ends_with_hyphen("word  ") is False

    def test_empty_string(self):
        """Test empty string returns False."""
        assert ends_with_hyphen("") is False

    def test_whitespace_only(self):
        """Test whitespace-only string returns False."""
        assert ends_with_hyphen("   ") is False


class TestHasHyphenatedLines:
    """Tests for has_hyphenated_lines function."""

    def test_no_hyphens(self):
        """Test pages without hyphenated lines."""
        pages = ["Line one", "Line two", "Line three"]
        assert has_hyphenated_lines(pages) is False

    def test_with_hyphen(self):
        """Test pages with hyphenated lines."""
        pages = ["Line one-", "continuation", "Line three"]
        assert has_hyphenated_lines(pages) is True

    def test_hyphen_in_middle(self):
        """Test that mid-line hyphens don't count."""
        pages = ["Line with-hyphen in middle", "Another line"]
        assert has_hyphenated_lines(pages) is False

    def test_empty_pages(self):
        """Test empty pages list."""
        assert has_hyphenated_lines([]) is False

    def test_multiline_page(self):
        """Test page with multiple lines."""
        pages = ["Line one\nLine two-\nLine three"]
        assert has_hyphenated_lines(pages) is True


class TestCharJoinHyphenBreak:
    """Tests for char_join_hyphen_break function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        result = char_join_hyphen_break("test-", "ing", scorer, scorer)
        assert isinstance(result, str)

    def test_no_hyphen_returns_newline(self):
        """Test that line without hyphen returns newline."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        result = char_join_hyphen_break("test", "ing", scorer, scorer)
        assert result == "\n"

    def test_empty_next_line_returns_newline(self):
        """Test that empty next line returns newline."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        result = char_join_hyphen_break("test-", "", scorer, scorer)
        assert result == "\n"


class TestDehyphenatePage:
    """Tests for dehyphenate_page function."""

    def test_no_hyphens_unchanged(self):
        """Test that page without hyphens is unchanged."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        page = "Line one\nLine two\nLine three"
        result = dehyphenate_page(page, scorer, scorer)
        assert result == page

    def test_empty_page(self):
        """Test empty page is unchanged."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        result = dehyphenate_page("", scorer, scorer)
        assert result == ""


class TestDehyphenateMiddlematter:
    """Tests for dehyphenate_middlematter function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        pages = ["Page one", "Page two"]
        result = dehyphenate_middlematter(pages, scorer)
        assert isinstance(result, list)
        assert len(result) == len(pages)

    def test_empty_pages(self):
        """Test with empty pages list."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        result = dehyphenate_middlematter([], scorer)
        assert result == []

    def test_preserves_unhyphenated_content(self):
        """Test that unhyphenated content is preserved."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        pages = ["Normal text here", "More normal text"]
        result = dehyphenate_middlematter(pages, scorer)
        assert result[0] == pages[0]
        assert result[1] == pages[1]


class TestDehyphenateBook:
    """Tests for dehyphenate_book function."""

    def test_missing_middlematter_raises(self):
        """Test that missing middlematter raises ValueError."""
        book = {"language_gen": "en"}

        with pytest.raises(ValueError, match="No middlematter found"):
            dehyphenate_book(book)

    def test_empty_middlematter_raises(self):
        """Test that empty middlematter raises ValueError."""
        book = {"middlematter": [], "language_gen": "en"}

        with pytest.raises(ValueError, match="No middlematter found"):
            dehyphenate_book(book)

    def test_no_hyphens_unchanged(self):
        """Test that book without hyphens is returned unchanged."""
        book = {"middlematter": ["normal text"], "language_gen": "en", "barcode_src": "123"}

        # No hyphens means no model loading needed
        result = dehyphenate_book(book)
        assert result is book
        assert result["middlematter"] == ["normal text"]

    def test_missing_language_raises(self):
        """Test that missing language raises ValueError when hyphens present."""
        book = {"middlematter": ["text with-\nhyphen"], "barcode_src": "123"}

        with pytest.raises(ValueError, match="No 'language_gen'"):
            dehyphenate_book(book)

    def test_preserves_other_fields(self):
        """Test that other book fields are preserved."""
        book = {
            "middlematter": ["normal text"],
            "language_gen": "en",
            "barcode_src": "123",
            "title": "Test Book",
        }

        result = dehyphenate_book(book)
        assert result["barcode_src"] == "123"
        assert result["title"] == "Test Book"
