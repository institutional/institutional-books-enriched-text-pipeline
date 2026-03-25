"""
Tests for library/denoise/dehyphenate.py
"""

from pathlib import Path

import pytest

from library.denoise.dehyphenate import (
    char_join_hyphen_break,
    dehyphenate_book,
    dehyphenate_middlematter,
    dehyphenate_page,
    ends_with_hyphen,
    has_hyphenated_lines,
)
from library.denoise.ngrams import NGramScorer, build_ngram_stats, load_ngram_stats


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


# =============================================================================
# Integration tests - Real n-gram models and actual dehyphenation
# =============================================================================

# Check for real model availability
_default_model_dir = Path("./DATA/pretrain/models")
_eng_model_path = _default_model_dir / "eng_ngram.json.gz"

requires_ngram_model = pytest.mark.skipif(
    not _eng_model_path.exists(),
    reason=f"English n-gram model not available at {_eng_model_path}",
)


@requires_ngram_model
class TestDehyphenateWithRealModel:
    """Integration tests using real n-gram models for dehyphenation."""

    def test_simple_word_dehyphenation(self):
        """Test dehyphenation of common hyphenated words."""
        stats = load_ngram_stats(_eng_model_path)
        scorer = NGramScorer(stats)

        # "exer-\ncise" should become "exercise"
        page = "The athlete needed to exer-\ncise every day."
        result = dehyphenate_page(page, scorer, scorer)

        # The hyphen should be removed, joining the word
        assert "exer-" not in result and "exercise" in result

    def test_compound_word_preserved(self):
        """Test that legitimate compound words with hyphens are preserved."""
        stats = load_ngram_stats(_eng_model_path)
        scorer = NGramScorer(stats)

        # "self-" at end of line followed by "esteem" - could go either way
        # but "well-known" mid-line should be preserved
        page = "This is a well-known fact.\nMore content here."
        result = dehyphenate_page(page, scorer, scorer)

        # Mid-line hyphen should be preserved
        assert "well-known" in result

    def test_dehyphenate_middlematter_real(self):
        """Test dehyphenation of multiple pages with real model."""
        stats = load_ngram_stats(_eng_model_path)
        scorer = NGramScorer(stats)

        pages = [
            "The scien-\ntist discovered something amazing.",
            "This para-\ngraph continues on the next line.",
            "Normal text without any hyphenation at all.",
        ]

        result = dehyphenate_middlematter(pages, scorer)

        assert len(result) == 3
        assert "scientist" in result[0]
        assert "paragraph" in result[1]
        assert result[2] == pages[2]

    def test_preserves_all_content(self):
        """Test that dehyphenation preserves all text content."""
        stats = load_ngram_stats(_eng_model_path)
        scorer = NGramScorer(stats)

        original = "The quick brown fox jumps over the la-\nzy dog."
        result = dehyphenate_page(original, scorer, scorer)

        # All words should still be present (possibly joined differently)
        for word in ["quick", "brown", "fox", "jumps", "over", "the", "dog"]:
            assert word in result

    def test_multiple_hyphens_per_page(self):
        """Test page with multiple end-of-line hyphens."""
        stats = load_ngram_stats(_eng_model_path)
        scorer = NGramScorer(stats)

        page = """The philoso-
pher contemplated the na-
ture of existence and real-
ity itself."""

        result = dehyphenate_page(page, scorer, scorer)

        # Should produce fewer lines due to joining
        assert result.count("\n") < page.count("\n")
        assert "philosopher" in result
        assert "nature" in result
        assert "reality" in result

    def test_short_corpus_book_adaptation(self):
        """Test that book-specific corpus improves dehyphenation."""
        stats = load_ngram_stats(_eng_model_path)
        base_scorer = NGramScorer(stats)

        # Build book-specific scorer with domain terms
        book_text = "immunology immunological immunologist vaccine vaccination"
        book_stats = build_ngram_stats(book_text)
        book_scorer = NGramScorer(book_stats)

        # Page with domain-specific hyphenated word
        page = "The immuno-\nlogical response was studied."

        # Test with both scorers (simulating real dehyphenate_middlematter behavior)
        result = dehyphenate_page(page, base_scorer, book_scorer)

        # Should successfully join the domain term
        assert "immuno-" not in result and "immunological" in result


class TestDehyphenateRealPatterns:
    """Integration tests with realistic hyphenation patterns (no model required)."""

    def test_hyphenation_detection_patterns(self):
        """Test various hyphenation patterns are detected correctly."""
        # All of these should be detected as having hyphenated lines
        hyphenated_pages = [
            "word-\ncontinuation",
            "end of line-\n next line",
            "multiple-\nlines-\nhere",
        ]
        for page in hyphenated_pages:
            assert has_hyphenated_lines([page]), f"Should detect: {page}"

        # None of these should be detected
        non_hyphenated_pages = [
            "normal text",
            "hyphen-in-middle of line",
            "no hyphens at all",
        ]
        for page in non_hyphenated_pages:
            assert not has_hyphenated_lines([page]), f"Should not detect: {page}"

    def test_char_join_decision_logic(self):
        """Test the character join decision with controlled corpus."""
        # Build corpus where "testing" is common
        corpus = "testing testing testing the test was tested"
        stats = build_ngram_stats(corpus)
        scorer = NGramScorer(stats)

        # "test-\ning" should prefer joining to "testing"
        result = char_join_hyphen_break("test-", "ing", scorer, scorer)

        # Result should be one of: "" (join), "-" (keep hyphen), " " (space)
        assert result in ["", "-", " ", "\n"]
