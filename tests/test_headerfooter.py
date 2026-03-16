"""
Tests for library/denoise/headerfooter.py

Migrated from prototype_pipeline/commands/denoise/test_headerfooter_removal.py
"""

import pytest

from library.denoise.headerfooter import (
    ngrams,
    clustered_group,
    remove_hf_lines,
    minhash_from_tokens,
    detect_and_clean_headers_footers,
)


class TestNgrams:
    """Test n-gram generation."""

    def test_5gram_basic(self):
        """Basic 5-gram generation."""
        result = list(ngrams("abcdefgh", n=5))
        expected = [
            ("a", "b", "c", "d", "e"),
            ("b", "c", "d", "e", "f"),
            ("c", "d", "e", "f", "g"),
            ("d", "e", "f", "g", "h"),
        ]
        assert result == expected

    def test_3gram(self):
        """3-gram generation."""
        result = list(ngrams("abcde", n=3))
        expected = [
            ("a", "b", "c"),
            ("b", "c", "d"),
            ("c", "d", "e"),
        ]
        assert result == expected

    def test_ngram_equals_length(self):
        """When n equals string length, one n-gram is produced."""
        result = list(ngrams("abc", n=3))
        assert result == [("a", "b", "c")]

    def test_ngram_exceeds_length(self):
        """When n exceeds string length, no n-grams are produced."""
        result = list(ngrams("ab", n=5))
        assert result == []

    def test_empty_string(self):
        """Empty string produces no n-grams."""
        result = list(ngrams("", n=3))
        assert result == []

    def test_single_char(self):
        """Single character with n=1."""
        result = list(ngrams("abc", n=1))
        assert result == [("a",), ("b",), ("c",)]


class TestClusteredGroup:
    """Test the clustered_group validation logic."""

    def test_valid_cluster_3_consecutive(self):
        """Three consecutive pages should be valid."""
        group = ((0, 0), (1, 0), (2, 0))
        assert clustered_group(group) is True

    def test_valid_cluster_within_5_pages(self):
        """Three pages within span of 5 should be valid."""
        group = ((0, 0), (2, 0), (4, 0))
        assert clustered_group(group) is True

    def test_invalid_cluster_too_spread(self):
        """Pages spread more than 5 apart should be invalid."""
        group = ((0, 0), (3, 0), (6, 0))
        assert clustered_group(group) is False

    def test_invalid_cluster_too_few(self):
        """Fewer than 3 pages should be invalid."""
        group = ((0, 0), (1, 0))
        assert clustered_group(group) is False

    def test_empty_group(self):
        """Empty group should be invalid."""
        group = ()
        assert clustered_group(group) is False

    def test_single_page(self):
        """Single page should be invalid."""
        group = ((0, 0),)
        assert clustered_group(group) is False

    def test_different_line_indices(self):
        """Line indices don't affect clustering (only page indices matter)."""
        group = ((0, 0), (1, 3), (2, 1))
        assert clustered_group(group) is True

    def test_exactly_5_page_span(self):
        """Span of exactly 5 pages (indices differ by 4) should be valid."""
        group = ((0, 0), (2, 0), (4, 0))
        assert clustered_group(group) is True

    def test_6_page_span_invalid(self):
        """Span of 6 pages (indices differ by 5) should be invalid."""
        group = ((0, 0), (2, 0), (5, 0))
        assert clustered_group(group) is False

    def test_many_pages_some_clustered(self):
        """Large group with some clustered subset should be valid."""
        group = ((0, 0), (1, 0), (2, 0), (10, 0), (20, 0))
        assert clustered_group(group) is True

    def test_many_pages_none_clustered(self):
        """Large group with no clustered subset should be invalid."""
        group = ((0, 0), (10, 0), (20, 0), (30, 0), (40, 0))
        assert clustered_group(group) is False


class TestRemoveHfLines:
    """Test the remove_hf_lines function."""

    def test_remove_single_line(self):
        """Remove a single line from one page."""
        pages = [["line0", "line1", "line2"]]
        lines_to_remove = {(0, 1)}
        result = remove_hf_lines(pages, lines_to_remove)
        assert result == [["line0", "line2"]]

    def test_remove_multiple_lines_same_page(self):
        """Remove multiple lines from same page."""
        pages = [["header", "content", "footer"]]
        lines_to_remove = {(0, 0), (0, 2)}
        result = remove_hf_lines(pages, lines_to_remove)
        assert result == [["content"]]

    def test_remove_from_multiple_pages(self):
        """Remove lines from different pages."""
        pages = [
            ["header1", "content1"],
            ["header2", "content2"],
        ]
        lines_to_remove = {(0, 0), (1, 0)}
        result = remove_hf_lines(pages, lines_to_remove)
        assert result == [["content1"], ["content2"]]

    def test_remove_nothing(self):
        """Empty removal set should preserve all lines."""
        pages = [["line0", "line1"]]
        lines_to_remove = set()
        result = remove_hf_lines(pages, lines_to_remove)
        assert result == pages

    def test_remove_all_lines(self):
        """Removing all lines should leave empty pages."""
        pages = [["line0", "line1"]]
        lines_to_remove = {(0, 0), (0, 1)}
        result = remove_hf_lines(pages, lines_to_remove)
        assert result == [[]]

    def test_empty_pages(self):
        """Empty pages should be handled."""
        pages = [[]]
        lines_to_remove = set()
        result = remove_hf_lines(pages, lines_to_remove)
        assert result == [[]]

    def test_invalid_indices_ignored(self):
        """Indices not in pages should be silently ignored."""
        pages = [["line0"]]
        lines_to_remove = {(0, 5), (5, 0)}
        result = remove_hf_lines(pages, lines_to_remove)
        assert result == [["line0"]]


class TestMinhashFromTokens:
    """Test MinHash generation from tokens."""

    def test_deterministic(self):
        """Same input should produce same MinHash."""
        mh1 = minhash_from_tokens("abcdefghij")
        mh2 = minhash_from_tokens("abcdefghij")
        assert mh1.jaccard(mh2) == 1.0

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different MinHashes."""
        mh1 = minhash_from_tokens("abcdefghij")
        mh2 = minhash_from_tokens("zyxwvutsrq")
        assert mh1.jaccard(mh2) < 1.0

    def test_similar_inputs_similar_hashes(self):
        """Similar inputs should have higher Jaccard similarity."""
        mh1 = minhash_from_tokens("abcdefghij")
        mh2 = minhash_from_tokens("abcdefghik")
        mh3 = minhash_from_tokens("zyxwvutsrq")
        assert mh1.jaccard(mh2) > mh1.jaccard(mh3)

    def test_custom_ngram_size(self):
        """Custom n-gram size should be respected."""
        mh_n3 = minhash_from_tokens("abcdefghij", n=3)
        mh_n5 = minhash_from_tokens("abcdefghij", n=5)
        assert mh_n3.jaccard(mh_n3) == 1.0
        assert mh_n5.jaccard(mh_n5) == 1.0

    def test_short_string(self):
        """Short strings (fewer ngrams) should still work."""
        mh = minhash_from_tokens("abcde", n=5)
        assert mh.jaccard(mh) == 1.0


class TestDetectAndRemoveHeadersFooters:
    """Integration tests for header/footer detection and removal."""

    def test_identical_headers_removed(self):
        """Identical headers across pages should be detected and removed."""
        pages = [
            "CHAPTER ONE\nContent of page 1\nMore content",
            "CHAPTER ONE\nContent of page 2\nMore content",
            "CHAPTER ONE\nContent of page 3\nMore content",
            "CHAPTER ONE\nContent of page 4\nMore content",
        ]
        cleaned_pages, removed = detect_and_clean_headers_footers(
            pages, header_size=1, sim_threshold=0.85
        )
        for page in cleaned_pages:
            assert "CHAPTER ONE" not in page

    def test_unique_content_preserved(self):
        """Unique content should not be removed."""
        pages = [
            "Unique header abc 1\nUnique content A bcd",
            "Unique header def 2\nUnique content B efg",
            "Unique header ghi 3\nUnique content C ijk",
        ]
        cleaned_pages, removed = detect_and_clean_headers_footers(pages)
        assert len(removed) == 0
        for original, cleaned in zip(pages, cleaned_pages):
            assert "\n".join(cleaned) == original

    def test_returns_removed_locations(self):
        """Function should return set of removed (page, line) locations."""
        pages = [
            "SAME FOOTER\nContent\nSAME FOOTER",
            "SAME FOOTER\nContent\nSAME FOOTER",
            "SAME FOOTER\nContent\nSAME FOOTER",
        ]
        _, removed = detect_and_clean_headers_footers(
            pages, header_size=1, footer_size=1, sim_threshold=0.85
        )
        assert isinstance(removed, set)
        for loc in removed:
            assert isinstance(loc, tuple)
            assert len(loc) == 2

    def test_empty_pages(self):
        """Empty pages should be handled gracefully."""
        pages = ["", "", ""]
        cleaned_pages, removed = detect_and_clean_headers_footers(pages)
        assert len(removed) == 0

    def test_short_lines_ignored(self):
        """Lines shorter than 6 characters should be ignored."""
        pages = [
            "Hi\nLonger content line here",
            "Hi\nLonger content line here",
            "Hi\nLonger content line here",
        ]
        _, removed = detect_and_clean_headers_footers(pages, header_size=2, sim_threshold=0.85)
        for loc in removed:
            assert loc[1] != 0
