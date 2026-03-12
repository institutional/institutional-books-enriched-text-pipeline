"""
Tests for library/denoise/ngrams
"""

import tempfile
from pathlib import Path

import pytest

from library.denoise.ngrams import (
    NGramScorer,
    NGramStats,
    build_ngram_stats,
    load_ngram_stats,
    merge_ngram_stats,
    save_ngram_stats,
)


class TestNGramStats:
    """Tests for NGramStats dataclass."""

    def test_default_init(self):
        """Test default initialization creates empty stats."""
        stats = NGramStats(max_n=5)
        assert stats.vocab_size == 0
        for n in range(1, 6):
            assert n in stats.counts
            assert len(stats.counts[n]) == 0
            assert stats.total_counts.get(n, 0) == 0

    def test_post_init_creates_counters(self):
        """Test __post_init__ creates counters for all n values."""
        stats = NGramStats(max_n=5)
        for n in range(1, 6):
            assert n in stats.counts


class TestBuildNgramStats:
    """Tests for build_ngram_stats function."""

    def test_empty_text(self):
        """Test building stats from empty text."""
        stats = build_ngram_stats("")
        assert stats.vocab_size == 0
        assert stats.total_counts[1] == 0

    def test_single_word(self):
        """Test building stats from single word."""
        stats = build_ngram_stats("hello")
        # Characters: h, e, l, l, o
        assert stats.vocab_size == 4  # h, e, l, o (l appears twice but is same vocab)
        assert stats.total_counts[1] == 5  # 5 characters
        assert stats.counts[1]["h"] == 1
        assert stats.counts[1]["l"] == 2

    def test_text_with_spaces(self):
        """Test that spaces become <sp> tokens."""
        stats = build_ngram_stats("a b")
        # After normalization and tokenization: "a <sp> b"
        assert "<sp>" in stats.counts[1]
        assert stats.counts[1]["<sp>"] >= 1

    def test_ngram_order(self):
        """Test that n-grams are built up to max_n."""
        stats = build_ngram_stats("abcde", max_n=3)
        # Unigrams: a, b, c, d, e
        assert stats.total_counts[1] == 5
        # Bigrams: "a b", "b c", "c d", "d e"
        assert stats.total_counts[2] == 4
        # Trigrams: "a b c", "b c d", "c d e"
        assert stats.total_counts[3] == 3

    def test_unicode_normalization(self):
        """Test that text is properly normalized."""
        # Curly quotes should be normalized to straight quotes
        stats = build_ngram_stats("\u201chello\u201d")
        assert '"' in str(stats.counts[1].keys())


class TestNGramScorer:
    """Tests for NGramScorer class."""

    def test_score_raw_empty(self):
        """Test scoring empty text returns -inf."""
        stats = NGramStats(max_n=5)
        scorer = NGramScorer(stats)
        assert scorer.score_raw("") == float("-inf")

    def test_score_raw_basic(self):
        """Test basic scoring returns finite value."""
        stats = build_ngram_stats("hello world")
        scorer = NGramScorer(stats)
        score = scorer.score_raw("hello")
        assert score != float("-inf")
        assert score < 0  # Log probabilities are negative

    def test_score_raw_seen_vs_unseen(self):
        """Test that seen n-grams score higher than unseen."""
        stats = build_ngram_stats("hello hello hello")
        scorer = NGramScorer(stats)
        seen_score = scorer.score_raw("hello")
        unseen_score = scorer.score_raw("world")
        # Seen text should score higher (less negative)
        assert seen_score > unseen_score

    def test_backoff_for_unseen(self):
        """Test that backoff is used for unseen n-grams."""
        stats = build_ngram_stats("aaa")
        scorer = NGramScorer(stats, max_n=5)
        # "xyz" is unseen, should use backoff
        score = scorer.score_raw("xyz")
        assert score != float("-inf")  # Backoff should prevent -inf

    def test_short_text(self):
        """Test scoring text shorter than max_n."""
        stats = build_ngram_stats("hello world")
        scorer = NGramScorer(stats, max_n=5)
        # "hi" is only 2 chars, shorter than max_n=5
        score = scorer.score_raw("hi")
        assert score != float("-inf")

    def test_api_compatibility(self):
        """Test that score_raw matches CharLM API."""
        stats = build_ngram_stats("test text")
        scorer = NGramScorer(stats)
        # Should accept raw text and return float
        result = scorer.score_raw("test")
        assert isinstance(result, float)


class TestMergeNgramStats:
    """Tests for merge_ngram_stats function."""

    def test_merge_empty(self):
        """Test merging empty list returns empty stats."""
        merged = merge_ngram_stats([])
        assert merged.vocab_size == 0

    def test_merge_single(self):
        """Test merging single stats returns same stats."""
        stats = build_ngram_stats("hello")
        merged = merge_ngram_stats([stats])
        assert merged.vocab_size == stats.vocab_size

    def test_merge_two(self):
        """Test merging two stats combines counts."""
        stats1 = build_ngram_stats("aaa")
        stats2 = build_ngram_stats("bbb")
        merged = merge_ngram_stats([stats1, stats2])

        # Should have both 'a' and 'b' in vocab
        assert "a" in merged.counts[1]
        assert "b" in merged.counts[1]
        # Counts should be summed
        assert merged.counts[1]["a"] == 3
        assert merged.counts[1]["b"] == 3


class TestSaveLoadNgramStats:
    """Tests for save_ngram_stats and load_ngram_stats functions."""

    def test_roundtrip(self):
        """Test saving and loading preserves stats."""
        stats = build_ngram_stats("hello world")

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = Path(f.name)

        try:
            save_ngram_stats(stats, path)
            loaded = load_ngram_stats(path)

            assert loaded.vocab_size == stats.vocab_size
            assert loaded.total_counts == stats.total_counts
            for n in stats.counts:
                assert dict(loaded.counts[n]) == dict(stats.counts[n])
        finally:
            path.unlink(missing_ok=True)

    def test_load_nonexistent(self):
        """Test loading nonexistent file raises IOError."""
        with pytest.raises(IOError, match="not found"):
            load_ngram_stats(Path("/nonexistent/path.json.gz"))

    def test_file_is_gzipped_json(self):
        """Test that saved file is valid gzipped JSON."""
        import gzip
        import json

        stats = build_ngram_stats("test")

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = Path(f.name)

        try:
            save_ngram_stats(stats, path)

            # Should be readable as gzipped JSON
            with gzip.open(path, "rt") as f:
                data = json.load(f)

            assert "counts" in data
            assert "vocab_size" in data
        finally:
            path.unlink(missing_ok=True)


class TestNGramScorerEdgeCases:
    """Edge case tests for NGramScorer."""

    def test_single_character_text(self):
        """Test scoring single character."""
        stats = build_ngram_stats("a")
        scorer = NGramScorer(stats)
        score = scorer.score_raw("a")
        assert score != float("-inf")

    def test_whitespace_only(self):
        """Test scoring whitespace-only text."""
        stats = build_ngram_stats("hello")
        scorer = NGramScorer(stats)
        # After normalization, this becomes empty
        score = scorer.score_raw("   ")
        assert score == float("-inf")

    def test_special_characters(self):
        """Test scoring text with special characters."""
        stats = build_ngram_stats("hello! world?")
        scorer = NGramScorer(stats)
        score = scorer.score_raw("hello!")
        assert score != float("-inf")

    def test_repeated_text(self):
        """Test scoring repeated text."""
        stats = build_ngram_stats("abc abc abc")
        scorer = NGramScorer(stats)
        score1 = scorer.score_raw("abc")
        score2 = scorer.score_raw("abc abc")
        # Longer matching text should have lower (more negative) total score
        # but higher per-character score
        assert score1 != float("-inf")
        assert score2 != float("-inf")


class TestScorerBackoffBehavior:
    """Tests for stupid backoff smoothing behavior."""

    def test_backoff_penalty_applied(self):
        """Test that backoff penalty is applied for unseen n-grams."""
        stats = build_ngram_stats("aaaa")
        scorer = NGramScorer(stats, max_n=3, backoff=0.4)

        # "aaa" is seen, "bbb" is unseen
        seen_score = scorer.score_raw("aaa")
        unseen_score = scorer.score_raw("bbb")

        # Unseen should be much lower due to backoff penalties
        assert seen_score > unseen_score

    def test_custom_backoff_value(self):
        """Test that custom backoff value affects scores."""
        stats = build_ngram_stats("hello")

        scorer_low = NGramScorer(stats, backoff=0.1)
        scorer_high = NGramScorer(stats, backoff=0.9)

        # Unseen text
        score_low = scorer_low.score_raw("xyz")
        score_high = scorer_high.score_raw("xyz")

        # Higher backoff = less penalty = higher score for unseen
        assert score_high > score_low

    def test_add_k_smoothing(self):
        """Test that add-k smoothing prevents zero probabilities."""
        stats = build_ngram_stats("a")
        scorer = NGramScorer(stats, add_k=0.001)

        # 'z' is completely unseen unigram
        score = scorer.score_raw("z")
        # Should not be -inf due to add-k smoothing at unigram level
        assert score != float("-inf")
