"""Tests for library/metadata modules."""

import pytest

from library.metadata.perplexity_stats import compute_perplexity_stats


class TestComputePerplexityStats:
    def test_empty_list(self):
        result = compute_perplexity_stats([])
        assert result == {}

    def test_all_invalid_values(self):
        result = compute_perplexity_stats([-1, -1, -1])
        assert result == {}

    def test_single_value(self):
        result = compute_perplexity_stats([10.0])
        assert result["perplexity_min"] == 10.0
        assert result["perplexity_max"] == 10.0
        assert result["perplexity_median"] == 10.0
        assert result["perplexity_avg"] == 10.0

    def test_multiple_values(self):
        values = [5.0, 10.0, 15.0, 20.0, 25.0]
        result = compute_perplexity_stats(values)
        assert result["perplexity_min"] == 5.0
        assert result["perplexity_max"] == 25.0
        assert result["perplexity_median"] == 15.0
        assert result["perplexity_avg"] == 15.0

    def test_filters_invalid_values(self):
        values = [10.0, -1.0, 20.0, -1.0, 30.0]
        result = compute_perplexity_stats(values)
        # Should only consider 10, 20, 30
        assert result["perplexity_min"] == 10.0
        assert result["perplexity_max"] == 30.0
        assert result["perplexity_avg"] == 20.0

    def test_percentiles(self):
        # Use values that make percentiles easy to verify
        values = list(range(1, 101))  # 1 to 100
        result = compute_perplexity_stats(values)
        # Percentiles should be approximately at those positions
        assert 9 <= result["perplexity_p10"] <= 11
        assert 29 <= result["perplexity_p30"] <= 31
        assert 69 <= result["perplexity_p70"] <= 71
        assert 89 <= result["perplexity_p90"] <= 91

    def test_contains_expected_keys(self):
        result = compute_perplexity_stats([10.0, 20.0])
        expected_keys = {
            "perplexity_min",
            "perplexity_max",
            "perplexity_median",
            "perplexity_avg",
            "perplexity_p10",
            "perplexity_p30",
            "perplexity_p70",
            "perplexity_p90",
        }
        assert set(result.keys()) == expected_keys


class TestTextStats:
    """Tests for text_stats module - requires tiktoken and polyglot."""

    @pytest.fixture
    def skip_if_no_tiktoken(self):
        """Skip test if tiktoken is not available."""
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_count_tokens(self, skip_if_no_tiktoken):
        from library.metadata.text_stats import count_tokens

        result = count_tokens("Hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty(self, skip_if_no_tiktoken):
        from library.metadata.text_stats import count_tokens

        result = count_tokens("")
        assert result == 0

    def test_compute_word_ngrams(self):
        from library.metadata.text_stats import compute_word_ngrams

        words = ["the", "quick", "brown", "fox"]

        bigrams = compute_word_ngrams(words, 2)
        assert len(bigrams) == 3
        assert "the quick" in bigrams

        trigrams = compute_word_ngrams(words, 3)
        assert len(trigrams) == 2
        assert "the quick brown" in trigrams

    def test_compute_word_ngrams_short_input(self):
        from library.metadata.text_stats import compute_word_ngrams

        words = ["hello"]
        bigrams = compute_word_ngrams(words, 2)
        assert bigrams == []

    def test_compute_tokenizability(self, skip_if_no_tiktoken):
        from library.metadata.text_stats import compute_tokenizability

        # Simple words should have high tokenizability
        words = ["hello", "world", "this", "is", "a", "test"]
        result = compute_tokenizability(words)
        assert 0 <= result <= 100

    def test_compute_tokenizability_empty(self, skip_if_no_tiktoken):
        from library.metadata.text_stats import compute_tokenizability

        result = compute_tokenizability([])
        assert result == 0.0
