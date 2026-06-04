"""Tests for library/metadata modules."""

import pytest

from library.metadata.bpb_stats import compute_bpb_stats


class TestComputeBPBStats:
    def test_empty_list(self):
        result = compute_bpb_stats([])
        assert result == {}

    def test_all_invalid_values(self):
        result = compute_bpb_stats([-1, -1, -1])
        assert result == {}

    def test_single_value(self):
        result = compute_bpb_stats([0.85])
        assert result["bpb_min"] == 0.85
        assert result["bpb_max"] == 0.85
        assert result["bpb_median"] == 0.85
        assert result["bpb_avg"] == 0.85

    def test_multiple_values(self):
        values = [0.5, 1.0, 1.5, 2.0, 2.5]
        result = compute_bpb_stats(values)
        assert result["bpb_min"] == 0.5
        assert result["bpb_max"] == 2.5
        assert result["bpb_median"] == 1.5
        assert result["bpb_avg"] == 1.5

    def test_filters_invalid_values(self):
        values = [0.8, -1.0, 1.2, -1.0, 1.6]
        result = compute_bpb_stats(values)
        assert result["bpb_min"] == 0.8
        assert result["bpb_max"] == 1.6
        assert abs(result["bpb_avg"] - 1.2) < 0.001

    def test_percentiles(self):
        values = [i / 100 for i in range(1, 101)]  # 0.01 to 1.0
        result = compute_bpb_stats(values)
        assert 0.09 <= result["bpb_p10"] <= 0.11
        assert 0.29 <= result["bpb_p30"] <= 0.31
        assert 0.69 <= result["bpb_p70"] <= 0.71
        assert 0.89 <= result["bpb_p90"] <= 0.91

    def test_contains_expected_keys(self):
        result = compute_bpb_stats([0.8, 1.2])
        expected_keys = {
            "bpb_min",
            "bpb_max",
            "bpb_median",
            "bpb_avg",
            "bpb_p10",
            "bpb_p30",
            "bpb_p70",
            "bpb_p90",
        }
        assert set(result.keys()) == expected_keys


class TestTextStats:
    """Tests for text_stats module - requires tiktoken and polyglot."""

    @pytest.fixture
    def skip_if_no_tiktoken(self):
        """Skip test if tiktoken is not available."""
        try:
            import tiktoken  # noqa: F401
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
