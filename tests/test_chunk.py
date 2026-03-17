"""Tests for library/chunk/ functions."""

from pathlib import Path

import numpy as np
import pytest

from const.config import PipelineConfig
from library.chunk.texttiling import (
    choose_boundaries,
    compute_depths,
    compute_windowed_similarities,
    segment_sentences,
    smooth_scores,
)
from library.chunk.utils import load_embedding_model, segments_from_starts

# Default embedding model path
_default_embedding_model = Path("./DATA/pretrain/models/BAAI_bge-m3_m2v_512dim")

# Check for C99 extension availability
try:
    from library.chunk.c99_banded_wrapper import c99_segment_banded

    _c99_available = True
except ImportError:
    _c99_available = False

# Skip markers
requires_embedding_model = pytest.mark.skipif(
    not _default_embedding_model.exists(),
    reason=f"Embedding model not available at {_default_embedding_model}",
)

requires_c99 = pytest.mark.skipif(
    not _c99_available, reason="C99 C++ extension not built"
)


class TestSegmentsFromStarts:
    def test_basic_segmentation(self):
        sentences = ["a", "b", "c", "d", "e"]
        starts = [0, 2, 4]
        result = segments_from_starts(sentences, starts)
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_single_segment(self):
        sentences = ["a", "b", "c"]
        starts = [0]
        result = segments_from_starts(sentences, starts)
        assert result == [["a", "b", "c"]]

    def test_empty_sentences(self):
        result = segments_from_starts([], [0])
        assert result == []

    def test_adds_zero_if_missing(self):
        sentences = ["a", "b", "c", "d"]
        starts = [2]  # Missing 0
        result = segments_from_starts(sentences, starts)
        assert result == [["a", "b"], ["c", "d"]]

    def test_deduplicates_starts(self):
        sentences = ["a", "b", "c", "d"]
        starts = [0, 0, 2, 2]
        result = segments_from_starts(sentences, starts)
        assert result == [["a", "b"], ["c", "d"]]


class TestComputeWindowedSimilarities:
    def test_identical_embeddings(self):
        # All same vector -> high similarity everywhere
        embeddings = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
        scores = compute_windowed_similarities(embeddings, window_size=1)
        assert len(scores) == 2
        assert all(s > 0.99 for s in scores)

    def test_orthogonal_embeddings(self):
        # Alternating orthogonal vectors -> low similarity
        embeddings = np.array(
            [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        scores = compute_windowed_similarities(embeddings, window_size=1)
        assert len(scores) == 3
        # Adjacent orthogonal vectors have ~0 similarity
        assert all(abs(s) < 0.1 for s in scores)

    def test_single_embedding(self):
        embeddings = np.array([[1, 0, 0]], dtype=np.float32)
        scores = compute_windowed_similarities(embeddings, window_size=1)
        assert len(scores) == 0

    def test_empty_embeddings(self):
        embeddings = np.array([], dtype=np.float32).reshape(0, 3)
        scores = compute_windowed_similarities(embeddings, window_size=1)
        assert len(scores) == 0


class TestSmoothScores:
    def test_smoothing_reduces_variance(self):
        scores = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        smoothed = smooth_scores(scores, width=3, passes=1)
        # Smoothing should reduce variance
        assert smoothed.std() < scores.std()

    def test_empty_scores(self):
        scores = np.array([], dtype=np.float32)
        smoothed = smooth_scores(scores, width=3, passes=1)
        assert len(smoothed) == 0

    def test_single_score(self):
        scores = np.array([0.5], dtype=np.float32)
        smoothed = smooth_scores(scores, width=3, passes=1)
        assert len(smoothed) == 1
        assert smoothed[0] == pytest.approx(0.5)


class TestComputeDepths:
    def test_valley_has_positive_depth(self):
        # Valley at index 1
        scores = np.array([0.8, 0.2, 0.8], dtype=np.float32)
        depths = compute_depths(scores)
        assert depths[1] > depths[0]
        assert depths[1] > depths[2]

    def test_peaks_have_zero_depth(self):
        # Peak at index 1
        scores = np.array([0.2, 0.8, 0.2], dtype=np.float32)
        depths = compute_depths(scores)
        assert depths[1] == pytest.approx(0.0)

    def test_empty_scores(self):
        scores = np.array([], dtype=np.float32)
        depths = compute_depths(scores)
        assert len(depths) == 0


class TestChooseBoundaries:
    def test_chooses_deep_valleys(self):
        # Clear valley at index 2
        depths = np.array([0.0, 0.1, 0.5, 0.1, 0.0], dtype=np.float32)
        boundaries = choose_boundaries(depths, depth_std=-1.0, min_gap=1)
        assert 2 in boundaries

    def test_respects_min_gap(self):
        # Two valleys close together
        depths = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        boundaries = choose_boundaries(depths, depth_std=-1.0, min_gap=3)
        # Only one should be chosen due to min_gap
        assert len(boundaries) <= 1

    def test_empty_depths(self):
        depths = np.array([], dtype=np.float32)
        boundaries = choose_boundaries(depths, depth_std=1.0, min_gap=1)
        assert boundaries == []


class TestSegmentSentencesMocked:
    """Tests for segment_sentences with mocked model."""

    def test_empty_sentences(self):
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        result = segment_sentences([], model=mock_model)
        assert result == []

    def test_single_sentence(self):
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        result = segment_sentences(["Hello."], model=mock_model)
        assert result == [0]


# =============================================================================
# Integration tests - Real embedding models
# =============================================================================


@requires_embedding_model
class TestTextTilingIntegration:
    """Integration tests for TextTiling chunking with real embedding model."""

    def test_load_embedding_model(self):
        """Test that embedding model loads successfully."""
        model = load_embedding_model(_default_embedding_model)
        assert model is not None
        assert hasattr(model, "encode")

    def test_model_encodes_sentences(self):
        """Test that model produces embeddings."""
        model = load_embedding_model(_default_embedding_model)
        sentences = ["Hello world.", "This is a test."]
        embeddings = model.encode(sentences)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Has embedding dimension

    def test_segment_sentences_with_real_model(self):
        """Test sentence segmentation with real model."""
        model = load_embedding_model(_default_embedding_model)

        sentences = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "Animals are wonderful pets.",
            "Now let us discuss mathematics.",
            "Two plus two equals four.",
            "Mathematics is a formal science.",
        ]

        starts = segment_sentences(sentences, model=model, min_segment_len=2)

        assert 0 in starts
        assert len(starts) >= 1

    def test_chunk_book_texttiling_with_real_model(self):
        """Test full book chunking with TextTiling."""
        from library.chunk.texttiling import chunk_book_texttiling

        book = {
            "barcode_src": "test",
            "middlematter_sentences": [
                "Chapter one begins with an introduction.",
                "The protagonist is introduced here.",
                "We learn about the setting.",
                "Now we move to chapter two.",
                "A new character appears.",
                "The plot thickens considerably.",
                "In conclusion, everything is resolved.",
                "The story ends happily.",
            ],
        }

        config = PipelineConfig()
        result = chunk_book_texttiling(book, config, model_dir=_default_embedding_model)

        assert "subtopic_paragraph_start_indices" in result
        assert "subtopic_section_start_indices" in result
        assert 0 in result["subtopic_paragraph_start_indices"]
        assert 0 in result["subtopic_section_start_indices"]

    def test_chunk_book_empty_sentences(self):
        """Test that empty sentences raises ValueError."""
        from library.chunk.texttiling import chunk_book_texttiling

        book = {
            "barcode_src": "test",
            "middlematter_sentences": [],
        }

        config = PipelineConfig()
        with pytest.raises(ValueError, match="No middlematter found"):
            chunk_book_texttiling(book, config, model_dir=_default_embedding_model)


@requires_embedding_model
@requires_c99
class TestC99Integration:
    """Integration tests for C99 chunking with real embedding model."""

    def test_c99_segment_banded(self):
        """Test C99 banded segmentation directly."""
        model = load_embedding_model(_default_embedding_model)

        sentences = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "Animals are wonderful pets.",
            "Now let us discuss mathematics.",
            "Two plus two equals four.",
            "Mathematics is a formal science.",
        ]

        from library.chunk.utils import compute_sentence_embeddings

        embeddings = compute_sentence_embeddings(sentences, model)
        segments = c99_segment_banded(embeddings, min_seg_len=2)

        assert len(segments) >= 1
        # Each segment is (start, end) tuple
        assert all(isinstance(s, tuple) and len(s) == 2 for s in segments)

    def test_chunk_book_c99_with_real_model(self):
        """Test full book chunking with C99."""
        from library.chunk.c99 import chunk_book_c99

        book = {
            "barcode_src": "test",
            "middlematter_sentences": [
                "Chapter one begins with an introduction.",
                "The protagonist is introduced here.",
                "We learn about the setting.",
                "Now we move to chapter two.",
                "A new character appears.",
                "The plot thickens considerably.",
                "In conclusion, everything is resolved.",
                "The story ends happily.",
            ],
        }

        config = PipelineConfig()
        result = chunk_book_c99(book, config, model_dir=_default_embedding_model)

        assert "subtopic_paragraph_start_indices" in result
        assert "subtopic_section_start_indices" in result
        assert 0 in result["subtopic_paragraph_start_indices"]
        assert 0 in result["subtopic_section_start_indices"]


@requires_c99
class TestC99ExtensionOnly:
    """Tests for C99 extension that don't require embedding model."""

    def test_c99_with_random_embeddings(self):
        """Test C99 segmentation with random embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)

        segments = c99_segment_banded(embeddings, min_seg_len=3)

        assert len(segments) >= 1
        # First segment starts at 0
        assert segments[0][0] == 0
        # Last segment ends at N-1 (inclusive end index)
        assert segments[-1][1] == 19

    def test_c99_respects_min_seg_len(self):
        """Test that C99 respects minimum segment length."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 64).astype(np.float32)

        segments = c99_segment_banded(embeddings, min_seg_len=5)

        # With min_seg_len=5 and only 10 sentences, should have at most 2 segments
        assert len(segments) <= 2
        for start, end in segments:
            # end is inclusive, so length is (end - start + 1)
            seg_len = end - start + 1
            assert seg_len >= 5 or end == 9  # Last segment may be shorter
