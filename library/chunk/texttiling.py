"""
texttiling.py - TextTiling-style topic chunking

This is very closely modelled on TextTiling, except with static embeddings
instead of the adhoc embeddings from their paper. From far away: compute mean
embeddings of each sentence, compute pairwise similarities (with some
smoothing), and look for "valleys" and "peaks" to identify natural segmentation
points.

Algorithm:
1. Compute embeddings for each sentence
2. Compute similarities between consecutive sentence windows
3. Optionally smooth the similarity scores
4. Identify valleys in similarity scores to determine chunk boundaries
"""

from pathlib import Path

import numpy as np
from model2vec import StaticModel

from const.config import PipelineConfig
from const.types import BookJSON, NormText
from library.chunk.utils import (
    compute_sentence_embeddings,
    load_embedding_model,
    segments_from_starts,
)


def chunk_book_texttiling(
    book: BookJSON,
    config: PipelineConfig,
    model: StaticModel | None = None,
    model_dir: Path | None = None,
    window_size: int = 5,
    smooth_width: int = 3,
    smooth_passes: int = 1,
    depth_std: float = 0.5,
    min_segment_len: int = 3,
) -> BookJSON:
    """
    Chunk a book's sentences into topic-based paragraphs and sections.

    Args:
        book: Book dictionary with 'middlematter_sentences' field
        config: Pipeline configuration (should contain model_paths.embedding)
        model: Pre-loaded embedding model (optional)
        model_dir: Directory containing embedding model (overrides config)
        window_size: Window size for similarity computation
        smooth_width: Width of smoothing window
        smooth_passes: Number of smoothing passes
        depth_std: Depth threshold in std deviations (lower = longer segments)
        min_segment_len: Minimum sentences per segment
    """
    sentences = book.get("middlematter_sentences", [])
    if not sentences:
        raise ValueError("No middlematter found.")

    if model is None:
        if model_dir is None:
            model_dir = config.model_paths.embedding
        model = load_embedding_model(model_dir)

    # Segment into paragraphs
    paragraph_starts = segment_sentences(
        sentences,
        model=model,
        window_size=window_size,
        smooth_width=smooth_width,
        smooth_passes=smooth_passes,
        depth_std=depth_std,
        min_segment_len=min_segment_len,
    )

    # Segment paragraphs into sections
    paragraphs = segments_from_starts(sentences, paragraph_starts)
    section_starts = segment_sentences(
        [" ".join(p) for p in paragraphs],
        model=model,
        window_size=window_size,
        smooth_width=smooth_width,
        smooth_passes=smooth_passes,
        depth_std=depth_std,
        min_segment_len=min_segment_len,
    )

    result = book
    result["subtopic_paragraph_start_indices"] = paragraph_starts
    result["subtopic_section_start_indices"] = section_starts
    return result


def segment_sentences(
    sentences: list[NormText],
    model: StaticModel,
    window_size: int = 5,
    smooth_width: int = 3,
    smooth_passes: int = 1,
    depth_std: float = 1.0,
    min_segment_len: int = 3,
) -> list[int]:
    """Segment a list of sentences into subtopic passages."""
    N = len(sentences)
    if N == 0:
        return []
    if N == 1:
        return [0]

    embeddings = compute_sentence_embeddings(sentences, model=model)

    sim_scores = compute_windowed_similarities(embeddings, window_size=window_size)
    smoothed = smooth_scores(sim_scores, width=smooth_width, passes=smooth_passes)
    depths = compute_depths(smoothed)

    boundaries = choose_boundaries(
        depths,
        depth_std=depth_std,
        min_gap=max(1, min_segment_len - 1),
    )

    # Convert gap indices to sentence start indices
    segment_starts: list[int] = [0]
    start = 0
    for gap_idx in boundaries:
        end = gap_idx + 1
        if end - start >= min_segment_len:
            segment_starts.append(end)
            start = end

    return segment_starts


def compute_windowed_similarities(
    embeddings: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Compute cosine similarity between left/right windows of sentences
    around each sentence boundary.
    """
    N = embeddings.shape[0]
    if N < 2:
        return np.array([], dtype=np.float32)

    scores = np.zeros(N - 1, dtype=np.float32)

    for i in range(N - 1):
        left_start = max(0, i - window_size + 1)
        left_end = i
        right_start = i + 1
        right_end = min(N - 1, i + window_size)

        left_vec = embeddings[left_start : left_end + 1].mean(axis=0)
        right_vec = embeddings[right_start : right_end + 1].mean(axis=0)

        left_norm = np.linalg.norm(left_vec) + 1e-8
        right_norm = np.linalg.norm(right_vec) + 1e-8
        left_vec /= left_norm
        right_vec /= right_norm

        scores[i] = float(np.dot(left_vec, right_vec))

    return scores


def smooth_scores(
    scores: np.ndarray,
    width: int = 3,
    passes: int = 1,
) -> np.ndarray:
    """Simple moving-average smoothing over the similarity curve."""
    if scores.size == 0 or width < 1:
        return scores

    half = width // 2
    s = scores.astype(np.float32)

    for _ in range(passes):
        s_new = np.zeros_like(s)
        for i in range(len(s)):
            left = max(0, i - half)
            right = min(len(s) - 1, i + half)
            s_new[i] = s[left : right + 1].mean()
        s = s_new

    return s


def compute_depths(smoothed_scores: np.ndarray) -> np.ndarray:
    """Compute TextTiling-style valley depth for each gap."""
    M = smoothed_scores.size
    depths = np.zeros(M, dtype=np.float32)

    for i in range(M):
        # Find left peak
        left = i
        while left > 0 and smoothed_scores[left - 1] >= smoothed_scores[left]:
            left -= 1

        # Find right peak
        right = i
        while right < M - 1 and smoothed_scores[right + 1] >= smoothed_scores[right]:
            right += 1

        depth = (smoothed_scores[left] - smoothed_scores[i]) + (
            smoothed_scores[right] - smoothed_scores[i]
        )
        depths[i] = max(0.0, depth)

    return depths


def choose_boundaries(
    depths: np.ndarray,
    depth_std: float = 1.0,
    min_gap: int = 3,
) -> list[int]:
    """Choose boundary gaps based on depth values."""
    M = depths.size
    if M == 0:
        return []

    mu = float(depths.mean())
    sigma = float(depths.std())
    threshold = mu - depth_std * sigma

    # Candidate gaps with depth above threshold
    candidate_idxs = [i for i, d in enumerate(depths) if d >= threshold and 0 < i < M - 1]

    # Sort by depth descending, enforce min_gap
    candidate_idxs.sort(key=lambda i: depths[i], reverse=True)

    chosen: list[int] = []
    for idx in candidate_idxs:
        if not chosen:
            chosen.append(idx)
            continue

        if min(abs(idx - c) for c in chosen) >= min_gap:
            chosen.append(idx)

    chosen.sort()
    return chosen
