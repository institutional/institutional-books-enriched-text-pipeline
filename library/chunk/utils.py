"""
utils.py - Shared utilities for chunking

Extracted from prototype_pipeline/commands/chunk/chunk_utils.py

Institutional Books - Enriched Text - 2026
"""

from pathlib import Path

import numpy as np
from model2vec import StaticModel

from const.types import NormText


def load_embedding_model(model_dir: Path) -> StaticModel:
    """Load a StaticModel embedding model by directory path."""
    model = StaticModel.from_pretrained(str(model_dir))
    return model


def compute_sentence_embeddings(
    sentences: list[NormText],
    model: StaticModel,
) -> np.ndarray:
    """
    Compute L2-normalized embeddings for a list of sentences.

    Returns:
        np.ndarray of shape (N, D) where N is number of sentences
        and D is embedding dimension.
    """
    embeddings = model.encode(sentences, batch_size=256)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms
    return embeddings.astype(np.float32)


def segments_from_starts(
    sentences: list[NormText], segment_starts: list[int]
) -> list[list[NormText]]:
    """Build sentence segments from a list of segment start indices."""
    if not sentences:
        return []

    starts = sorted(set(segment_starts))
    if not starts or starts[0] != 0:
        starts = [0] + starts

    segments: list[list[NormText]] = []
    N = len(sentences)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else N
        if start < N:
            segments.append(sentences[start:end])
    return segments
