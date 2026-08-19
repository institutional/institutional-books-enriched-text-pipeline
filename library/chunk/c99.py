"""
c99.py - C99-style topic chunking

C99 refers to "Advances in domain independent linear text segmentation"
by Choi (2000). Algorithm:
1. Compute embeddings for each sentence
2. Compute similarity matrix for sentence pairs
3. Replace similarity values with ranks in local neighborhoods
4. Identify subsegments with high local density

Note: The core C99 algorithm is implemented in C++ for performance.
This module provides the Python wrapper and book-level interface.

Institutional Books - Enriched Text - 2026
"""

from pathlib import Path

from loguru import logger
from model2vec import StaticModel

from const.config import PipelineConfig
from const.types import BookJSON, NormText
from library.chunk.utils import (
    compute_sentence_embeddings,
    load_embedding_model,
    segments_from_starts,
)


def chunk_book_c99(
    book: BookJSON,
    config: PipelineConfig,
    model: StaticModel | None = None,
    model_dir: Path | None = None,
    band_width: int = 80,
    mask_size: int = 9,
    min_segment_len: int = 3,
    max_segments: int = 0,
) -> BookJSON:
    """
    Chunk a book's sentences into topic-based paragraphs and sections.

    Args:
        book: Book dictionary with 'middlematter_sentences' field
        config: Pipeline configuration (should contain model_paths.embedding)
        model: Pre-loaded embedding model (optional)
        model_dir: Directory containing embedding model (overrides config)
        band_width: Band width around diagonal for similarity computation
        mask_size: Neighborhood mask size for rank-based similarity
        min_segment_len: Minimum sentences per segment
        max_segments: Maximum segments (0 = no limit)
    """
    sentences = book.get("middlematter_sentences", [])
    if not sentences:
        raise ValueError("No middlematter found.")

    # Load model if needed
    if model is None:
        if model_dir is None:
            model_dir = config.model_paths.embedding
        model = load_embedding_model(model_dir)

    # Segment into paragraphs
    paragraph_starts = segment_sentences(
        sentences,
        model=model,
        band_width=band_width,
        mask_size=mask_size,
        min_segment_len=min_segment_len,
        max_segments=max_segments,
    )

    # Segment paragraphs into sections
    paragraphs = segments_from_starts(sentences, paragraph_starts)
    section_starts = segment_sentences(
        [" ".join(p) for p in paragraphs],
        model=model,
        band_width=band_width,
        mask_size=mask_size,
        min_segment_len=min_segment_len,
        max_segments=max_segments,
    )

    result = book
    result["subtopic_paragraph_start_indices"] = paragraph_starts
    result["subtopic_section_start_indices"] = section_starts
    return result


def segment_sentences(
    sentences: list[NormText],
    model: StaticModel,
    band_width: int = 80,
    mask_size: int = 9,
    min_segment_len: int = 3,
    max_segments: int = 0,
) -> list[int]:
    """Segment a list of sentences into subtopic passages using C99."""
    N = len(sentences)
    if N == 0:
        return []
    if N == 1:
        return [0]

    embeddings = compute_sentence_embeddings(sentences, model=model)

    # Import C99 implementation
    try:
        from library.chunk.c99_banded_wrapper import c99_segment_banded

        segment_boundaries = c99_segment_banded(
            embeddings,
            band_width=band_width,
            mask_size=mask_size,
            min_seg_len=min_segment_len,
            max_segments=max_segments,
        )
        segment_starts = [start for start, _ in segment_boundaries]
    except ImportError:
        # Fallback: treat entire text as one segment
        logger.warning("C99 import not available. Using trivial chunking...")
        segment_starts = [0]

    return segment_starts
