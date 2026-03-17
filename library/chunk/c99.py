"""
c99.py - C99-style topic chunking

C99 refers to "Advances in domain independent linear text segmentation"
by Choi (1999). Algorithm:
1. Compute embeddings for each sentence
2. Compute similarity matrix for sentence pairs
3. Replace similarity values with ranks in local neighborhoods
4. Identify subsegments with high local density

Note: The core C99 algorithm is implemented in C++ for performance.
This module provides the Python wrapper and book-level interface.
"""

from pathlib import Path
from typing import Any

from model2vec import StaticModel

from library.chunk.utils import (
    load_embedding_model,
    compute_sentence_embeddings,
    segments_from_starts,
)


def chunk_book_c99(
    book: dict[str, Any],
    config: dict[str, Any] | None = None,
    model: StaticModel | None = None,
    model_dir: Path | None = None,
    band_width: int = 80,
    mask_size: int = 9,
    min_segment_len: int = 3,
    max_segments: int = 0,
) -> dict[str, Any]:
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

    Returns:
        Book dictionary with paragraph and section indices added
    """
    sentences = book.get("middlematter_sentences", [])
    if not sentences:
        return book

    # Load model if needed
    if model is None:
        if model_dir is None:
            if config is None:
                raise RuntimeError("No embedding model or config provided")
            model_dir = Path(
                config.get("model_paths", {}).get(
                    "embedding", "./DATA/distilled_models/BAAI_bge-m3_m2v_512dim"
                )
            )
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

    result = dict(book)
    result["subtopic_paragraph_start_indices"] = paragraph_starts
    result["subtopic_section_start_indices"] = section_starts
    return result


def segment_sentences(
    sentences: list[str],
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
        segment_starts = [0]

    return segment_starts
