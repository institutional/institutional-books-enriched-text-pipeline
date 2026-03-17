"""
c99_banded_wrapper.py - Python wrapper for C99 banded segmentation

Provides a unified interface to the C99 algorithm for topic segmentation.
Requires the C++ extension (no pure Python fallback exists).

The C++ extension can be built from extensions/c99_cpp/.

Fallback philosophy: Fail fast with ImportError. Unlike simhash which has a
pure Python fallback, C99 has no fallback implementation. Callers must decide
how to handle the missing extension (e.g., skip segmentation, use single segment).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Try to import the C++ extension
_cpp_module = None

try:
    # First try: installed as a package
    import _c99_cpp
    _cpp_module = _c99_cpp
except ImportError:
    try:
        # Second try: built in-place in extensions/c99_cpp/
        ext_dir = Path(__file__).parent.parent.parent / "extensions" / "c99_cpp"
        if ext_dir.exists():
            sys.path.insert(0, str(ext_dir))
            import _c99_cpp
            _cpp_module = _c99_cpp
            sys.path.pop(0)
    except ImportError:
        pass

if _cpp_module is None:
    raise ImportError(
        "C99 C++ extension not found. Build it with:\n"
        "  cd extensions/c99_cpp && python setup.py build_ext --inplace"
    )


def c99_segment_banded(
    embeddings: np.ndarray,
    band_width: int = 80,
    mask_size: int = 9,
    min_seg_len: int = 3,
    max_segments: int = 0,
    density_gain_threshold: float = 0.0,
) -> list[tuple[int, int]]:
    """
    Segment sentences using the C99 banded algorithm.

    Args:
        embeddings: Sentence embeddings, shape (N, dim), float32 or float64.
                   Will be converted to contiguous float32.
        band_width: Width of the band around diagonal for similarity (default: 80)
        mask_size: Size of neighborhood mask for rank computation (default: 9)
        min_seg_len: Minimum sentences per segment (default: 3)
        max_segments: Maximum number of segments, 0 = no limit (default: 0)
        density_gain_threshold: Minimum gain to continue splitting (default: 0.0)

    Returns:
        List of (start, end) tuples for each segment.
    """
    # Ensure contiguous float32 array
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Call the C++ implementation
    segments = _cpp_module.c99_segment_banded(
        emb,
        band_width=band_width,
        mask_size=mask_size,
        min_seg_len=min_seg_len,
        max_segments=max_segments,
        density_gain_threshold=density_gain_threshold,
    )

    return segments
