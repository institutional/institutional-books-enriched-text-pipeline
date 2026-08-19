"""
simhash_fast.py - Unified Python interface for simhash, using C++ if available

This module provides a unified API for simhash operations. When the C++
extension is available (compiled from extensions/simhash_cpp/), it uses
the faster C++ implementation. Otherwise, it falls back to the pure
Python implementation in utils/simhash.py.

Both implementations produce identical results.

*Always use the C++ extension if available.*

Institutional Books - Enriched Text - 2026
"""

from __future__ import annotations

from utils.simhash import (
    are_near_duplicates as _py_are_near_duplicates,
)
from utils.simhash import (
    extract_bands as _py_extract_bands,
)
from utils.simhash import (
    hamming_distance as _py_hamming_distance,
)

# Import pure Python implementation as fallback
from utils.simhash import (
    simhash128 as _py_simhash_128,
)

# Try to import the C++ extension
_cpp_module = None

try:
    # Import from extensions.built package if available
    from extensions.built import _simhash_cpp

    _cpp_module = _simhash_cpp
except ImportError:
    pass


def is_cpp_available() -> bool:
    """Check if the C++ extension is available."""
    return _cpp_module is not None


def get_backend() -> str:
    """Get the current backend name ('cpp' or 'python')."""
    return "cpp" if _cpp_module is not None else "python"


def simhash128(text: str, ngram_size: int = 5) -> int:
    """
    Compute a 128-bit simhash for the given text.

    Uses C++ implementation if available, otherwise pure Python.
    Both implementations produce identical results.

    Args:
        text: Input text to hash
        ngram_size: Size of character n-grams (default: 5)

    Returns:
        128-bit simhash as a Python int
    """
    if _cpp_module is not None:
        return _cpp_module.simhash_128(text, ngram_size)
    else:
        return _py_simhash_128(text, ngram_size)


def simhash128_batch(texts: list[str], ngram_size: int = 5) -> list[int]:
    """
    Compute 128-bit simhashes for a batch of texts.

    Uses C++ implementation if available for better performance.

    Args:
        texts: List of input texts
        ngram_size: Size of character n-grams (default: 5)

    Returns:
        List of 128-bit simhashes as Python ints
    """
    if _cpp_module is not None:
        return _cpp_module.simhash_128_batch(texts, ngram_size)
    else:
        return [_py_simhash_128(text, ngram_size) for text in texts]


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Compute the Hamming distance between two 128-bit hashes.

    Uses C++ POPCNT intrinsics if available for fast bit counting.

    Args:
        hash1: First 128-bit hash
        hash2: Second 128-bit hash

    Returns:
        Number of differing bits (0-128)
    """
    if _cpp_module is not None:
        return _cpp_module.hamming_distance(hash1, hash2)
    else:
        return _py_hamming_distance(hash1, hash2)


def are_near_duplicates(hash1: int, hash2: int, threshold: int = 5) -> bool:
    """
    Check if two hashes are within the specified Hamming distance threshold.

    Args:
        hash1: First 128-bit hash
        hash2: Second 128-bit hash
        threshold: Maximum Hamming distance to be considered duplicates

    Returns:
        True if Hamming distance <= threshold
    """
    if _cpp_module is not None:
        return _cpp_module.are_near_duplicates(hash1, hash2, threshold)
    else:
        return _py_are_near_duplicates(hash1, hash2, threshold)


def extract_bands(hash_value: int) -> list[int]:
    """
    Extract band values from a 128-bit hash for LSH.

    Splits the hash into 6 bands for locality-sensitive hashing.

    Args:
        hash_value: 128-bit simhash

    Returns:
        List of 6 band values
    """
    if _cpp_module is not None:
        return _cpp_module.extract_bands(hash_value)
    else:
        return _py_extract_bands(hash_value)
