"""
simhash.py - Simhash implementation for near-duplicate detection

This is a pure python implementation of simhash using murmurhash3. This has two
roles:

    1. Provide a low-dependency duplication option
    2. Serve as a reference for testing a compiled C/CPP implementation

Institutional Books - Enriched Text - 2026
"""

import mmh3


def ngrams(text: str, n: int = 5) -> list[str]:
    """Convert to lowercase and return ngrams."""
    text = text.lower()
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def has_sufficient_entropy(ngram: str, threshold: float = 0.4) -> bool:
    """
    Check if an ngram has sufficient entropy to be included in simhash.

    Returns True if the number of unique characters is at least
    threshold * len(ngram).
    """
    return len(set(ngram)) >= threshold * len(ngram)


def simhash128(text: str, ngram_size: int = 5) -> int:
    """Compute a 128-bit simhash for the given text using murmurhash3."""
    # If the text is too short, hash the (lowercased) text directly
    if len(text) < 2 * ngram_size:
        text = text.lower()
        return mmh3.hash128(text.encode("utf-8"), signed=False)

    weights = [0] * 128
    for ngram in ngrams(text, ngram_size):
        if not has_sufficient_entropy(ngram):
            continue
        h = mmh3.hash128(ngram.encode("utf-8"), signed=False)
        for i in range(128):
            bitmask = 1 << i
            if h & bitmask:
                weights[i] += 1
            else:
                weights[i] -= 1

    ret = 0
    for i in range(128):
        if weights[i] >= 0:
            ret |= 1 << i
    return ret


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute the Hamming distance between two 128-bit hashes."""
    x = hash1 ^ hash2
    return bin(x).count("1")


def are_near_duplicates(hash1: int, hash2: int, threshold: int = 5) -> bool:
    """Return True if the two hashes are within the given threshold."""
    return hamming_distance(hash1, hash2) <= threshold


# Band configuration: 6 bands covering 128 bits
# Bits per band: [22, 21, 21, 21, 21, 22] = 128 total
BAND_BITS = [22, 21, 21, 21, 21, 22]
NUM_BANDS = len(BAND_BITS)


def extract_bands(simhash: int) -> list[int]:
    """
    Extract band values from a 128-bit simhash.

    Returns a list of 6 integers, one for each band.
    """
    bands: list[int] = []
    offset = 0
    for bits in BAND_BITS:
        mask = (1 << bits) - 1
        band_value = (simhash >> offset) & mask
        bands.append(band_value)
        offset += bits
    return bands
