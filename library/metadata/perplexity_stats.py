"""
library/metadata/perplexity_stats.py - perplexity statistics computation.
"""

import statistics

from const.types import PerplexityStats


def compute_perplexity_stats(perplexities: list[float]) -> PerplexityStats:
    """
    Compute perplexity statistics from a list of values.

    Excludes values where perplexity == -1 (too short to compute).

    Returns:
        Dictionary with perplexity statistics, or empty dict if no valid values.
    """
    # Filter out invalid perplexities (-1 means too short)
    valid = [p for p in perplexities if p > 0]

    if not valid:
        return {}

    sorted_vals = sorted(valid)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        """Compute p-th percentile (0-100)."""
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = f + 1 if f + 1 < n else f
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

    return {
        "perplexity_min": min(valid),
        "perplexity_max": max(valid),
        "perplexity_median": statistics.median(valid),
        "perplexity_avg": statistics.mean(valid),
        "perplexity_p10": percentile(10),
        "perplexity_p30": percentile(30),
        "perplexity_p70": percentile(70),
        "perplexity_p90": percentile(90),
    }
