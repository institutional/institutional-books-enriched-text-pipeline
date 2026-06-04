"""
library/metadata/bpb_stats.py - bits-per-byte statistics computation.

Institutional Books - Enriched Text - 2026
"""

import statistics

from const.types import BPBStats


def compute_bpb_stats(bpb_values: list[float]) -> BPBStats:
    """
    Compute BPB statistics from a list of values.

    Excludes values where bpb == -1 (too short to compute).

    Returns:
        Dictionary with BPB statistics, or empty dict if no valid values.
    """
    valid = [v for v in bpb_values if v > 0]

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
        "bpb_min": min(valid),
        "bpb_max": max(valid),
        "bpb_median": statistics.median(valid),
        "bpb_avg": statistics.mean(valid),
        "bpb_p10": percentile(10),
        "bpb_p30": percentile(30),
        "bpb_p70": percentile(70),
        "bpb_p90": percentile(90),
    }
