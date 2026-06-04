"""
library/metadata/language_stats.py - Language distribution computation.

Computes a distribution summary from pre-detected paragraph languages:
which languages appear (with at least 5 paragraphs) and their proportions.
"""

from collections import Counter

from library.annotate.language import UNKNOWN

MIN_PARAGRAPHS = 5


def compute_language_distribution(
    languages: list[str],
) -> dict[str, list[str] | list[float]]:
    """
    Compute language distribution from a list of per-paragraph language codes.

    Args:
        languages: List of ISO-639-3 language codes, one per paragraph.

    Returns:
        Dict with "languages" (sorted by frequency) and "proportion" (matching order).
        Empty lists if no language meets the minimum paragraph threshold.
    """
    if not languages:
        return {"languages": [], "proportion": []}

    counts = Counter(lang for lang in languages if lang != UNKNOWN)

    total_known = sum(counts.values())
    if total_known == 0:
        return {"languages": [], "proportion": []}

    filtered = {
        lang: count for lang, count in counts.items() if count >= MIN_PARAGRAPHS
    }

    if not filtered:
        return {"languages": [], "proportion": []}

    sorted_langs = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    langs = [lang for lang, _ in sorted_langs]
    proportions = [round(count / total_known, 4) for _, count in sorted_langs]

    return {"languages": langs, "proportion": proportions}
