"""
ngrams.py - pure python ngram language model logic

Fast, in-memory ngram tools. This is a cheap, less accurate, but sufficient
alternative to KenLM models.

Institutional Books - Enriched Text - 2026
"""

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from math import log10
from pathlib import Path

from const.types import HardNormText, RawText
from library.denoise.uniformize import normalize_text, to_char_tokens


@dataclass
class NGramStats:
    max_n: int

    # { n : { ngram : count } }
    counts: dict[int, Counter[str]] = field(default_factory=dict)  # type: ignore

    # { n : total count }
    total_counts: dict[int, int] = field(default_factory=dict)  # type: ignore

    # unique unigrams
    vocab_size: int = 0

    def __post_init__(self):
        # Set up counters
        for n in range(1, self.max_n + 1):
            if n not in self.counts:
                self.counts[n] = Counter()
            if n not in self.total_counts:
                self.total_counts[n] = 0


class NGramScorer:
    """
    Ngram language model with naive backoff smoothing.
    """

    def __init__(
        self,
        stats: NGramStats,
        max_n: int = 5,
        backoff: float = 0.4,
        add_k: float = 0.001,
    ):
        """
        Initialize scorer with precomputed ngram statistics.
        """
        self.stats = stats
        self.max_n = max_n
        self.backoff = backoff
        self.add_k = add_k
        self._log_backoff = log10(backoff)

    def _score_ngram(self, ngram: tuple[str, ...], n: int) -> float:
        """
        Score single ngram using backoff with add-k smoothing.
        """
        ngram_key = " ".join(ngram)
        count = self.stats.counts[n].get(ngram_key, 0)
        total = self.stats.total_counts[n]
        vocab = max(self.stats.vocab_size, 1)

        if count > 0 or n == 1:
            # with add-k smoothing
            return log10((count + self.add_k) / (total + self.add_k * vocab))
        else:
            # backoff to (n-1) gram with penalty
            return self._log_backoff + self._score_ngram(ngram[1:], n - 1)

    def score_raw(self, raw_text: RawText) -> float:
        """
        Returns log10 probability score for raw input text.
        """
        norm = normalize_text(raw_text)
        if not norm:
            return float("-inf")

        toks = to_char_tokens(norm)
        tokens = toks.split()

        if len(tokens) < self.max_n:
            # not enough tokens for full ngram
            if not tokens:
                return float("-inf")
            n = len(tokens)
            ngram = tuple(tokens)
            return self._score_ngram(ngram, n)

        # score all ngrams
        total_score = 0.0
        for i in range(len(tokens) - self.max_n + 1):
            ngram = tuple(tokens[i : i + self.max_n])
            total_score += self._score_ngram(ngram, self.max_n)

        return total_score


def build_ngram_stats(text: HardNormText, max_n: int = 5) -> NGramStats:
    stats = NGramStats(max_n=max_n)

    if not text:
        return stats

    toks = to_char_tokens(text)
    tokens = toks.split()

    if not tokens:
        return stats

    for n in range(1, max_n + 1):
        counter: Counter[str] = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            counter[ngram] += 1
        stats.counts[n] = counter
        stats.total_counts[n] = sum(counter.values())

    stats.vocab_size = len(stats.counts[1])
    return stats


def merge_ngram_stats(stats_list: list[NGramStats]) -> NGramStats:
    """
    Merge multiple NGramStats into one.

    For example: combine a language stat model with book stat model.
    """
    if not stats_list:
        return NGramStats(max_n=1)
    if len(stats_list) == 1:
        return stats_list[0]

    max_n = 0
    for stats in stats_list:
        max_n = max(max_n, stats.max_n)

    merged = NGramStats(max_n=max_n)
    all_unigrams: set[str] = set()

    for stats in stats_list:
        for n in stats.counts:
            if n not in merged.counts:
                merged.counts[n] = Counter()
            merged.counts[n].update(stats.counts[n])
            merged.total_counts[n] = merged.total_counts.get(n, 0) + stats.total_counts.get(n, 0)
        all_unigrams.update(stats.counts.get(1, {}).keys())

    merged.vocab_size = len(all_unigrams)
    return merged


def save_ngram_stats(stats: NGramStats, path: Path) -> None:
    """
    Save NGramStats to a gzipped JSON file.

    The format is human-readable when decompressed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "counts": {str(n): dict(c) for n, c in stats.counts.items()},
        "total_counts": {str(n): v for n, v in stats.total_counts.items()},
        "vocab_size": stats.vocab_size,
    }
    with gzip.open(path, "wt", encoding="utf8") as f:
        json.dump(data, f, separators=(",", ":"))


def load_ngram_stats(path: Path) -> NGramStats:
    """
    Load NGramStats from file.
    """
    if not path.exists():
        raise IOError(f"Ngram stats file not found: {path}")
    with gzip.open(path, "rt", encoding="utf8") as f:
        data = json.load(f)
    max_n = len(data["counts"])
    stats = NGramStats(max_n=max_n)
    stats.counts = {int(n): Counter(c) for n, c in data["counts"].items()}
    stats.total_counts = {int(n): v for n, v in data["total_counts"].items()}
    stats.vocab_size = data["vocab_size"]
    return stats
