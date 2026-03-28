"""
types.py - user types
"""

from typing import Any, Callable, NewType, NotRequired, TypedDict

from const.config import PipelineConfig

type BookJSON = dict[str, Any]
type LastCompletedStep = str
type ErrorMessage = str


type LanguageCode = str
type BooksByLangDict = dict[LanguageCode, list[BookJSON]]


# Distinct types for better type checking
RawText = NewType("RawText", str)  # pre normalization
NormText = NewType("NormText", str)  # post normalization
CharTokens = NewType("CharTokens", str)

# Semantically different but not type different
type RawPage = RawText
type NormPage = NormText

type PageIdx = int
type SimhashHash = int


# StepFunction(book, config, 'nupunkt' | 'sat') -> BookJSON
type StepFunction = Callable[[BookJSON, PipelineConfig, str | None], BookJSON]


class ManifestStats(TypedDict):
    shard_id: str
    filename: str
    segmenter: str
    book_count: int


class ShardStats(TypedDict):
    total_books: int
    nupunkt_books: int
    sat_books: int
    nupunkt_shards: int
    sat_shards: int
    total_shards: int


class ProcessStats(TypedDict):
    total: int
    complete: int
    incomplete: int
    already_complete: NotRequired[int]


class BookSimhashes(TypedDict):
    book_id: str
    simhashes: list[SimhashHash]


class BookPerplexities(TypedDict):
    book_id: str
    perplexities: list[float]


class TextStats(TypedDict):
    """Text statistics for a book's middlematter."""

    token_count: int
    char_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    section_count: int
    bigram_count: int
    bigram_count_unique: int
    trigram_count: int
    trigram_count_unique: int
    tokenizability_o200k_base_ratio: float


class PerplexityStats(TypedDict, total=False):
    """Perplexity statistics for a book. All fields optional (empty if no valid perplexities)."""

    perplexity_min: float
    perplexity_max: float
    perplexity_median: float
    perplexity_avg: float
    perplexity_p10: float
    perplexity_p30: float
    perplexity_p70: float
    perplexity_p90: float
