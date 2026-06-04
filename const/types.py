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
SoftNormText = NewType("SoftNormText", str)  # after light normalization (NFC + spaces + zero-width)
HardNormText = NewType("HardNormText", str)  # after full normalization (NFKC + all)
NormText = SoftNormText  # general "normalized text" used throughout pipeline
CharTokens = NewType("CharTokens", str)

# Semantically different but not type different
type RawPage = RawText
type SoftNormPage = SoftNormText
type NormPage = SoftNormText

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


class BookBPB(TypedDict):
    book_id: str
    bpb_values: list[float]


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


class BPBStats(TypedDict, total=False):
    """Bits-per-byte statistics for a book. All fields optional (empty if no valid values)."""

    bpb_min: float
    bpb_max: float
    bpb_median: float
    bpb_avg: float
    bpb_p10: float
    bpb_p30: float
    bpb_p70: float
    bpb_p90: float
