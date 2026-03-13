"""
types.py - user types
"""

from typing import Any, Callable, NewType, TypedDict

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
