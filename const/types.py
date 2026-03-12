"""
types.py - user types
"""

from typing import Any, Callable, TypedDict

from const.config import PipelineConfig

type BookJSON = dict[str, Any]
type LastCompletedStep = str
type ErrorMessage = str


type LanguageCode = str
type BooksByLangDict = dict[LanguageCode, list[BookJSON]]


type RawPage = str

# After normalization
type NormPage = str

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
