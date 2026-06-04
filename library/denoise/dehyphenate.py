"""
dehyphenate.py - remove hyphenation artifacts

Typographical constraints from print media often lead to unusual text artifacts
when the physical context is removed. One such artifact is extra end-of-line
hyphenatation to split long words and make efficient use of line widths.

This module tries to detect and remove extra hyphens at ends of lines.

The approach is ngram-based. We combine a general language ngram model with
ngrams from the book in question, and compare 5-gram scores with end-of-line
hyphens removed vs kept-with-a-space vs kept-wthout-a-space.

NOTE: earlier versions of this module used the KenLM ngram language model.
"""

from pathlib import Path

from loguru import logger

from const.config import PipelineConfig
from const.types import BookJSON, HardNormText, NormPage, SoftNormText
from library.denoise.ngrams import NGramScorer, build_ngram_stats, load_ngram_stats
from library.denoise.uniformize import HYPHEN_LIKE

_END_OF_LINE_HYPHENS = HYPHEN_LIKE | {"-"}


def ends_with_hyphen(text: SoftNormText) -> bool:
    """Returns True if the given text ends with a hyphen-like character."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] in _END_OF_LINE_HYPHENS


def has_hyphenated_lines(pages: list[NormPage]) -> bool:
    """Check if any line in any page ends with a hyphen."""
    for page in pages:
        for line in page.splitlines():
            if ends_with_hyphen(SoftNormText(line)):
                return True
    return False


def dehyphenate_book(
    book: BookJSON,
    config: PipelineConfig | None = None,
    lm_cache: dict[str, NGramScorer] | None = None,
) -> BookJSON:
    """
    Dehyphenate middlematter pages in a book.

    Book needs to have 'middlematter' and 'language_gen' fields. The
    'middlematter' field will be replaced with a dehyphenated version.
    """
    book_id = book.get("barcode_src", "UNKNOWN")
    middlematter = book.get("middlematter", [])
    if not middlematter:
        raise ValueError(f"No middlematter found for book {book_id}")

    # Check if any lines end with hyphens
    if not has_hyphenated_lines(middlematter):
        logger.info(f"No end of line hyphens in book {book_id}.")
        return book

    lang = book.get("language_gen")
    if not lang:
        logger.info(f"No 'language_gen' found in book {book_id}")
        lang = "unknown"

    if lm_cache is None:
        lm_cache = {}

    model_dir = config.model_paths.ngram if config else Path("./DATA/pretrain/models")
    base_scorer = load_ngram_scorer(lang, model_dir, lm_cache)

    dehyphenated = dehyphenate_middlematter(middlematter, base_scorer)
    result = book
    result["middlematter"] = dehyphenated
    return result


def load_ngram_scorer(lang: str, model_dir: Path, cache: dict[str, NGramScorer]) -> NGramScorer:
    """Load the base n-gram scorer for the given language."""
    if lang in cache:
        return cache[lang]

    stats_path = model_dir / f"{lang}_ngram.json.gz"
    if not stats_path.exists():
        logger.warning(
            f"N-gram stats not found for language '{lang}': {model_dir}/{lang}_ngram.json.gz"
        )
        logger.warning("Defaulting to English for dehyphenation")
        stats_path = model_dir / "eng_ngram.json.gz"
        lang = "eng"
    stats = load_ngram_stats(stats_path)
    cache[lang] = NGramScorer(stats)
    return cache[lang]


def dehyphenate_middlematter(pages: list[NormPage], base_scorer: NGramScorer) -> list[NormPage]:
    """
    Apply dehyphenation to each page.

    Builds a book-specific model to help with dehyphenation decisions.
    """
    # Build hard-normalized corpus for ngram scoring
    corpus = "\n".join(pages)
    corpus = HardNormText(corpus)

    # Build n-gram stats in memory
    book_stats = build_ngram_stats(corpus, max_n=5)
    book_scorer = NGramScorer(book_stats)

    return [dehyphenate_page(page, base_scorer, book_scorer) for page in pages]


def dehyphenate_page(
    page_text: NormPage, base_scorer: NGramScorer, book_scorer: NGramScorer, window: int = 10
) -> NormPage:
    """Return page text with dehyphenation applied."""
    lines = page_text.splitlines()
    if not lines:
        return page_text

    result: list[SoftNormText] = []
    i = 0
    while i < len(lines):
        if i < len(lines) - 1 and ends_with_hyphen(lines[i].rstrip()):  # type: ignore
            join_char = char_join_hyphen_break(
                lines[i],  # type: ignore
                lines[i + 1],  # type: ignore
                base_scorer,
                book_scorer,
                window=window,
            )
            if join_char == "":
                stripped = lines[i].rstrip()
                while stripped and stripped[-1] in _END_OF_LINE_HYPHENS:
                    stripped = stripped[:-1]
                joined = stripped + lines[i + 1].lstrip()
                lines[i + 1] = joined
            elif join_char == "-":
                joined = lines[i].rstrip() + lines[i + 1].lstrip()
                lines[i + 1] = joined
            else:  # join_char == ' '
                joined = lines[i].rstrip() + " " + lines[i + 1].lstrip()
                lines[i + 1] = joined
        else:
            result.append(lines[i])  # type: ignore
        i += 1
    return "\n".join(result)  # type: ignore


def char_join_hyphen_break(
    prev_line: SoftNormText,
    next_line: SoftNormText,
    base_lm: NGramScorer,
    book_lm: NGramScorer,
    window: int = 10,
) -> str:
    """
    Determine whether a hyphen at the end of prev_line should be removed,
    kept with a space, or kept without a space.
    """
    prev_stripped = prev_line.strip()
    if not ends_with_hyphen(prev_stripped):  # type: ignore
        return "\n"

    next_stripped = next_line.strip()
    if not next_stripped:
        return "\n"

    cand_keep = prev_stripped[-window:] + next_stripped[:window]
    cand_join = prev_stripped[-window:-1] + next_stripped[:window]
    cand_space = prev_stripped[-window:] + " " + next_stripped[:window]

    score_keep = (book_lm.score_raw(cand_keep) + base_lm.score_raw(cand_keep)) / len(cand_keep)  # type: ignore
    score_join = (book_lm.score_raw(cand_join) + base_lm.score_raw(cand_join)) / len(cand_join)  # type: ignore
    score_space = (book_lm.score_raw(cand_space) + base_lm.score_raw(cand_space)) / len(cand_space)  # type: ignore

    if max(score_keep, score_join, score_space) == score_space:
        return " "
    if max(score_keep, score_join, score_space) == score_join:
        return ""
    return "-"
