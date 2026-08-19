"""
step04_dehyphenate.py - dehyphenation

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path
from typing import Any

import click

from const.config import PipelineConfig, load_config
from library.denoise.dehyphenate import NGramScorer, dehyphenate_book

# Single-language scorer cache (evicts on language change to limit memory)
_scorer_cache: dict[str, NGramScorer] = {}


def _get_single_lang_cache(lang: str) -> dict[str, NGramScorer]:
    """Return cache, evicting if language changed."""
    global _scorer_cache
    if _scorer_cache and lang not in _scorer_cache:
        _scorer_cache.clear()
    return _scorer_cache


def process_book(
    book: dict[str, Any],
    config: PipelineConfig,
    segmenter: str | None = None,
) -> dict[str, Any]:
    """
    Process a single book through step 04.

    Args:
        book: Book dictionary with 'middlematter' and 'language_gen' fields
        config: Pipeline configuration
        segmenter: Segmenter type (unused for this step)

    Returns:
        Processed book dictionary with dehyphenated middlematter
    """
    lang = book.get("language_gen", "")
    if not lang:
        # default to English, the most common --- and perform dehyphenation
        lang = "eng"
    cache = _get_single_lang_cache(lang)
    return dehyphenate_book(book, config=config, lm_cache=cache)


@click.command()
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-file", type=click.Path(path_type=Path), required=True)
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), default=None)
def main(input_file: Path, output_file: Path, config_file: Path | None):
    """Run step 04 (dehyphenation) on a JSONL file."""
    config = load_config(config_file) if config_file else PipelineConfig()

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            result = process_book(book, config)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
