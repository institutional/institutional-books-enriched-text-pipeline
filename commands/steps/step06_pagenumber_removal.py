"""
step06_pagenumber_removal.py - remove page numbers
"""

import json
from pathlib import Path

import click

from const.config import PipelineConfig, load_config
from const.types import BookJSON
from library.denoise.pagenumber import remove_page_numbers_book


def process_book(
    book: BookJSON,
    config: PipelineConfig,
    segmenter: str | None = None,
) -> BookJSON:
    """
    Process a single book through step 06: remove page numbers.

    This modifies the 'middlematter' field.
    """
    return remove_page_numbers_book(book, config=None)


@click.command()
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-file", type=click.Path(path_type=Path), required=True)
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), default=None)
def main(input_file: Path, output_file: Path, config_file: Path | None):
    """Run step 06 (page number removal) on a JSONL file."""
    config = load_config(config_file) if config_file else PipelineConfig()

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            result = process_book(book, config)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
