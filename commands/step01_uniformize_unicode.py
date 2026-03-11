"""
step01_uniformize_unicode.py - unicode normalization
"""

import json
from pathlib import Path

import click
from loguru import logger

from const.config import PipelineConfig, load_config
from const.types import BookJSON
from library.denoise.uniformize import uniformize_book


def process_book(
    book: BookJSON,
    config: PipelineConfig,
    segmenter: str | None = None,
) -> BookJSON:
    """
    Process a single book through step 01.

    Uniformizes unicode on text pages (from 'text_by_page_src' field) and returns a JSON
    with the new 'uniformized_text' field. Old text fields are removed.
    """
    return uniformize_book(book)


@click.command()
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-file", type=click.Path(path_type=Path), required=True)
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), default=None)
def main(input_file: Path, output_file: Path, config_file: Path | None):
    """Run step 01 (unicode normalization) on a JSONL file."""
    logger.info(f"Running Step 1 on {input_file}.")
    config = load_config(config_file) if config_file else PipelineConfig()
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            result = process_book(book, config)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
