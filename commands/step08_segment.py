"""
step09_segment.py - sentence segmentation
"""

import json
from pathlib import Path

import click

from const.config import PipelineConfig, load_config
from const.types import BookJSON
from library.segment.nupunkt_segmenter import segment_book_nupunkt
from library.segment.sat_segmenter import SaT, segment_book_sat

# Module-level SAT model cache
_sat_model: SaT | None = None


def get_sat_model() -> SaT:
    """Get or load the SAT model (cached)."""
    global _sat_model
    if _sat_model is None:
        from library.segment.sat_segmenter import get_sat_model as load_sat

        _sat_model = load_sat()
    return _sat_model


def process_book(
    book: BookJSON,
    config: PipelineConfig,
    segmenter: str | None = None,
) -> BookJSON:
    """
    Process a single book through step 09 and segment sentences.
    """
    if segmenter is None:
        raise ValueError("Segmenter must be provided for segmentation step")

    if segmenter not in ("nupunkt", "sat"):
        raise ValueError(f"Invalid segmenter: {segmenter}. Must be 'nupunkt' or 'sat'.")

    if segmenter == "nupunkt":
        return segment_book_nupunkt(book, config=config)
    else:
        sat_model = get_sat_model()
        return segment_book_sat(book, config=config, sat_model=sat_model)


@click.command()
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-file", type=click.Path(path_type=Path), required=True)
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--segmenter", type=click.Choice(["nupunkt", "sat"]), required=True)
def main(input_file: Path, output_file: Path, config_file: Path | None, segmenter: str):
    """Run step 09 (segmentation) on a JSONL file."""
    config = load_config(config_file) if config_file else PipelineConfig()

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            result = process_book(book, config, segmenter=segmenter)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
