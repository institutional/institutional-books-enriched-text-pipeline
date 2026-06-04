"""
step10_chunk.py - topic-based chunking

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path

import click
from model2vec import StaticModel

from const.config import ChunkingAlgorithm, PipelineConfig, load_config
from const.types import BookJSON
from library.chunk.c99 import chunk_book_c99
from library.chunk.texttiling import chunk_book_texttiling
from library.chunk.utils import load_embedding_model

# Module-level model cache
_embedding_model: StaticModel | None = None


def get_embedding_model(config: PipelineConfig) -> StaticModel:
    """Get or load the embedding model (cached)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedding_model(config.model_paths.embedding)
    return _embedding_model


def process_book(
    book: BookJSON,
    config: PipelineConfig,
    segmenter: str | None = None,
) -> BookJSON:
    """
    Process a single book through step 10 - chunking.

    Adds 'subtopic_paragraph_start_indices' and 'subtopic_section_start_indices' fields.
    """
    model = get_embedding_model(config)

    if config.chunking.algorithm == ChunkingAlgorithm.C99:
        return chunk_book_c99(book, config=config, model=model)
    else:
        return chunk_book_texttiling(book, config=config, model=model)


@click.command()
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-file", type=click.Path(path_type=Path), required=True)
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option(
    "--algorithm",
    type=click.Choice(["texttiling", "c99"]),
    default=None,
    help="Override chunking algorithm from config",
)
def main(
    input_file: Path,
    output_file: Path,
    config_file: Path | None,
    algorithm: str | None,
):
    """Run step 10 (chunking) on a JSONL file."""
    config = load_config(config_file) if config_file else PipelineConfig()

    # Override algorithm if specified on command line
    if algorithm is not None:
        config.chunking.algorithm = ChunkingAlgorithm(algorithm)

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            result = process_book(book, config)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
