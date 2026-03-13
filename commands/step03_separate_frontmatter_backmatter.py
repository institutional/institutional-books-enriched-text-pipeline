"""
step03_separate_frontmatter_backmatter.py - separate FMBM

This uses pretrained model2vec classifiers.
"""

import json
from pathlib import Path
from typing import Any

import click

from const.config import PipelineConfig, load_config
from const.types import BookJSON
from library.denoise.frontmatter import (
    separate_endmatter_book,
    load_classifier,
    PageClassifier,
)

# module-level classifier cache
_classifier: PageClassifier | None = None


def get_classifier(config: PipelineConfig) -> PageClassifier:
    """
    Get or load the MM/EM classifier.
    """
    global _classifier
    if _classifier is None:
        # The classifier path could come from config.
        # This is the default path.
        module_path = Path(__file__).parent
        classifier_path = module_path / "../DATA/pretrain/models/mmem_classifier/"
        _classifier = load_classifier(classifier_path)
    return _classifier


def process_book(
    book: BookJSON,
    config: PipelineConfig,
    segmenter: str | None = None,
    classifier: PageClassifier | None = None,
) -> BookJSON:
    """
    Process a single book through step 03.

    Separates middlematter from endmatter based on static embeddings of pages.
    Removes the 'uniformized_text' field and adds 'frontmatter', 'middlematter',
    and 'backmatter' fields.
    """
    if classifier is None:
        classifier = get_classifier(config)

    return separate_endmatter_book(book, config=None, classifier=classifier)


@click.command()
@click.option("--input-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-file", type=click.Path(path_type=Path), required=True)
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), default=None)
def main(input_file: Path, output_file: Path, config_file: Path | None):
    """Run step 03 (frontmatter separation) on a JSONL file."""
    config = load_config(config_file) if config_file else PipelineConfig()
    classifier = get_classifier(config)

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            book = json.loads(line)
            result = process_book(book, config, classifier=classifier)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
