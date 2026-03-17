"""
library/annotate/endmatter.py - endmatter annotation for front/backmatter pages

Uses the em_subclassifier model to classify endmatter pages into

- TOC_INDEX: Table of contents or indices
- BIBLIO: Bibliography or references
- OTHERENDMATTER: Other endmatter content
"""

from pathlib import Path
from typing import Protocol

from const.types import NormPage
from library.annotate.tags import build_endmatter_tag


class EndmatterClassifier(Protocol):
    """Protocol for endmatter subclassifiers."""

    def predict(self, texts: list[NormPage]) -> list[str]: ...


def load_em_subclassifier(path: Path) -> EndmatterClassifier:
    """
    Load the pretrained endmatter subclassifier model.

    Args:
        path: Path to the em_subclassifier model directory.

    Returns:
        A classifier that predicts TOC_INDEX, BIBLIO, or OTHERENDMATTER.
    """
    from model2vec.inference import StaticModelPipeline

    return StaticModelPipeline.from_pretrained(path)  # type: ignore


def annotate_endmatter_pages(
    pages: list[NormPage],
    classifier: EndmatterClassifier,
) -> list[str]:
    """
    Annotate endmatter pages with their classification tags.

    Args:
        pages: List of page texts (frontmatter or backmatter).
        classifier: Endmatter subclassifier model.

    Returns:
        List of tagged page strings, one per page.
    """
    if not pages:
        return []

    # Classify all pages
    predictions = classifier.predict(pages)

    # Build tagged output for each page
    tagged_pages = []
    for page, label in zip(pages, predictions):
        tagged_pages.append(build_endmatter_tag(page, label))

    return tagged_pages
