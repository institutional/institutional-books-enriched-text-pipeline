"""
library/annotate/endmatter.py - endmatter annotation for front/backmatter pages

Uses the em_subclassifier model to classify endmatter pages into

- TOC_INDEX: Table of contents or indices
- BIBLIO: Bibliography or references
- OTHERENDMATTER: Other endmatter content
"""

import warnings
from pathlib import Path
from typing import Protocol

from const.types import NormPage
from library.annotate.tags import (
    build_backmatter_tag,
    build_endmatter_page_tag,
    build_frontmatter_tag,
)


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


def _annotate_endmatter_pages(
    pages: list[NormPage],
    classifier: EndmatterClassifier,
) -> list[str]:
    """
    Classify and tag endmatter pages as divs.

    Args:
        pages: List of page texts.
        classifier: Endmatter subclassifier model.

    Returns:
        List of tagged page divs, or empty list if no non-empty pages.
    """
    non_empty_pages = [p for p in pages if p and p.strip()]

    if not non_empty_pages:
        return []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        predictions = classifier.predict(non_empty_pages)

    return [
        build_endmatter_page_tag(page, label)
        for page, label in zip(non_empty_pages, predictions)
    ]


def annotate_frontmatter(
    pages: list[NormPage],
    classifier: EndmatterClassifier,
) -> str:
    """
    Annotate frontmatter pages wrapped in a <header> tag.

    Args:
        pages: List of frontmatter page texts.
        classifier: Endmatter subclassifier model.

    Returns:
        Tagged string like: <header><div class="toc_index">...</div>...</header>
    """
    tagged = _annotate_endmatter_pages(pages, classifier)
    if not tagged:
        return ""
    return build_frontmatter_tag("\n".join(tagged))


def annotate_backmatter(
    pages: list[NormPage],
    classifier: EndmatterClassifier,
) -> str:
    """
    Annotate backmatter pages wrapped in a <footer> tag.

    Args:
        pages: List of backmatter page texts.
        classifier: Endmatter subclassifier model.

    Returns:
        Tagged string like: <footer><div class="biblio">...</div>...</footer>
    """
    tagged = _annotate_endmatter_pages(pages, classifier)
    if not tagged:
        return ""
    return build_backmatter_tag("\n".join(tagged))
