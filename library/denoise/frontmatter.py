"""
library/denoise/frontmatter.py - separate frontmatter and backmatter

Institutional Books - Enriched Text - 2026
"""

from pathlib import Path
from typing import Protocol

from const.config import PipelineConfig
from const.types import BookJSON, NormPage, PageIdx

type PageClassification = str


class PageClassifier(Protocol):
    def predict(self, texts: list[NormPage]) -> list[PageClassification]: ...


ENDMATTER_LABELS = {"ENDMATTER", "TOC_INDEX", "BIBLIO", "OTHERENDMATTER"}
MIDDLEMATTER_LABELS = {"MIDDLEMATTER"}


def separate_endmatter_book(
    book: BookJSON,
    classifier: PageClassifier,
    config: PipelineConfig | None = None,
) -> BookJSON:
    """
    Separate endmatter and middlematter from a book.

    Creates 'frontmatter', 'middlematter', and 'backmatter' fields from a
    'uniformized_text' field. Removes the 'uniformized_text' field.
    """
    uniformized_text: list[NormPage]
    uniformized_text = book.get("uniformized_text", "")
    if not uniformized_text:
        book_id = book.get("barcode_src", "UNKNOWN")
        raise ValueError(f"No 'uniformized_text' found in book {book_id}")
    front_end, main_end, _ = identify_frontmatter_backmatter(uniformized_text, classifier)

    result = book
    result["frontmatter"] = uniformized_text[:front_end]
    result["middlematter"] = uniformized_text[front_end:main_end]
    result["backmatter"] = uniformized_text[main_end:]

    # remove uniformized_text field
    result.pop("uniformized_text", None)

    return result


def identify_frontmatter_backmatter(
    pages: list[NormPage], classifier: PageClassifier
) -> tuple[PageIdx, PageIdx, PageIdx]:
    """
    Identify indices of pages separating frontmatter, main content, and
    backmatter.

    Returns a tuple of (frontmatter_end_index, main_content_end_index, total_pages),
    so that:
        frontmatter = raw_pages[:frontmatter_end_index]
        main_content = raw_pages[frontmatter_end_index:main_content_end_index]
        backmatter = raw_pages[main_content_end_index:]
    """
    # Predict all pages using m2v default batching
    predictions = classifier.predict(pages)

    # Empty pages may register as 'MIDDLEMATTER' by default; we treat them as 'ENDMATTER'
    predictions = [
        "ENDMATTER" if (pred == "MIDDLEMATTER" and not page.strip()) else pred
        for pred, page in zip(predictions, pages)
    ]

    total = len(pages)
    middle_indices = [i for i, lbl in enumerate(predictions) if lbl in MIDDLEMATTER_LABELS]

    # If no middlematter found, assume parsing failed; return no front/back matter
    if len(middle_indices) < 2:
        return 0, total, total

    # We assume frontmatter ends when middlematter begins, and backmatter
    # begins when middlematter ends.
    first_mid = middle_indices[0]
    front_end = first_mid

    last_mid = middle_indices[-1]
    main_end = last_mid + 1

    # Ensure indices are sane
    front_end = max(0, min(front_end, total))
    main_end = max(front_end, min(main_end, total))

    return front_end, main_end, total


def load_classifier(path: Path) -> PageClassifier:
    """
    Load pretrained m2v classifier.
    """
    from model2vec.inference import StaticModelPipeline

    return StaticModelPipeline.from_pretrained(path)  # type: ignore
