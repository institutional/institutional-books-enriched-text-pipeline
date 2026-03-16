"""
sat.py - sentence segmentation using Segment-Any-Text

This is slow and wants a GPU.
"""

from typing import cast, no_type_check

from loguru import logger
from wtpsplit import SaT

from const.config import PipelineConfig
from const.types import BookJSON, NormText

# Global model cache
_sat_model: SaT | None = None


def segment_book_sat(
    book: BookJSON,
    config: PipelineConfig,
    sat_model: SaT | None = None,
) -> BookJSON:
    """
    Segment a book's middlematter into sentences using SAT.

    Removes 'middlematter' field and inserts 'middlematter_sentences' field.
    """
    middlematter_pages = book.get("middlematter", [])
    if not middlematter_pages:
        book_id = book.get("barcode_src", book.get("book_id", "unknown"))
        raise ValueError(f"{book_id} missing middlematter")

    if sat_model is None:
        sat_model = get_sat_model()

    combined_middlematter = "\n".join(middlematter_pages)
    combined_middlematter = cast(NormText, combined_middlematter)
    segments = segment_text_with_sat(sat_model, combined_middlematter)

    result = book
    result["middlematter_sentences"] = segments
    result.pop("middlematter", None)
    return result


def get_sat_model() -> SaT:
    """Get or load the SAT model (cached globally)."""
    global _sat_model
    if _sat_model is None:
        _sat_model = load_sat_model()
    return _sat_model


def get_device() -> str:
    """Get the best available device for SAT."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_sat_model() -> SaT:
    """Load the SaT model for segmentation."""
    device = get_device()
    if device == "cpu":
        raise RuntimeError(
            "No GPU available for SaT segmentation. Please run on a machine with a GPU."
        )
    logger.info(f"Loading SAT on {device}.")
    sat = SaT("sat-3l-sm")  # -sm means general segmentation
    sat.half().to(device)  # larger models are available, but are expensive
    return sat


@no_type_check
def segment_text_with_sat(sat_model: SaT, text: str) -> list[str]:
    """Segment the given text using the specified SaT model."""
    segments = sat_model.split(text)
    return segments
