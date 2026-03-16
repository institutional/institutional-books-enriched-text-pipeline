"""
nupunkt_segmenter.py - sentence segmentation using Nupunkt models

For each book, adapts the base language model to that book's text.
"""

from pathlib import Path
from typing import cast

import nupunkt
from nupunkt import PunktSentenceTokenizer
from nupunkt.training import train_model

from const.config import PipelineConfig
from const.types import BookJSON, NormText


def segment_book_nupunkt(
    book: BookJSON,
    config: PipelineConfig,
    model_dir: Path | None = None,
    adapt_model: bool = True,
) -> BookJSON:
    """
    Segment a book's middlematter into sentences using Nupunkt.

    Removes 'middlematter' and creates 'middlematter_sentences'.
    """
    lang = book.get("language_gen")
    if not lang:
        book_id = book.get("barcode_src", book.get("book_id", "unknown"))
        raise ValueError(f"No 'language_gen' found in book {book_id}")

    middlematter_pages = book.get("middlematter", [])
    if not middlematter_pages:
        book_id = book.get("barcode_src", book.get("book_id", "unknown"))
        raise ValueError(f"{book_id} missing middlematter")

    if model_dir is None:
        model_dir = config.model_paths.ngram if config else Path("./DATA/pretrain/models")
    model_file = locate_base_model_for_language(lang, model_dir)

    combined_middlematter = "\n".join(middlematter_pages)
    combined_middlematter = cast(NormText, combined_middlematter)
    if adapt_model:
        tokenizer = adapt_model_to_book_inmemory(
            base_model_path=model_file,
            corpus=combined_middlematter,
        )
        segments = tokenizer.tokenize(combined_middlematter)
    else:
        # Use base model directly
        segments = segment_text_with_nupunkt(model_path=model_file, text=combined_middlematter)

    result = book
    result["middlematter_sentences"] = segments
    result.pop("middlematter", None)
    return result


def locate_base_model_for_language(language_code: str, model_dir: Path) -> Path:
    """Locate the base NuPunkt model file for the given language code."""
    model_file = model_dir / f"{language_code}_nupunkt.bin"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Base model for language {language_code} not found at {model_file}"
        )
    return model_file


def adapt_model_to_book_inmemory(
    base_model_path: Path,
    corpus: str,
) -> PunktSentenceTokenizer:
    """
    Adapt the given NuPunkt model to the specific book text, entirely in-memory.

    Taken from https://github.com/alea-institute/nupunkt/blob/main/docs/training-guide.md#incremental-training
    """
    ## NOTE
    #
    # This is fragile and involves accessing a model's _params directly. We add
    # additional comments containing current behavior to help identify future
    # changes, if necessary.
    old_model = nupunkt.load(str(base_model_path))
    old_abbrevs = list(old_model._params.abbrev_types)  # type: ignore

    # Train in-memory (output_path=None means no file written)
    # https://github.com/alea-institute/nupunkt/blob/be0b754ff9637e411e411ccf8c35197b9f2dfde0/nupunkt/training/core.py#L201
    trainer = train_model(
        corpus,
        abbreviations=old_abbrevs,
        output_path=None,
        verbose=False,
    )
    return PunktSentenceTokenizer(trainer.get_params())


def segment_text_with_nupunkt(model_path: Path, text: NormText) -> list[NormText]:
    """Segment the given text using the specified NuPunkt model."""
    sentences = nupunkt.sent_tokenize(text, model=str(model_path), adaptive=True)
    return sentences  # type: ignore
