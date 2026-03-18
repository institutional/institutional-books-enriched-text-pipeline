"""
compute_perplexity.py - core perplexity computation logic.
"""

import math
import sys

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from const.types import BookJSON, BookPerplexities
from library.chunk.utils import segments_from_starts


def load_perplexity_model(
    model_name: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM model and tokenizer for perplexity computation."""
    logger.info(f"Loading model {model_name} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    logger.info(f"Model ready on {device}.")
    return model, tokenizer


def compute_perplexity(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_length: int = 30000,
) -> float:
    """
    Compute perplexity for a single text.

    Returns -1 for text shorter than 5 characters (too short for meaningful perplexity).
    """
    if len(text) < 5:
        return -1.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
    )
    if inputs.get("overflowing_tokens") is not None and len(inputs["overflowing_tokens"]) > 0:
        logger.warning("Truncation occurred!")
        inputs.pop("overflowing_tokens")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

    return math.exp(loss)


def compute_perplexities_in_book(
    book: BookJSON,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> BookPerplexities:
    """
    Compute perplexities for all subtopic paragraph chunks in a book.
    """
    book_id = book.get("barcode_src", "UNKNOWN")
    if book_id == "UNKNOWN":
        raise ValueError("Unknown book encountered")

    sentences = book.get("middlematter_sentences", [])
    if not sentences:
        raise ValueError(f"No sentences found in {book_id}.")

    para_starts = book.get("subtopic_paragraph_start_indices", [])
    if not para_starts:
        raise ValueError(f"No paragraph indices in {book_id}. Run chunking steps first.")

    paragraphs: list[str] = [
        " ".join(segments) for segments in segments_from_starts(sentences, para_starts)
    ]

    perplexities: list[float] = []
    # disable tqdm when not run interactively
    for idx, para in tqdm(enumerate(paragraphs), disable=not sys.stderr.isatty()):
        ppl = compute_perplexity(para, model, tokenizer, device)
        perplexities.append(ppl)
        if idx % 500 == 499:
            logger.debug(f"Processed {idx} paragraphs from book.")

    return BookPerplexities(book_id=book_id, perplexities=perplexities)
