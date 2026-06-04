"""
compute_bpb.py - core bits-per-byte computation logic.

Institutional Books - Enriched Text - 2026
"""

import math

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from const.types import BookBPB, BookJSON
from library.chunk.utils import segments_from_starts

LOG2 = math.log(2)


def load_bpb_model(
    model_name: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM model and tokenizer for BPB computation."""
    logger.info(f"Loading model {model_name} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    logger.info(f"Model ready on {device}.")
    return model, tokenizer


def compute_bpb(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_length: int = 30000,
) -> float:
    """
    Compute bits-per-byte for a single text.

    Returns -1 for text shorter than 5 characters.
    """
    if len(text) < 5:
        return -1.0

    n_bytes = len(text.encode("utf-8"))

    token_count = len(tokenizer.encode(text, add_special_tokens=True))
    if token_count > max_length:
        logger.warning(f"Truncation occurred: {token_count} tokens exceeds max_length {max_length}")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        mean_loss = outputs.loss.item()

    n_predicted = inputs["input_ids"].shape[1] - 1
    total_loss = mean_loss * n_predicted
    return total_loss / (n_bytes * LOG2)


def compute_bpb_batched(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 32,
    max_length: int = 30000,
) -> list[float]:
    """
    Compute bits-per-byte for a list of texts using length-bucketed batched inference.

    Groups texts by token length into buckets so that padding within each batch
    is minimal. Uses right-padding with attention mask.

    Returns a list of BPB values (one per input text), with -1.0 for texts
    shorter than 5 characters.
    """
    results: list[float] = [0.0] * len(texts)

    batch_entries: list[tuple[int, str, int, int]] = []  # (idx, text, token_count, n_bytes)
    for i, text in enumerate(texts):
        if len(text) < 5:
            results[i] = -1.0
        else:
            token_count = len(tokenizer.encode(text, add_special_tokens=True))
            n_bytes = len(text.encode("utf-8"))
            batch_entries.append((i, text, token_count, n_bytes))

    if not batch_entries:
        return results

    batch_entries.sort(key=lambda x: x[2])

    original_padding_side = tokenizer.padding_side
    original_pad_token = tokenizer.pad_token_id
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        for batch_start in range(0, len(batch_entries), batch_size):
            batch_slice = batch_entries[batch_start : batch_start + batch_size]
            idx_slice = [entry[0] for entry in batch_slice]
            text_slice = [entry[1] for entry in batch_slice]
            bytes_slice = [entry[3] for entry in batch_slice]

            try:
                inputs = tokenizer(
                    text_slice,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                logits_f32 = logits.float()

                for j in range(len(text_slice)):
                    n_tokens = attention_mask[j].sum().item()
                    if n_tokens <= 1:
                        results[idx_slice[j]] = -1.0
                        continue
                    shift_logits_j = logits_f32[j, : n_tokens - 1, :]
                    shift_labels_j = input_ids[j, 1:n_tokens]
                    loss_sum = F.cross_entropy(
                        shift_logits_j, shift_labels_j, reduction="sum"
                    ).item()
                    results[idx_slice[j]] = loss_sum / (bytes_slice[j] * LOG2)

            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "INT_MAX" in str(e):
                    logger.warning(
                        f"Batch of {len(text_slice)} failed ({e}), falling back to sequential"
                    )
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    for j, text in enumerate(text_slice):
                        results[idx_slice[j]] = compute_bpb(
                            text, model, tokenizer, device, max_length
                        )
                else:
                    raise
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.pad_token_id = original_pad_token

    return results


def compute_bpb_in_book(
    book: BookJSON,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 16,
) -> BookBPB:
    """
    Compute bits-per-byte for all subtopic paragraph chunks in a book.

    Uses batched inference for throughput. Falls back to sequential on OOM.
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

    bpb_values = compute_bpb_batched(
        paragraphs, model, tokenizer, device, batch_size=batch_size
    )

    return BookBPB(book_id=book_id, bpb_values=bpb_values)
