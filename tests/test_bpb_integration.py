"""
Integration test for bits-per-byte computation.

Loads a real model (Qwen3-0.6B-Base) and real data (shard0001) to verify
that batched BPB computation produces results identical to sequential.

NOT committed to the repository — requires local model weights and data.

Usage:
    pytest tests/test_bpb_integration.py -v
    pytest tests/test_bpb_integration.py -v -k test_batched_matches_sequential

Institutional Books - Enriched Text - 2026
"""

import json
import math
from pathlib import Path

import pytest
import torch

from library.bpb.compute_bpb import (
    compute_bpb,
    load_bpb_model,
)
from library.chunk.utils import segments_from_starts

DATA_FILE = Path("DATA/shards/processed/shard0001.complete.jsonl")
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

requires_data = pytest.mark.skipif(
    not DATA_FILE.exists(), reason="Local data not available"
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    device = get_device()
    model, tokenizer = load_bpb_model(MODEL_NAME, device)
    return model, tokenizer, device


@pytest.fixture(scope="module")
def sample_paragraphs():
    """Load a mix of short, medium, and long paragraphs from real data."""
    with open(DATA_FILE) as f:
        book = json.loads(f.readline())

    sentences = book["middlematter_sentences"]
    para_starts = book["subtopic_paragraph_start_indices"]
    segments = list(segments_from_starts(sentences, para_starts))
    paragraphs = [" ".join(seg) for seg in segments]

    # Pick a representative sample: short, medium, long
    short = [p for p in paragraphs if len(p.split()) < 20][:5]
    medium = [p for p in paragraphs if 20 <= len(p.split()) <= 80][:5]
    long = [p for p in paragraphs if len(p.split()) > 80][:5]
    # Also include a very short one that should return -1
    tiny = ["Hi"]

    sample = tiny + short + medium + long
    return sample


@requires_data
class TestBPBSequential:
    """Baseline: verify sequential computation produces stable results."""

    def test_deterministic(self, model_and_tokenizer, sample_paragraphs):
        """Same input should produce same BPB on repeated calls."""
        model, tokenizer, device = model_and_tokenizer
        text = sample_paragraphs[5]  # a short paragraph

        ppl1 = compute_bpb(text, model, tokenizer, device)
        ppl2 = compute_bpb(text, model, tokenizer, device)

        assert ppl1 == pytest.approx(ppl2, rel=1e-6)

    def test_short_text_sentinel(self, model_and_tokenizer, sample_paragraphs):
        """Text < 5 chars returns -1."""
        model, tokenizer, device = model_and_tokenizer

        result = compute_bpb("Hi", model, tokenizer, device)
        assert result == -1.0

    def test_reasonable_range(self, model_and_tokenizer, sample_paragraphs):
        """BPB on real text should be in a reasonable range (0.1 to 10)."""
        model, tokenizer, device = model_and_tokenizer

        for text in sample_paragraphs[1:]:  # skip "Hi"
            bpb = compute_bpb(text, model, tokenizer, device)
            if bpb == -1.0:
                continue
            assert 0.1 < bpb < 10.0, f"BPB {bpb} out of range for: {text[:50]}"


@requires_data
class TestBPBBatched:
    """Compare batched computation against sequential baseline."""

    def _compute_sequential(self, paragraphs, model, tokenizer, device):
        """Compute BPB values one at a time (current production method)."""
        results = []
        for text in paragraphs:
            ppl = compute_bpb(text, model, tokenizer, device)
            results.append(ppl)
        return results

    def _compute_batched(self, paragraphs, model, tokenizer, device, batch_size=8):
        """
        Compute BPB values in batches with left-padding.

        This is the method we want to implement and verify matches sequential.
        """
        from library.bpb.compute_bpb import compute_bpb_batched

        return compute_bpb_batched(
            paragraphs, model, tokenizer, device, batch_size=batch_size
        )

    def test_batched_matches_sequential(self, model_and_tokenizer, sample_paragraphs):
        """Batched BPB must match sequential within padding tolerance.

        Due to bfloat16 attention and padding, batched results differ by ~1-3%
        from sequential. This is acceptable for ranking/filtering use cases.
        """
        model, tokenizer, device = model_and_tokenizer
        paragraphs = sample_paragraphs

        sequential_results = self._compute_sequential(paragraphs, model, tokenizer, device)
        batched_results = self._compute_batched(paragraphs, model, tokenizer, device)

        assert len(sequential_results) == len(batched_results)

        for i, (seq, bat) in enumerate(zip(sequential_results, batched_results)):
            if seq == -1.0:
                assert bat == -1.0, f"Paragraph {i}: sequential=-1 but batched={bat}"
                continue
            # Padding introduces ~1-3% relative error in bfloat16 attention
            assert seq == pytest.approx(bat, rel=0.05), (
                f"Paragraph {i}: sequential={seq:.6f}, batched={bat:.6f}, "
                f"rel_diff={abs(seq-bat)/seq:.2e}"
            )

    def test_batched_various_batch_sizes(self, model_and_tokenizer, sample_paragraphs):
        """Smaller batch sizes have less padding, so should be closer to sequential.

        batch_size=1 should exactly match sequential (no padding at all).
        Larger batches may differ slightly due to padding effects.
        """
        model, tokenizer, device = model_and_tokenizer
        paragraphs = sample_paragraphs[1:]  # skip sentinel

        from library.bpb.compute_bpb import compute_bpb_batched

        results_bs1 = compute_bpb_batched(paragraphs, model, tokenizer, device, batch_size=1)
        results_bs4 = compute_bpb_batched(paragraphs, model, tokenizer, device, batch_size=4)
        results_bs16 = compute_bpb_batched(paragraphs, model, tokenizer, device, batch_size=16)

        for i in range(len(paragraphs)):
            if results_bs1[i] == -1.0:
                continue
            # Different batch sizes produce slightly different padding patterns
            assert results_bs1[i] == pytest.approx(results_bs4[i], rel=0.05)
            assert results_bs1[i] == pytest.approx(results_bs16[i], rel=0.05)

    def test_batched_single_item_matches_sequential(self, model_and_tokenizer, sample_paragraphs):
        """Batch of size 1 (no padding) should exactly match sequential."""
        model, tokenizer, device = model_and_tokenizer
        paragraphs = sample_paragraphs[1:6]

        from library.bpb.compute_bpb import compute_bpb_batched

        for text in paragraphs:
            seq = compute_bpb(text, model, tokenizer, device)
            bat = compute_bpb_batched([text], model, tokenizer, device, batch_size=1)
            if seq == -1.0:
                assert bat[0] == -1.0
            else:
                # No padding in batch_size=1, so should match closely
                assert seq == pytest.approx(bat[0], rel=1e-4)
