"""
steps.py - a registry of steps

This is the canonical order or processing steps.
"""

from typing import Optional

MAIN_STEPS = [
    "step01_uniformize_unicode",
    "step02_remove_duplicate_pages",
    "step03_separate_frontmatter_backmatter",
    "step04_dehyphenate",
    "step05_headerfooter_removal",
    "step06_pagenumber_removal",
    # "step07_validate_segmenter",
    # "step08_segment",
    # "step9_stray_number_removal",
    # "step10_chunk",
    # "step11_compute_perplexity",
]

PERPLEXITY_STEPS = [
    # "step11_compute_perplexity",
]

DEDUP_STEPS = [
    # "step12_deduplicate"
]

POST_STEPS = [
    # "step13_annotate",
    # "step14_add_metadata",
    # "step15_clean",
]

STEP_ORDER = MAIN_STEPS + PERPLEXITY_STEPS + DEDUP_STEPS + POST_STEPS


def validate_step_name(step_name: str) -> bool:
    return step_name in STEP_ORDER


def get_step_index(step_name: str) -> int:
    """Get the index of a step in STEP_ORDER."""
    try:
        return STEP_ORDER.index(step_name)
    except ValueError:
        raise ValueError(f"Unknown step: {step_name}. Valid steps: {STEP_ORDER}")


def get_step_range(
    start_step: Optional[str] = None,
    end_step: Optional[str] = None,
    step_list: list[str] = MAIN_STEPS,
) -> list[str]:
    """
    Get a range of steps from start to end (inclusive).

    Returns a list of step names in the range.
    """
    if start_step is None:
        start_idx = 0
    else:
        if start_step not in step_list:
            raise ValueError(f"Start step '{start_step}' not in step list")
        start_idx = step_list.index(start_step)

    if end_step is None:
        end_idx = len(step_list)
    else:
        if end_step not in step_list:
            raise ValueError(f"End step '{end_step}' not in step list")
        end_idx = step_list.index(end_step) + 1  # remember to be inclusive!

    return step_list[start_idx:end_idx]
