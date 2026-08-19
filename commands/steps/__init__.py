"""
commands/steps - step registry and individual step commands.

This package contains:
- The canonical ordering of processing steps (MAIN_STEPS, POST_STEPS)
- Individual step command modules (step01_*, step02_*, ..., step15_*)

Institutional Books - Enriched Text - 2026
"""

from typing import Optional

MAIN_STEPS = [
    "step01_uniformize_unicode",
    "step02_remove_duplicate_pages",
    "step03_separate_frontmatter_backmatter",
    "step04_dehyphenate",
    "step05_headerfooter_removal",
    "step06_pagenumber_removal",
    "step07_validate_segmenter",
    "step08_segment",
    "step09_remove_stray_numbers",
    "step10_chunk",
]

# BPB computation is handled by a separate CLI (GPU-intensive):
#   compute_bpb.py - compute per-paragraph bits-per-byte → *.bpb.jsonl

# Deduplication is handled by separate CLIs.

# Post processing is handled separately:
POST_STEPS = [
    "step13_annotate",
    "step14_add_metadata",
    "step15_clean",
]


def validate_step_name(step_name: str) -> bool:
    return step_name in MAIN_STEPS


def get_step_index(step_name: str) -> int:
    """Get the index of a step in MAIN_STEPS."""
    try:
        return MAIN_STEPS.index(step_name)
    except ValueError:
        raise ValueError(f"Unknown step: {step_name}. Valid steps: {MAIN_STEPS}")


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


def get_remaining_steps(
    last_completed_step: str,
    end_step: Optional[str] = None,
    step_list: list[str] = MAIN_STEPS,
) -> list[str]:
    """
    Get the remaining steps after a given completed step.

    Args:
        last_completed_step: The last step that completed successfully
        end_step: Optional final step (inclusive)
        step_list: List of steps to use

    Returns:
        List of step names to run (starting from the step AFTER last_completed_step)
    """
    if last_completed_step == "step0_prestart":
        # Nothing completed yet, run all steps
        start_idx = 0
    elif last_completed_step not in step_list:
        raise ValueError(f"Unknown step: {last_completed_step}. Valid steps: {step_list}")
    else:
        start_idx = step_list.index(last_completed_step) + 1

    if start_idx >= len(step_list):
        # All steps already completed
        return []

    if end_step is None:
        end_idx = len(step_list)
    else:
        if end_step not in step_list:
            raise ValueError(f"End step '{end_step}' not in step list")
        end_idx = step_list.index(end_step) + 1

    return step_list[start_idx:end_idx]
