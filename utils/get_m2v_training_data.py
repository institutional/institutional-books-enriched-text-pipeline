"""
get_m2v_training_data.py - download model2vec classifier training data

This module will download training data from GitHub releases when made public.
Currently a noop stub that expects data to already be present locally.

Training data location: DATA/release_assets/m2v_training_data/
Expected files:
  - m2v_training_data_topclassifier.jsonl (for fmem_classifier)
  - m2v_training_data_subclassifier.jsonl (for em_subclassifier)

Institutional Books - Enriched Text - 2026
"""

from pathlib import Path

from loguru import logger

# Default paths
DEFAULT_TRAINING_DATA_DIR = Path("DATA/release_assets/m2v_training_data")
TOPCLASSIFIER_FILE = "m2v_training_data_topclassifier.jsonl"
SUBCLASSIFIER_FILE = "m2v_training_data_subclassifier.jsonl"


def get_training_data(
    output_dir: Path = DEFAULT_TRAINING_DATA_DIR,
    force: bool = False,  # noqa: ARG001 - reserved for future download functionality
) -> Path:
    """
    Ensure training data is available locally.

    When made public, this will download training data from the GitHub
    releases page if not already present. Currently a noop that expects
    the data to already exist.

    Args:
        output_dir: Directory to store/check for training data
        force: If True, re-download even if files exist (reserved for future use)

    Returns:
        Path to the training data directory

    Raises:
        FileNotFoundError: If training data is not present
    """
    output_dir = Path(output_dir)

    # Check if training data exists
    topclassifier_path = output_dir / TOPCLASSIFIER_FILE
    subclassifier_path = output_dir / SUBCLASSIFIER_FILE

    missing = []
    if not topclassifier_path.exists():
        missing.append(TOPCLASSIFIER_FILE)
    if not subclassifier_path.exists():
        missing.append(SUBCLASSIFIER_FILE)

    if missing:
        # TODO: When public, download from GitHub releases instead of raising
        # Example future implementation:
        #   download_from_github_release(
        #       repo="institutional/institutional-books-enriched-text-pipeline",
        #       tag="v1.0.0",
        #       assets=missing,
        #       output_dir=output_dir,
        #   )
        raise FileNotFoundError(
            f"Training data not found: {missing}. "
            f"Expected in {output_dir}. "
            "This data will be available from GitHub releases when made public."
        )

    logger.info(f"Training data available at: {output_dir}")
    return output_dir


def get_topclassifier_training_path(
    training_dir: Path = DEFAULT_TRAINING_DATA_DIR,
) -> Path:
    """Get path to top classifier (fmem) training data."""
    return training_dir / TOPCLASSIFIER_FILE


def get_subclassifier_training_path(
    training_dir: Path = DEFAULT_TRAINING_DATA_DIR,
) -> Path:
    """Get path to subclassifier (em) training data."""
    return training_dir / SUBCLASSIFIER_FILE
