"""
train_m2v_classifier.py - script to train a model2vec classifier on training data

Institutional Books - Enriched Text - 2026
"""

import json
from collections import Counter
from pathlib import Path
from typing import no_type_check

import click
from imblearn.over_sampling import RandomOverSampler  # type: ignore
from loguru import logger
from model2vec.train import StaticModelForClassification
from sklearn.model_selection import train_test_split  # type: ignore

from library.denoise.uniformize import normalize_unicode_in_page


@no_type_check
def load_and_split_json(jsonl_path, test_size=0.2, seed=42):
    """
    Load a JSONL dataset (list of dicts with 'content' and 'target') and split into train/test sets.
    Applies oversampling to the training set to balance classes; the test set is
    left at its natural class distribution so metrics reflect real inference.

    Outputs (X_train, y_train, X_test, y_test).
    """
    logger.info("Loading JSONL...")
    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    # Defensively apply the same light/NFC page normalization the classifiers
    # receive at inference (uniformized_text via normalize_unicode_in_page), so
    # training does not depend on however the training-data content was produced.
    X = [normalize_unicode_in_page(ex["content"]) for ex in data]
    y = [ex["target"] for ex in data]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Oversample the TRAINING set to balance classes. The test set is left at its
    # natural class distribution so evaluation reflects real inference conditions.
    oversampler = RandomOverSampler(random_state=seed)
    train_indices = [[i] for i in range(len(X_train))]
    train_indices_resampled, y_train_resampled = oversampler.fit_resample(train_indices, y_train)
    X_train = [X_train[idx[0]] for idx in train_indices_resampled]
    y_train = y_train_resampled
    logger.info("JSONL loaded.")

    return X_train, y_train, X_test, y_test


@no_type_check
def train_classifier(X_train, y_train, model_name):
    logger.info("Training classifier...")
    classifier = StaticModelForClassification.from_pretrained(model_name=model_name)
    classifier.fit(X_train, y_train, min_epochs=25)
    logger.info("Classifier training done.")
    return classifier


@no_type_check
@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Training JSONL file",
)
@click.option(
    "--model-name", type=str, required=True, help="Directory containing m2v distilled model"
)
@click.option("--test-size", type=float, default=0.2, help="Test set fraction")
@click.option("--seed", type=int, default=42, help="Seed for random generation")
@click.option(
    "--output-dir", type=click.Path(), required=True, help="Directory for trained classifier"
)
def main(input_file: Path, model_name: str, test_size: float, seed: int, output_dir: Path):
    """
    Train classifier from JSON training data.
    """
    logger.info(f"Loading training data from {input_file}")
    X_train, y_train, X_test, y_test = load_and_split_json(
        input_file, test_size=test_size, seed=seed
    )
    logger.info(
        f"Loaded {len(X_train) + len(X_test)} examples. Train: {len(X_train)}, Test: {len(X_test)}"
    )
    dist = Counter(y_train + y_test)
    logger.info(f"Class distribution: {dict(dist)}")

    classifier = train_classifier(X_train, y_train, model_name=model_name)
    results = classifier.evaluate(X_test, y_test)
    logger.info(f"Training results:\n{results}")
    logger.info("Saving trained classifier pipeline...")
    pipeline = classifier.to_pipeline()
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
