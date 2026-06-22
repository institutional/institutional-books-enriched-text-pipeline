"""
setup_pipeline.py - train base models

Samples books per language from shards and trains Ngram and Nupunkt models for
later use in the pipeline. Also checks model2vec distilled models and
classifiers (and optionally distills another model).

In a standard workflow, this should be run after `prepare_shards.py`.

Institutional Books - Enriched Text - 2026
"""

import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import click
import torch
from loguru import logger
from tqdm import tqdm

from const.config import PipelineConfig, load_config
from const.languages import is_nupunkt_language
from const.types import BookJSON, BooksByLangDict, HardNormText, RawPage
from library.denoise.ngrams import build_ngram_stats, save_ngram_stats
from library.denoise.uniformize import normalize_text, normalize_unicode_in_page
from utils.jsonl_io import iter_jsonl


def stream_books_from_shards(shard_dir: Path) -> Iterator[BookJSON]:
    """
    Stream book records from all shard files in a directory.

    Handles both .jsonl and .jsonl.gz files.
    """
    # Find all shard files (both compressed and uncompressed)
    shard_paths = sorted(set(shard_dir.glob("*.jsonl")) | set(shard_dir.glob("*.jsonl.gz")))
    for shard_path in tqdm(shard_paths):
        yield from iter_jsonl(shard_path)


def sample_books_by_language(shard_dir: Path, max_books_per_language: int = 30) -> BooksByLangDict:
    """
    Sample books from shards, grouped by language.

    Returns a dictionary mapping language codes to lists of BookJSONs.
    """
    books_by_lang: BooksByLangDict = defaultdict(list)
    for book in stream_books_from_shards(shard_dir):
        lang = book.get("language_gen", "")
        if not lang:
            continue
        if len(books_by_lang[lang]) < max_books_per_language:
            books_by_lang[lang].append(book)
    return dict(books_by_lang)


def extract_pages_from_book(book: BookJSON) -> list[RawPage]:
    pages = book.get("text_by_page_src", [])
    if not pages:
        logger.warning(f"No pages detected in book {book.get('barcode_src', 'UNKNOWN')}.")
    return pages


def soft_normalize_page_for_nupunkt(raw_page: RawPage) -> str:
    """
    Normalize a page the way the Nupunkt segmenter sees text at inference time.

    Mirrors step-1 uniformization (light/NFC normalization via
    normalize_unicode_in_page) followed by the segmenter's newline flattening
    (cf. segment_book_nupunkt in library/segment/nupunkt_segmenter.py), so the
    Nupunkt training corpus uses the same normalization as middlematter at
    segmentation -- NOT the hard/NFKC normalization used for the n-gram models.
    """
    soft_page = normalize_unicode_in_page(raw_page)
    return re.sub(r" +", " ", soft_page.replace("\n", " ")).strip()


def build_nupunkt_corpus_from_books(
    books: list[BookJSON],
    output_path: Path,
) -> int:
    """
    Build the Nupunkt training corpus (one soft-normalized page per line).

    Uses soft (light/NFC) normalization to match the normalization applied to
    middlematter at segmentation time, keeping training and inference consistent.

    Returns the total number of nonempty pages in this corpus.
    """
    # TODO: Validate output and require overwrite permission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_written = 0
    with open(output_path, "w", encoding="utf8") as out:
        for book in books:
            pages = extract_pages_from_book(book)
            for page in pages:
                if not page:
                    continue
                page_text = soft_normalize_page_for_nupunkt(page)
                if not page_text:  # e.g. if page only had whitespace
                    continue
                out.write(page_text + "\n")
                lines_written += 1

    # TODO: log appropriate statistics

    return lines_written


def build_ngram_stats_for_language(
    books: list[BookJSON],
    output_path: Path,
    max_n: int = 5,
) -> int:
    """
    Build ngram statistics from book records and save to disk.

    Returns the total number of nonempty pages processed.
    """
    # TODO: Validate output and require overwrite permission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_processed = 0

    corpus_parts: list[HardNormText] = []
    for book in books:
        pages = extract_pages_from_book(book)
        for page in pages:
            if not page:
                continue
            page_tokens = normalize_text(page)
            if not page_tokens:
                continue
            corpus_parts.append(page_tokens)
            pages_processed += 1
    # Build ngram stats
    corpus = "\n".join(corpus_parts)
    corpus = HardNormText(corpus)
    stats = build_ngram_stats(corpus, max_n=max_n)
    save_ngram_stats(stats, output_path)

    # TODO: log appropriate statistics

    return pages_processed


def train_nupunkt(
    input_corpus: Path,
    output_path: Path,
    batch_size: int = 100000,
    prune_freq: int = 10000,
    min_type_freq: int = 3,
) -> None:
    """
    Train Nupunkt sentence segmentation model.
    """
    # TODO: Validate output and require overwrite permission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training Nupunkt model: {output_path}")

    cmd = [
        "nupunkt",
        "train",
        str(input_corpus),
        "--batch-size",
        str(batch_size),
        "--prune-freq",
        str(prune_freq),
        "--min-type-freq",
        str(min_type_freq),
        "-o",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.debug(f"nupunkt: {line}")
        logger.info(f"Nupunkt model complete: {output_path}")

    except FileNotFoundError:
        logger.error("nupunkt command not found. Is Nupunkt installed?")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Nupunkt training failed: {e}")
        raise


def distill_model2vec(
    output_dir: Path,
    model_name: str = "BAAI/bge-m3",
    dim: int = 512,
) -> Path:
    """
    Distill a model2vec embedding model.

    Args:
        output_dir: Directory to save distilled model
        model_name: Base model to distill
        dim: Output embedding dimension

    Returns:
        Path to distilled model directory
    """
    from model2vec.distill import distill

    full_name = model_name.replace("/", "_") + f"_m2v_{dim}dim"
    save_dir = output_dir / full_name

    if save_dir.exists():
        logger.info(f"Distilled model already exists: {save_dir}")
        return save_dir

    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Distilling model '{model_name}' to {dim} dimensions...")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.__version__ < "2.8"  # Apple silicon compat problems
    ):
        device = "mps"

    logger.info(f"Using device: {device}")
    distilled_model = distill(model_name=model_name, pca_dims=dim, device=device)
    distilled_model.save_pretrained(str(save_dir))

    logger.info(f"Distilled model saved to: {save_dir}")
    return save_dir


def train_m2v_classifier(
    training_file: Path,
    model_dir: Path,
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Train a model2vec classifier.

    Args:
        training_file: Path to JSONL training data
        model_dir: Path to distilled model2vec model
        output_dir: Directory to save trained classifier
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
    """
    from utils.train_m2v_classifier import load_and_split_json, train_classifier

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training classifier: {output_dir}")

    # Load and split data
    X_train, y_train, X_test, y_test = load_and_split_json(
        training_file, test_size=test_size, seed=seed
    )
    logger.info(
        f"Loaded {len(X_train) + len(X_test)} examples. Train: {len(X_train)}, Test: {len(X_test)}"
    )

    # Train classifier
    classifier = train_classifier(X_train, y_train, model_name=str(model_dir))

    # Evaluate and save
    results = classifier.evaluate(X_test, y_test)
    logger.info(f"Training results:\n{results}")

    logger.info("Saving trained classifier pipeline...")
    pipeline = classifier.to_pipeline()
    pipeline.save_pretrained(str(output_dir))
    logger.info(f"Classifier training complete: {output_dir}")


def train_endmatter_classifiers(
    output_dir: Path,
    model_name: str = "BAAI/bge-m3",
    model_dim: int = 512,
    overwrite: bool = False,
    m2v_training_data_dir: Path | None = None,
) -> None:
    """
    Train the endmatter classifiers (mmem_classifier and em_subclassifier).

    Requires:
        - Distilled model2vec model at output_dir/<model_name>_m2v_<dim>dim
        - Training data at m2v_training_data_dir (or default location)

    Args:
        output_dir: Directory containing distilled model and for classifier output
        model_name: Base model name used for distillation
        model_dim: Dimension used for distillation
        overwrite: If True, retrain even if classifiers exist
        m2v_training_data_dir: Directory containing training data files
    """
    from utils.get_m2v_training_data import (
        get_subclassifier_training_path,
        get_topclassifier_training_path,
        get_training_data,
    )

    # Check training data is available
    try:
        training_dir = get_training_data(output_dir=m2v_training_data_dir) if m2v_training_data_dir else get_training_data()
    except FileNotFoundError as e:
        logger.warning(f"Skipping classifier training: {e}")
        return

    # Build paths
    full_model_name = model_name.replace("/", "_") + f"_m2v_{model_dim}dim"
    distilled_model_dir = output_dir / full_model_name

    if not distilled_model_dir.exists():
        logger.error(f"Distilled model not found: {distilled_model_dir}")
        logger.error("Cannot train classifiers without distilled model.")
        return

    # Train mmem_classifier (top classifier for frontmatter/endmatter detection)
    mmem_classifier_dir = output_dir / "mmem_classifier"
    if mmem_classifier_dir.exists() and not overwrite:
        logger.info(f"mmem_classifier exists, skipping: {mmem_classifier_dir}")
    else:
        logger.info("Training mmem_classifier (frontmatter/endmatter top classifier)...")
        try:
            train_m2v_classifier(
                training_file=get_topclassifier_training_path(training_dir),
                model_dir=distilled_model_dir,
                output_dir=mmem_classifier_dir,
            )
        except Exception as e:
            logger.error(f"mmem_classifier training failed: {e}")

    # Train em_subclassifier (endmatter subtype classifier)
    em_subclassifier_dir = output_dir / "em_subclassifier"
    if em_subclassifier_dir.exists() and not overwrite:
        logger.info(f"em_subclassifier exists, skipping: {em_subclassifier_dir}")
    else:
        logger.info("Training em_subclassifier (endmatter subtype classifier)...")
        try:
            train_m2v_classifier(
                training_file=get_subclassifier_training_path(training_dir),
                model_dir=distilled_model_dir,
                output_dir=em_subclassifier_dir,
            )
        except Exception as e:
            logger.error(f"em_subclassifier training failed: {e}")


def setup_pipeline(
    shard_dir: Path,
    output_dir: Path,
    max_books_per_language: int = 30,
    ngram_order: int = 5,
    distill_model: bool = True,
    model_name: str = "BAAI/bge-m3",
    model_dim: int = 512,
    overwrite: bool = False,
    m2v_training_data_dir: Path | None = None,
) -> None:
    """
    Train all Ngram and Nupunkt LMs from shards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Sampling up to {max_books_per_language} books per language...")
    books_by_lang = sample_books_by_language(shard_dir, max_books_per_language)
    logger.info(f"Found {len(books_by_lang)} languages")
    for lang, books in sorted(books_by_lang.items()):
        logger.debug(f"  {lang}: {len(books)} books")

    languages_processed = 0
    ngram_models_built = 0
    nupunkt_models_trained = 0

    # Train models for each language
    for lang, books in sorted(books_by_lang.items()):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing language: {lang} ({len(books)} books)")
        logger.info(f"{'=' * 60}")

        lang_stats: dict[str, Any]
        lang_stats = {"books_sampled": len(books)}

        # ngram model
        ngram_model = output_dir / f"{lang}_ngram.json.gz"
        if ngram_model.exists() and not overwrite:
            logger.info(f"Ngram stats exist, skipping: {ngram_model}")
        else:
            logger.info(f"Building ngram stats: {ngram_model}")
            try:
                pages = build_ngram_stats_for_language(books, ngram_model, max_n=ngram_order)
                ngram_models_built += 1
                lang_stats["ngram_pages"] = pages
                lang_stats["ngram_built"] = True
            except Exception as e:
                logger.error(f"N-gram stats building failed for {lang}: {e}")
                lang_stats["ngram_error"] = str(e)

        # Build and train Nupunkt if applicable
        if is_nupunkt_language(lang):
            nupunkt_corpus = output_dir / f"{lang}_nupunkt.txt"
            nupunkt_model = output_dir / f"{lang}_nupunkt.bin"

            if nupunkt_model.exists() and not overwrite:
                logger.info(f"Nupunkt model exists, skipping: {nupunkt_model}")
            else:
                logger.info(f"Building Nupunkt corpus: {nupunkt_corpus}")
                lines = build_nupunkt_corpus_from_books(books, nupunkt_corpus)
                lang_stats["nupunkt_corpus_lines"] = lines

                logger.info("Training Nupunkt model...")
                try:
                    train_nupunkt(nupunkt_corpus, nupunkt_model)
                    nupunkt_models_trained += 1
                    lang_stats["nupunkt_trained"] = True
                except Exception as e:
                    logger.error(f"Nupunkt training failed for {lang}: {e}")
                    lang_stats["nupunkt_error"] = str(e)
        else:
            logger.info("Skipping Nupunkt (language not in NUPUNKT_LANGUAGES)")

        languages_processed += 1
        logger.debug(lang_stats)

    logger.info(f"{languages_processed=}")
    logger.info(f"{ngram_models_built=}")
    logger.info(f"{nupunkt_models_trained=}")

    # Distill model2vec and train classifiers
    if distill_model:
        logger.info(f"\n{'=' * 60}")
        logger.info("Distilling model2vec model")
        logger.info(f"{'=' * 60}")

        try:
            distill_model2vec(output_dir, model_name, model_dim)
        except Exception as e:
            logger.error(f"Model distillation failed: {e}")

        # Train endmatter classifiers after distillation
        logger.info(f"\n{'=' * 60}")
        logger.info("Training endmatter classifiers")
        logger.info(f"{'=' * 60}")

        train_endmatter_classifiers(
            output_dir=output_dir,
            model_name=model_name,
            model_dim=model_dim,
            overwrite=overwrite,
            m2v_training_data_dir=m2v_training_data_dir,
        )

    logger.info("\nSetup pipeline complete!")


################################################################################


@click.command()
@click.option(
    "--shard-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing raw shard files (from prepare_shards)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for trained models",
)
@click.option(
    "--max-books",
    type=int,
    default=20,
    help="Maximum books to sample per language (default: 20)",
)
@click.option(
    "--ngram-order",
    type=int,
    default=5,
    help="N-gram order (default: 5)",
)
@click.option(
    "--no-distill",
    is_flag=True,
    default=False,
    help="Skip model2vec distillation",
)
@click.option(
    "--model-name",
    type=str,
    default="BAAI/bge-m3",
    help="Base model for distillation (default: BAAI/bge-m3)",
)
@click.option(
    "--model-dim",
    type=int,
    default=512,
    help="Output dimension for distillation (default: 512)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing models",
)
@click.option(
    "--m2v-training-data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing m2v training data (default from config)",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional config file (YAML)",
)
def main(
    shard_dir: Path,
    output_dir: Path,
    max_books: int,
    ngram_order: int,
    no_distill: bool,
    model_name: str,
    model_dim: int,
    overwrite: bool,
    m2v_training_data_dir: Path | None,
    config_file: Path | None,
):
    """
    Train language models from prepared shards.

    Reads raw shard files created by prepare_shards, samples books per
    language, and trains Ngram and Nupunkt models for the processing
    pipeline. Also distills a model2vec model for chunking.

    Example:
        python commands/setup_pipeline.py \\
            --shard-dir ./DATA/shards/raw \\
            --output-dir ./DATA/models
    """
    config = load_config(config_file) if config_file else PipelineConfig()

    # Use CLI arg if provided, otherwise fall back to config
    training_data_dir = m2v_training_data_dir or config.model_paths.m2v_training_data_dir

    click.echo(f"Setting up pipeline from: {shard_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Max books per language: {max_books}")

    setup_pipeline(
        shard_dir=shard_dir,
        output_dir=output_dir,
        max_books_per_language=max_books,
        ngram_order=ngram_order,
        distill_model=not no_distill,
        model_name=model_name,
        model_dim=model_dim,
        overwrite=overwrite,
        m2v_training_data_dir=training_data_dir,
    )


if __name__ == "__main__":
    main()
