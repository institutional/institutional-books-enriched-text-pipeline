# 📚 Institutional Books - Enriched Text Pipeline

The Institutional Data Initiative's pipeline for producing **Institutional Books -  Enriched Text** datasets.

This pipeline produces a configurable, annotated version of the OCRed text instead of a single
"cleaned" stream of tokens. It separates endmatter from the main text, segments volumes into
sentences and subtopic paragraphs, detects per-paragraph language, identifies collection-wide
duplicate clusters, and (optionally) scores every paragraph by bits-per-byte.

This pipeline comes with a library of cleaning, chunking, and annotating utilities.

- 🤗 [Institutional Books Collection on HuggingFace](https://huggingface.co/collections/institutional/institutional-books)
- 🤗 [IB-HL-ET on HuggingFace](https://huggingface.co/datasets/institutional/institutional-books-hl-enriched-text)
- 🧰 [IB-ET parser library](https://github.com/institutional/institutional-books-enriched-text-parser)
- 📄 [IB-HL-ET Technical Report](soon to appear)
- 🌐 [Website](https://institutional.org/)

---

## Summary
- [Getting started](#getting-started)
- [Pipeline overview](#pipeline-overview)
- [Configuration](#configuration)
- [CLI: `prepare-shards`](#cli-prepare-shards)
- [CLI: `setup-pipeline`](#cli-setup-pipeline)
- [CLI: `process-shard`](#cli-process-shard)
- [CLI: `compute-bpb`](#cli-compute-bpb)
- [CLI: Deduplication](#cli-deduplication)
- [CLI: `postprocess-shard`](#cli-postprocess-shard)
- [Output format](#output-format)
- [Development](#development)
- [About IDI](#about-idi)
- [Cite](#cite)

---

## Getting started

### Machine-level dependencies
- [uv](https://docs.astral.sh/uv/)
- A C++ compiler and toolchain (used to build the optional simhash extension)
- ICU development headers, required by `polyglot`/`pyicu`:

```bash
# Debian/Ubuntu
sudo apt-get install libicu-dev pkg-config
```

### Step by step setup
```bash
# Clone project
git clone https://github.com/institutional/institutional-books-enriched-text-pipeline.git
cd institutional-books-enriched-text-pipeline

# Install dependencies
uv sync --all-extras

# The C++ extensions (fast simhash and C99) are compiled automatically by
# uv sync --all-extras above. For slightly different configurations, they
# can also be manually built in place.
uv run python setup.py build_ext --inplace

# Run commands
uv run pipeline.py --help
uv run pipeline.py prepare-shards --help
```

All pipeline commands are subcommands of `pipeline.py` and support a `--help` flag listing available
options. Each is also runnable directly as a module (e.g. `python -m commands.process_shard`), and
installing the package provides an equivalent `ibet-pipeline` console script.

[☝️ Back to summary](#summary)

---

## Pipeline overview

The pipeline is a sequence of text-to-text transformations. We organize, annotate, and analyze
existing text. Books are processed shard by shard. Each shard is a JSONL file of a configurable
number of books (see `--shard-size`), partitioned by language into **Nupunkt** shards (languages
with English-like punctuation) and
**SaT** shards (everything else). Most stages share no state across shards and can be executed in
parallel without additional configuration.

The full pipeline runs these commands in order. The `#` column numbers the commands; the
internal step numbers (1–15) referenced in the sections below are the finer-grained processing
steps that these commands run (e.g. `process-shard` runs steps 1–10, deduplication is step 11,
BPB is step 12, and `postprocess-shard` runs steps 13–15).

| # | Command | Parallelism | Purpose |
| --- | --- | --- | --- |
| 1 | `prepare-shards` | single process | Download books from HuggingFace and partition into shards |
| 2 | `setup-pipeline` | single process | Train n-gram and Nupunkt models from sampled books |
| 3 | `process-shard` | per shard | Main processing steps (steps 1–10) |
| 4 | `dedup-compute-simhashes` | per shard | Simhashes for every paragraph |
| 5 | `dedup-find-duplicates` | single process | Cluster near-duplicate paragraphs across all shards |
| 6 | `dedup-build-lookup` | single process | Invert clusters into per-shard lookup sidecars |
| 7 | `dedup-annotate` | per shard | Mark duplicate/representative paragraphs in books |
| 8 | `compute-bpb` | per shard (GPU recommended) | Per-paragraph bits-per-byte scores (optional) |
| 9 | `postprocess-shard` | per shard | Post-processing steps, producing final output |

### Main processing steps

Run via `process-shard`, implemented in `commands/steps/`:

1. `uniformize_unicode` - Normalize Unicode, quotes, hyphens, whitespace
2. `remove_duplicate_pages` - Remove exact duplicate pages within a book
3. `separate_frontmatter_backmatter` - Split into frontmatter/middlematter/backmatter using a trained classifier
4. `dehyphenate` - Rejoin hyphenated words across line breaks
5. `headerfooter_removal` - Remove running headers and footers
6. `pagenumber_removal` - Remove page numbers
7. `validate_segmenter` - Confirm segmenter choice (Nupunkt vs SaT)
8. `segment` - Split text into sentences
9. `remove_stray_numbers` - Clean up orphaned numbers
10. `chunk` - Detect subtopic paragraphs and sections using TextTiling or C99

### Deduplication steps

This is `step 11`, but consists of substeps. Some (but not all) can be done in parallel.

1. `dedup-compute-simhashes` - compute simhashes for every shard (parallelizable)
2. `dedup-find-duplicates` - gather all computed simhashes and detect duplicate clusters (single process)
3. `dedup-build-lookup` - create per-shard data files for fast annotation (single process)
4. `dedup-annotate` - modify shards with duplication data (parallelizable)

### (Optional) BPB step

This is step 12: `compute-bpb` will compute per-paragraph bits-per-byte scores.

### Post-processing steps

Run via `postprocess-shard`:

13. `annotate` - Build annotated XML-like strings with semantic tags
14. `add_metadata` - Compute text statistics (tokens, n-grams, bits-per-byte stats)
15. `clean` - Remove intermediate fields and produce final output

[☝️ Back to summary](#summary)

---

## Configuration

Commands that accept `--config-file` read a YAML file to customize pipeline settings:

```yaml
# Paths to trained models (from setup-pipeline)
model_paths:
  ngram: ./DATA/pretrain/models
  nupunkt: ./DATA/pretrain/models
  embedding: ./DATA/pretrain/models/BAAI_bge-m3_m2v_512dim
  mmem_classifier: ./DATA/pretrain/models/mmem_classifier
  em_subclassifier: ./DATA/pretrain/models/em_subclassifier
  m2v_training_data_dir: ./DATA/release_assets/m2v_training_data

# Chunking algorithm: "texttiling" or "c99"
chunking:
  algorithm: texttiling

# Sentence segmentation
segment:
  sat_model_name: sat-3l-sm

# Bits-per-byte computation (GPU recommended)
bpb:
  model_name: Qwen/Qwen3-0.6B-Base

# Compress output shard files with gzip
use_gzip: false
```

All options shown above are defaults and can be omitted if unchanged.

[☝️ Back to summary](#summary)

---

## CLI: `prepare-shards`

Downloads books from the HuggingFace dataset and partitions them into shards based on Nupunkt
compatibility. Supports interruption and resumption via a progress file.

```bash
uv run pipeline.py prepare-shards \
    --output-dir DATA/shards \
    --shard-size 1000
```

| Option name | Description |
| --- | --- |
| `--output-dir` | Output directory for shards (default: `./DATA/shards`) |
| `--shard-size` | Number of books per shard (default: 1000) |
| `--dataset` | HuggingFace dataset identifier (default: `institutional/institutional-books-1.0`) |
| `--split` | Dataset split to use (default: `train`) |
| `--max-books` | Maximum number of books to process (default: no limit) |
| `--gzip/--no-gzip` | Compress shard files with gzip (default: no compression) |

[☝️ Back to summary](#summary)

---

## CLI: `setup-pipeline`

Samples books per language from the raw shards and trains the n-gram and Nupunkt models used by
later steps. Also checks the model2vec distilled embedding model and the endmatter classifiers,
optionally distilling a new embedding model. The synthetically generated training data for the
endmatter classifier and subclassifier is published as a GitHub release on this repository and is
downloaded automatically when needed.

```bash
uv run pipeline.py setup-pipeline \
    --shard-dir DATA/shards/raw \
    --output-dir DATA/pretrain/models
```

| Option name | Description |
| --- | --- |
| `--shard-dir` | Directory containing raw shard files, from `prepare-shards` (required) |
| `--output-dir` | Directory for trained models (required) |
| `--max-books` | Maximum books to sample per language (default: 20) |
| `--ngram-order` | N-gram order (default: 5) |
| `--no-distill` | Skip model2vec distillation |
| `--model-name` | Base model for distillation (default: `BAAI/bge-m3`) |
| `--model-dim` | Output dimension for distillation (default: 512) |
| `--overwrite` | Overwrite existing models |
| `--m2v-training-data-dir` | Directory containing model2vec training data (default from config) |
| `--config-file` | Optional config file (YAML) |

[☝️ Back to summary](#summary)

---

## CLI: `process-shard`

Runs the main processing steps (1–10) on a single shard. Shards are independent, so many instances can run in parallel.

```bash
uv run pipeline.py process-shard \
    --shard-id 0001 \
    --segmenter nupunkt \
    --input-dir DATA/shards/raw \
    --output-dir DATA/shards/processed
```

| Option name | Description |
| --- | --- |
| `--shard-id` | Shard identifier, e.g. `0001` (required) |
| `--segmenter` | Segmenter for this shard: `nupunkt` or `sat` (required) |
| `--input-dir` | Input directory containing raw shards (default: `./DATA/shards/raw`) |
| `--output-dir` | Output directory for processed shards (default: `./DATA/shards/processed`) |
| `--log-dir` | Log directory (default: `./DATA/logs/shards`) |
| `--config-file` | Pipeline configuration file (YAML) |
| `--start-step` | First step to run (default: `step01_uniformize_unicode`) |
| `--end-step` | Last step to run (default: last of main steps) |
| `--resume` | Resume from previous progress (skip already-processed books) |
| `--reprocess-incomplete` | Reprocess books from an incomplete JSONL file (ignores `--shard-id`, `--input-dir`) |

[☝️ Back to summary](#summary)

---

## CLI: `compute-bpb`

Computes per-paragraph bits-per-byte scores using a causal language model, writing `*.bpb.jsonl`
sidecar files consumed by `postprocess-shard`. Designed to run on GPU nodes, possibly separately
from the main pipeline.

```bash
uv run pipeline.py compute-bpb \
    --input-file DATA/shards/processed/shard0001.complete.jsonl \
    --output-file DATA/bpb/shard0001.bpb.jsonl
```

| Option name | Description |
| --- | --- |
| `--input-file` | Input JSONL file with chunked books, after step 10 (required) |
| `--output-file` | Output `.bpb.jsonl` file (required) |
| `--config-file` | Optional config file (YAML) |
| `--resume` | Resume from previous progress (skip already-processed books) |
| `--batch-size` | Batch size for paragraph inference, tune per GPU memory (default: 16) |

[☝️ Back to summary](#summary)

---

## CLI: Deduplication

Cross-book paragraph deduplication runs as four commands. The first and last are per-shard and
parallelizable; the middle two run once over all shards. Duplicates are *identified but not
removed*: each cluster of near-duplicate paragraphs gets a representative, and the other members are
annotated as duplicates, preserving uninterrupted reading context in every volume.

### `dedup-compute-simhashes`

Computes a 128-bit simhash for every paragraph in a shard, writing `*.simhashes.jsonl` files. Uses
the C++ extension when available, with an identical pure-Python fallback.

```bash
uv run pipeline.py dedup-compute-simhashes \
    --input-file DATA/shards/processed/shard0001.complete.jsonl \
    --output-file DATA/dedup/simhashes/shard0001.simhashes.jsonl
```

| Option name | Description |
| --- | --- |
| `--input-file` | Input JSONL file with chunked books (required) |
| `--output-file` | Output JSONL file for simhash records (required) |
| `--ngram-size` | N-gram size for simhash computation (default: 9) |

### `dedup-find-duplicates`

Finds near-duplicate paragraph clusters across all shards by Hamming distance over the simhashes.
Single process; needs all `*.simhashes.jsonl` files.

```bash
uv run pipeline.py dedup-find-duplicates \
    --input-dir DATA/dedup/simhashes \
    --output-file DATA/dedup/clusters.json
```

| Option name | Description |
| --- | --- |
| `--input-dir` | Directory containing `*.simhashes.jsonl` files (required) |
| `--output-file` | Output JSON file for cluster information (required) |
| `--threshold` | Hamming distance threshold for duplicates (default: 5) |
| `--workers` | Number of parallel workers (default: number of CPUs) |
| `--benchmark` | Print timing breakdown for each phase |
| `--temp-dir` | Directory for temp files; not cleaned up if specified |

### `dedup-build-lookup`

Inverts `clusters.json` once into per-shard `*.lookup.jsonl` sidecars, enabling parallel annotation
without rereading cluster file.

```bash
uv run pipeline.py dedup-build-lookup \
    --shard-dir DATA/shards/processed \
    --clusters-file DATA/dedup/clusters.json \
    --output-dir DATA/dedup/lookups
```

| Option name | Description |
| --- | --- |
| `--shard-dir` | Directory of `*.complete.jsonl` shard files (required) |
| `--clusters-file` | Clusters JSON file from `dedup-find-duplicates` (required) |
| `--output-dir` | Directory to write per-shard `<stem>.lookup.jsonl` sidecars (required) |
| `--shard-glob` | Glob for shard files within `--shard-dir` (default: `*.complete.jsonl`) |
| `--workers` | Parallel workers for the barcode-indexing phase |

### `dedup-annotate`

Annotates each book with duplicate-paragraph and representative-paragraph information from its shard's lookup file.

```bash
uv run pipeline.py dedup-annotate \
    --shard-file DATA/shards/processed/shard0001.complete.jsonl \
    --lookup-file DATA/dedup/lookups/shard0001.lookup.jsonl
```

| Option name | Description |
| --- | --- |
| `--shard-file` | Shard JSONL file to annotate - will be overwritten (required) |
| `--lookup-file` | Per-shard lookup sidecar from `dedup-build-lookup` (required) |

An additional command, `dedup-mark-removed`, marks paragraphs from specified cluster IDs for removal (`--shard-file`, `--removal-file`).

[☝️ Back to summary](#summary)

---

## CLI: `postprocess-shard`

Runs post-processing steps 13–15 on a deduplicated shard: builds annotated text, computes metadata, and strips intermediate fields.

```bash
uv run pipeline.py postprocess-shard \
    --input-file DATA/shards/deduped/shard0001.deduped.jsonl \
    --output-file DATA/shards/final/shard0001.final.jsonl \
    --bpb-file DATA/bpb/shard0001.bpb.jsonl
```

| Option name | Description |
| --- | --- |
| `--input-file` | Input JSONL file, after deduplication (required) |
| `--output-file` | Output JSONL file for final books (required) |
| `--bpb-file` | Optional `.bpb.jsonl` file with bits-per-byte values |
| `--config-file` | Pipeline configuration file (YAML) |
| `--start-step` | First step to run (default: `step13_annotate`) |
| `--end-step` | Last step to run (default: `step15_clean`) |
| `--keep-sentences/--no-keep-sentences` | Also retain the raw `middlematter_sentences` list in the final output (default: drop) |
| `--keep-indices/--no-keep-indices` | Also retain the `subtopic_paragraph_start_indices`/`subtopic_section_start_indices` arrays in the final output (default: drop) |
| `--resume` | Resume from previous progress (skip already-processed books) |

[☝️ Back to summary](#summary)

---

## Output format

The final output is one JSON object per book, per line. Each book retains a small set of fields:

| Field | Description |
| --- | --- |
| `barcode_src` | Book identifier from the source dataset |
| `language_gen` | Primary language, taken from IB-HL (published as `primary_language_gen`) |
| `annotated_frontmatter` | Frontmatter as an annotated HTML-like string |
| `annotated_middlematter` | Main text as an annotated HTML-like string, with section, paragraph, and duplicate annotations |
| `annotated_backmatter` | Backmatter as an annotated HTML-like string |
| `metadata` | Text statistics: token count, word count, n-gram and bits-per-byte statistics, etc. |

Intermediate fields produced during processing (page lists, sentence lists, structural indices,
dedup maps) are removed by step 15 (`clean`). The document text and its subtopic/section structure
are already encoded in the annotated strings, so the raw sentence list and paragraph/section index
arrays are dropped by default; pass `--keep-sentences` and/or `--keep-indices` to
`postprocess-shard` to retain `middlematter_sentences`, `subtopic_paragraph_start_indices`, and
`subtopic_section_start_indices` alongside the annotated output. In the published HuggingFace
dataset (built via `scripts/build_hf_parquet.py`), the annotated fields appear as `frontmatter_gen`,
`middlematter_gen`, and `backmatter_gen`, alongside per-book statistics and an opinionated
`processed_middlematter_gen` convenience column.

### Annotated strings

The annotated fields are HTML-escaped strings assembled from a small set of HTML-like tags, chosen
so that annotation (tags and attributes) stays separate from primary content (inner text): stripping
all tags and keeping only the inner text recovers the volume text.

In the endmatter, each non-empty page becomes one `<div>` classed by its detected type: `toc_index`,
`biblio`, or `otherendmatter`. The middlematter is grouped into subtopic `<section>` elements
containing subtopic `<p>` paragraphs, annotated with per-paragraph bits-per-byte (`data-bpb`),
detected language (`data-language`, ISO-639-3), and duplicate-cluster information: the
representative of a duplicate cluster carries `data-representative` and `data-clusterid`, and
duplicated passages are wrapped in `<aside data-cluster="...">`:

```html
<section data-bpb="0.7311">
  <p data-bpb="0.7189" data-language="eng"
     data-representative data-clusterid="ABCDEF:15">
    ... source text for a duplicate class ...
  </p>
  <p data-bpb="0.7433" data-language="eng">
    ... paragraph text ...
  </p>
  <aside data-cluster="GHIJKL:21">
    <p data-bpb="0.7311" data-language="eng">
      ... duplicate text ...
    </p>
  </aside>
</section>
```

These attributes support filtering and structured processing without a fixed editorial choice: for
example, long-context English prose can be obtained by dropping endmatter and filtering on
`data-language`, while other use cases can retain the layers they need. A small, dependency-free
[parser library](https://github.com/institutional/institutional-books-enriched-text-parser) is
available for parsing and adapting the output to specific use cases.

[☝️ Back to summary](#summary)

---

## Development

```bash
# Run all tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run black .

# Rebuild C++ extensions
uv run python setup.py build_ext --inplace
```

Repository layout:

- `commands/` - CLI commands and pipeline step implementations (`commands/steps/`)
- `library/` - Core processing logic (denoise, segment, chunk, deduplicate, bpb, annotate, metadata)
- `const/` - Configuration, types, and language tables
- `utils/` - Shared utilities (simhash, union-find, atomic writes)
- `extensions/simhash_cpp/` - pybind11 C++ extension for fast simhash
- `tests/` - Test suite

A few additional scripts are in `scripts` that facilitate working with raw data from the pipeline.

[☝️ Back to summary](#summary)

---

## About IDI

The Institutional Data Initiative at Harvard Law School Library works with knowledge institutions,
from libraries and museums to cultural groups and government agencies, to refine and publish their
collections as data.
[Reach out to collaborate on your collections](https://institutional.org/#get-involved).

[☝️ Back to summary](#summary)

---

## Cite

```bibtext
@misc{lowryduda2026institutionalbooksenriched,
      title={Institutional Books - Enriched Text: A customizable multilingual open-source pipeline for denoising, deduplicating, and annotating OCR text at scale}, 
      author={David Lowry-Duda and Matteo Cargnelutti and Catherine Brobston and Salwa Ismail and Greg Leppert and Amanda Watson and Jonathan Zittrain},
      year={2026},
      eprint={2608.19026},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.19026}, 
}
```

[☝️ Back to summary](#summary)
