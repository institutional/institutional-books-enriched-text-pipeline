
# Scripts

Helper scripts for development, testing, analysis, and data preparation.

## Deduplication Analysis

### analyze_clusters.py

Analyze duplicate clusters from `dedup_find_duplicates` output. Streams
`clusters.json` to avoid loading the full file into memory. Produces summary
statistics and detailed breakdowns.

```bash
python scripts/analyze_clusters.py DATA/Cluster/clusters.json
```

### find_large_clusters.py

Find and display the largest duplicate clusters by member count.

```bash
python scripts/find_large_clusters.py
```

### show_cluster.py

Print the paragraphs in a specific duplicate cluster. Given a cluster ID (e.g.
`32044098672728:587`), looks it up in `clusters.json`, retrieves paragraph text
from parquet files, and prints all members.

```bash
python scripts/show_cluster.py CLUSTER_ID
```

### sample_duplicate_paragraphs.py

Sample duplicate clusters across different size categories and display actual
paragraph text side by side for manual inspection.

```bash
python scripts/sample_duplicate_paragraphs.py
```

### inspect_skipped_buckets.py

Inspect skipped buckets from `dedup_find_duplicates`. Decodes doc indices from
the skipped buckets JSONL into human-readable `book_id:paragraph_idx` form.

```bash
python scripts/inspect_skipped_buckets.py
```

## Data Indexing and Conversion

### build_barcode_index.py

Build a barcode-to-parquet-file index for fast book lookups. Reads only the
`barcode_src` column from each parquet file and writes a compact JSON mapping.

```bash
python scripts/build_barcode_index.py \
    --parquet-dir DATA/Cluster/parquet_shards \
    --output DATA/Cluster/barcode_index.json
```

### extract_book_metadata.py

Extract per-book metadata (barcode, language, paragraph count) from parquet
shards for use in cluster analysis.

```bash
python scripts/extract_book_metadata.py \
    --parquet-dir DATA/Cluster/parquet_shards \
    --output DATA/Cluster/book_metadata.jsonl
```

### parquet_to_jsonl.py

Convert parquet shard files to JSONL, matching shard assignments from existing
perplexity files.

```bash
python scripts/parquet_to_jsonl.py \
    --parquet-dir DATA/Cluster/parquet_shards \
    --perplexity-dir DATA/Cluster/perplexity \
    --output-dir DATA/Cluster/processed_shards
```

## Sampling and Training Data Preparation

These scripts form a pipeline for sampling books by language, extracting
paragraphs, and preparing train/test splits for LLM training experiments.

### Pipeline overview

```
sample_books_by_language.py
        │
        ▼
  sampled_books.json
        │
        ├──► extract_final_sampled.py ──► FINAL.jsonl
        │                                     │
        │                                     ▼
        │                          extract_paragraphs.py ──► paragraphs.jsonl
        │                                                         │
        ├──► train_test_split.py ──► train_book_ids.json          │
        │                           test_book_ids.json            │
        │                                  │                      │
        │                                  ▼                      │
        │                          split_paragraphs.py ──► train.jsonl
        │                                                  test.jsonl
        │
        └──► extract_sampled_books.py ──► shard0001_nupunkt.jsonl
                                          shard0001_sat.jsonl
                                          ...
```

### sample_books_by_language.py

Sample up to N books per language from the full dataset. Joins barcode/language
info from parquet shards with shard numbers from perplexity files. Excludes
unknown languages.

```bash
python scripts/sample_books_by_language.py \
    --parquet-dir DATA/Cluster/parquet_shards \
    --perplexity-dir DATA/Cluster/perplexity \
    --output sampled_books.json \
    --n 30 \
    --seed 42
```

Output: JSON array of `{"book_id", "language", "shard_number"}` objects.

### extract_final_sampled.py

Pull sampled books out of the final processed JSONL shards into a single file.
Looks up each book by shard number and barcode.

```bash
python scripts/extract_final_sampled.py \
    --sampled-books sampled_books.json \
    --final-dir DATA/Cluster/final \
    --output FINAL.jsonl
```

### extract_sampled_books.py

Extract sampled books from raw (unprocessed) JSONL shards into new shards
separated by segmenter type (nupunkt vs sat). Intended for re-processing
experiments on a different system.

```bash
python scripts/extract_sampled_books.py \
    --sampled-books sampled_books.json \
    --source-dir DATA/shards/raw \
    --output-dir DATA/sampled_shards \
    --shard-size 10
```

Output: `shard0001_nupunkt.jsonl`, `shard0002_nupunkt.jsonl`, ...,
`shard0001_sat.jsonl`, etc. Each shard has at most `--shard-size` books.

### extract_paragraphs.py

Extract individual paragraphs from `annotated_middlematter` in processed books.
Strips all HTML tags and produces one JSONL line per paragraph with a unique
`doc_id` of the form `book_id:PAR_NUMBER`.

```bash
python scripts/extract_paragraphs.py \
    --input FINAL.jsonl \
    --output paragraphs.jsonl \
    --filter-perplexity \
    --filter-duplicates
```

Flags:
- `--filter-perplexity` — omit paragraphs with perplexity outside the p10–p90
  range (per-book thresholds from metadata)
- `--filter-duplicates` — omit paragraphs marked as duplicates (`<aside
  data-cluster=...>` tags)

Output: JSONL with `{"book_id", "doc_id", "text"}`.

### train_test_split.py

Stratified train/test split of book IDs by language. Each language contributes
at least 1 book to the test set.

```bash
python scripts/train_test_split.py \
    --sampled-books sampled_books.json \
    --train-output train_book_ids.json \
    --test-output test_book_ids.json \
    --test-fraction 0.1 \
    --seed 42
```

Output: Two JSON files, each a flat list of book ID strings.

### split_paragraphs.py

Split a paragraphs JSONL file into train and test sets using the book ID lists
from `train_test_split.py`.

```bash
python scripts/split_paragraphs.py \
    --input paragraphs.jsonl \
    --train-ids train_book_ids.json \
    --test-ids test_book_ids.json \
    --train-output train.jsonl \
    --test-output test.jsonl
```

Output: Two JSONL files with the same `{"book_id", "doc_id", "text"}` schema.
