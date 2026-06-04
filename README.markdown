# Institutional Books Enriched Text Pipeline #

The Institutional Data Initiative's pipeline for cleaning and optimizing the
Institutional Books 1.0 collection for text enrichment.


## Work in Progress ##

The current version of this repository is a **Work in Progress.**


## Installation ##

    # for polyglot
    sudo apt-get install libicu-dev pkg-config

    # everything else
    uv sync --all-extras

We suppose there is a basic dev environment with compilers, etc.


## Configuration ##

Create a `config.yaml` file to customize pipeline settings:

```yaml
# Paths to trained models (from setup_pipeline)
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

# Perplexity computation (GPU recommended)
perplexity:
  enabled: false
  model_name: Qwen/Qwen3-0.6B-Base

# Compress output shard files with gzip
use_gzip: false
```

All options shown above are defaults and can be omitted if unchanged.


## About IDI ##

The Institutional Data Initiative at Harvard Law School Library works with
knowledge institutions—from libraries and museums to cultural groups and
government agencies—to refine and publish their collections as data.
[Reach out to collaborate on your collections](https://institutionaldatainitiative.org/#get-involved).
