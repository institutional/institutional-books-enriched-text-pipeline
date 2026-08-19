"""
config.py - config management

Institutional Books - Enriched Text - 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelPaths:
    """Paths to model directories."""

    ngram: Path = Path("./DATA/pretrain/models")
    nupunkt: Path = Path("./DATA/pretrain/models")
    embedding: Path = Path("./DATA/pretrain/models/BAAI_bge-m3_m2v_512dim")
    mmem_classifier: Path = Path("./DATA/pretrain/models/mmem_classifier")
    em_subclassifier: Path = Path("./DATA/pretrain/models/em_subclassifier")
    m2v_training_data_dir: Path = Path("./DATA/release_assets/endmatter_training_data")


class ChunkingAlgorithm(str, Enum):
    TEXTTILING = "texttiling"
    C99 = "c99"


@dataclass
class ChunkingConfig:
    """Chunking step configuration."""

    algorithm: ChunkingAlgorithm = ChunkingAlgorithm.TEXTTILING


@dataclass
class SegmentConfig:
    """Segmentation configuration."""

    sat_model_name: str = "sat-3l-sm"


@dataclass
class BPBConfig:
    """Bits-per-byte computation configuration."""

    model_name: str = "Qwen/Qwen3-0.6B-Base"


@dataclass
class PipelineConfig:
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    bpb: BPBConfig = field(default_factory=BPBConfig)
    use_gzip: bool = False  # Compress shard files with gzip

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        model_paths = ModelPaths(
            ngram=Path(data.get("model_paths", {}).get("ngram", "./DATA/pretrain/models")),
            nupunkt=Path(data.get("model_paths", {}).get("nupunkt", "./DATA/pretrain/models")),
            embedding=Path(
                data.get("model_paths", {}).get(
                    "embedding", "./DATA/pretrain/models/BAAI_bge-m3_m2v_512dim"
                )
            ),
            mmem_classifier=Path(
                data.get("model_paths", {}).get(
                    "mmem_classifier", "./DATA/pretrain/models/mmem_classifier"
                )
            ),
            em_subclassifier=Path(
                data.get("model_paths", {}).get(
                    "em_subclassifier", "./DATA/pretrain/models/em_subclassifier"
                )
            ),
            m2v_training_data_dir=Path(
                data.get("model_paths", {}).get(
                    "m2v_training_data_dir", "./DATA/release_assets/endmatter_training_data"
                )
            ),
        )
        chunking = ChunkingConfig(
            algorithm=ChunkingAlgorithm(data.get("chunking", {}).get("algorithm", "texttiling")),
        )
        segment = SegmentConfig(
            sat_model_name=data.get("segment", {}).get("sat_model_name", "sat-3l-sm"),
        )
        bpb = BPBConfig(
            model_name=data.get("bpb", {}).get("model_name", "Qwen/Qwen3-0.6B-Base"),
        )
        return cls(
            model_paths=model_paths,
            chunking=chunking,
            segment=segment,
            bpb=bpb,
            use_gzip=data.get("use_gzip", False),
        )


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return PipelineConfig.from_dict(data or {})
