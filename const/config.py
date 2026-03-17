"""
config.py - config management
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


class ChunkingAlgorithm(str, Enum):
    TEXTTILING = "texttiling"
    C99 = "c99"


@dataclass
class ChunkingConfig:
    """Chunking step configuration."""

    algorithm: ChunkingAlgorithm = ChunkingAlgorithm.TEXTTILING


@dataclass
class PerplexityFilterConfig:
    """Perplexity filtering configuration."""

    enabled: bool = False


@dataclass
class PipelineConfig:
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    perplexity_filter: PerplexityFilterConfig = field(default_factory=PerplexityFilterConfig)

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
        )
        chunking = ChunkingConfig(
            algorithm=ChunkingAlgorithm(data.get("chunking", {}).get("algorithm", "texttiling")),
        )
        perplexity_filter = PerplexityFilterConfig(
            enabled=data.get("perplexity_filter", {}).get("enabled", False),
        )
        return cls(
            model_paths=model_paths,
            chunking=chunking,
            perplexity_filter=perplexity_filter,
        )


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return PipelineConfig.from_dict(data or {})
