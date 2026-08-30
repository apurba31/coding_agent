"""Data models for embedding operations."""

from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    cache_dir: str | None = None


@dataclass
class EmbeddingResult:
    """Result of embedding a single text."""

    text: str
    vector: list[float]
    dimension: int

    @property
    def norm(self) -> float:
        """Calculate L2 norm of vector."""
        import math
        return math.sqrt(sum(x**2 for x in self.vector))
