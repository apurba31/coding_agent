"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod

from .models import EmbeddingConfig, EmbeddingResult


class Embedder(ABC):
    """Abstract base class for embedding providers.

    The rest of the system depends on this interface, not on any
    specific embedding implementation (e.g., Sentence Transformers).
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize embedder.

        Args:
            config: Embedding configuration. Defaults to EmbeddingConfig().
        """
        self.config = config or EmbeddingConfig()

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult containing vector and metadata.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If embedding fails.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts efficiently.

        Implementations should leverage batch processing where available.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If embedding fails.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embedding vectors.

        Returns:
            Number of dimensions in the embedding space.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model.

        Returns:
            Model name/identifier.
        """
        raise NotImplementedError
