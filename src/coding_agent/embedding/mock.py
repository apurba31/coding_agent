"""Mock embedder for testing without loading models."""

import hashlib
from typing import Optional

from .embedder import Embedder
from .models import EmbeddingConfig, EmbeddingResult


class MockEmbedder(Embedder):
    """Mock embedder for testing purposes.

    Generates deterministic but meaningful embeddings without loading
    actual ML models. Useful for fast unit tests.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        embedding_dim: int = 768,
    ):
        """Initialize mock embedder.

        Args:
            config: Embedding configuration.
            embedding_dim: Dimension of mock embeddings.
        """
        super().__init__(config)
        self._embedding_dim = embedding_dim

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate deterministic mock embedding for text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with mock vector.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        vector = self._generate_deterministic_vector(text)
        return EmbeddingResult(
            text=text,
            vector=vector,
            dimension=len(vector),
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate mock embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult objects.

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            raise ValueError("All texts are empty")

        return [self.embed_text(text) for text in non_empty_texts]

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of mock embeddings.

        Returns:
            Fixed dimension.
        """
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return the mock model name.

        Returns:
            Model identifier.
        """
        return "mock"

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Generate deterministic vector from text hash.

        Ensures same text always produces same embedding, but different
        texts produce different embeddings.

        Args:
            text: Input text.

        Returns:
            Deterministic vector of fixed dimension.
        """
        hash_val = hashlib.sha256(text.encode()).hexdigest()
        vector = []

        for i in range(self._embedding_dim):
            # Use different parts of hash for each dimension
            hash_chunk = hash_val[(i * 2) % len(hash_val) : (i * 2 + 2) % len(hash_val)]
            if not hash_chunk:
                hash_chunk = hash_val[: min(2, len(hash_val))]

            # Convert hex to float in range [-1, 1]
            val = int(hash_chunk, 16) / 256.0
            vector.append(val - 0.5)  # Center around 0

        # Normalize to unit length
        norm = sum(x**2 for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector
