"""Sentence Transformers embedding implementation."""

import numpy as np
from typing import Optional

from .embedder import Embedder
from .models import EmbeddingConfig, EmbeddingResult


class SentenceTransformerEmbedder(Embedder):
    """Embedding provider using Sentence Transformers.

    Leverages pre-trained models for semantic understanding of code.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize Sentence Transformers embedder.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        super().__init__(config)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
            cache_folder=self.config.cache_dir,
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with vector.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        vector = self._model.encode(text, convert_to_numpy=True)
        if self.config.normalize:
            vector = self._normalize(vector)

        return EmbeddingResult(
            text=text,
            vector=vector.tolist(),
            dimension=len(vector),
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts efficiently using batch processing.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult objects.

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Filter out empty strings
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            raise ValueError("All texts are empty")

        vectors = self._model.encode(
            non_empty_texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if self.config.normalize:
            vectors = np.array([self._normalize(v) for v in vectors])

        return [
            EmbeddingResult(
                text=text,
                vector=vector.tolist(),
                dimension=len(vector),
            )
            for text, vector in zip(non_empty_texts, vectors)
        ]

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embedding vectors.

        Returns:
            Number of dimensions.
        """
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            Model identifier.
        """
        return self.config.model_name

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector.

        Args:
            vector: NumPy array to normalize.

        Returns:
            Normalized vector.
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
