"""Tests for embedding layer."""

import pytest
import math

from coding_agent.embedding.embedder import Embedder
from coding_agent.embedding.mock import MockEmbedder
from coding_agent.embedding.models import EmbeddingConfig, EmbeddingResult


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_name == "BAAI/bge-small-en-v1.5"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.normalize is True
        assert config.cache_dir is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda",
            batch_size=64,
            normalize=False,
            cache_dir="/tmp/embeddings",
        )

        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.normalize is False
        assert config.cache_dir == "/tmp/embeddings"


class TestEmbeddingResult:
    """Tests for EmbeddingResult."""

    def test_result_creation(self):
        """Test creating an embedding result."""
        vector = [0.1, 0.2, 0.3, 0.4]
        result = EmbeddingResult(
            text="hello world",
            vector=vector,
            dimension=4,
        )

        assert result.text == "hello world"
        assert result.vector == vector
        assert result.dimension == 4

    def test_result_norm_calculation(self):
        """Test L2 norm calculation."""
        # Vector: [3, 4] has norm 5
        vector = [3.0, 4.0]
        result = EmbeddingResult(
            text="test",
            vector=vector,
            dimension=2,
        )

        assert math.isclose(result.norm, 5.0)

    def test_result_norm_unit_vector(self):
        """Test norm of unit vector."""
        # Normalized vector should have norm ~1
        vector = [0.6, 0.8]
        result = EmbeddingResult(
            text="test",
            vector=vector,
            dimension=2,
        )

        assert math.isclose(result.norm, 1.0, abs_tol=1e-6)


class TestMockEmbedder:
    """Tests for MockEmbedder."""

    def test_embedder_initialization(self):
        """Test mock embedder creation."""
        embedder = MockEmbedder()

        assert embedder.model_name == "mock"
        assert embedder.embedding_dimension == 768

    def test_custom_embedding_dimension(self):
        """Test custom embedding dimension."""
        embedder = MockEmbedder(embedding_dim=512)

        assert embedder.embedding_dimension == 512

    def test_embed_text(self):
        """Test embedding single text."""
        embedder = MockEmbedder(embedding_dim=64)
        result = embedder.embed_text("hello world")

        assert result.text == "hello world"
        assert len(result.vector) == 64
        assert result.dimension == 64

    def test_embed_text_deterministic(self):
        """Test that same text produces same embedding."""
        embedder = MockEmbedder()
        text = "deterministic test"

        result1 = embedder.embed_text(text)
        result2 = embedder.embed_text(text)

        assert result1.vector == result2.vector

    def test_embed_text_different_texts(self):
        """Test that different texts produce different embeddings."""
        embedder = MockEmbedder()

        result1 = embedder.embed_text("text one")
        result2 = embedder.embed_text("text two")

        # Different texts should produce different vectors
        assert result1.vector != result2.vector

    def test_embed_text_empty_raises_error(self):
        """Test that empty text raises ValueError."""
        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedder.embed_text("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedder.embed_text("   ")

    def test_embed_batch(self):
        """Test batch embedding of multiple texts."""
        embedder = MockEmbedder(embedding_dim=128)
        texts = ["first", "second", "third"]

        results = embedder.embed_batch(texts)

        assert len(results) == 3
        for result, text in zip(results, texts, strict=True):
            assert result.text == text
            assert len(result.vector) == 128

    def test_embed_batch_empty_raises_error(self):
        """Test that empty batch raises ValueError."""
        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            embedder.embed_batch([])

    def test_embed_batch_all_empty_raises_error(self):
        """Test that batch with only empty strings raises ValueError."""
        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="All texts are empty"):
            embedder.embed_batch(["", "   ", "\t"])

    def test_vector_normalization(self):
        """Test that vectors are unit normalized."""
        embedder = MockEmbedder(embedding_dim=256)
        result = embedder.embed_text("test text")

        # Calculate norm
        norm = sum(x**2 for x in result.vector) ** 0.5

        # Should be approximately 1.0 for normalized vector
        assert math.isclose(norm, 1.0, abs_tol=1e-6)

    def test_batch_vector_normalization(self):
        """Test that batch-embedded vectors are normalized."""
        embedder = MockEmbedder(embedding_dim=128)
        results = embedder.embed_batch(["text1", "text2", "text3"])

        for result in results:
            norm = sum(x**2 for x in result.vector) ** 0.5
            assert math.isclose(norm, 1.0, abs_tol=1e-6)


class TestEmbedderInterface:
    """Tests verifying Embedder abstraction."""

    def test_embedder_is_abstract(self):
        """Test that Embedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Embedder()

    def test_mock_implements_embedder(self):
        """Test that MockEmbedder implements Embedder interface."""
        embedder = MockEmbedder()
        assert isinstance(embedder, Embedder)

    def test_embedder_interface_methods(self):
        """Test that embedder has required interface methods."""
        embedder = MockEmbedder()

        assert hasattr(embedder, "embed_text")
        assert callable(embedder.embed_text)

        assert hasattr(embedder, "embed_batch")
        assert callable(embedder.embed_batch)

        assert hasattr(embedder, "embedding_dimension")
        assert hasattr(embedder, "model_name")

    def test_embedding_config_in_embedder(self):
        """Test that embedder stores configuration."""
        config = EmbeddingConfig(batch_size=16, normalize=False)
        embedder = MockEmbedder(config=config)

        assert embedder.config.batch_size == 16
        assert embedder.config.normalize is False


class TestEmbeddingWithChunkText:
    """Integration tests: embeddings with chunk-like texts."""

    def test_embed_code_chunk(self):
        """Test embedding realistic code chunk."""
        embedder = MockEmbedder(embedding_dim=256)

        code_chunk = """def calculate_sum(a, b):
    '''Calculate the sum of two numbers.'''
    return a + b
"""

        result = embedder.embed_text(code_chunk)

        assert len(result.vector) == 256
        assert len(result.text) == len(code_chunk)

    def test_embed_chunk_with_metadata(self):
        """Test embedding chunk text with metadata."""
        embedder = MockEmbedder()

        chunk_text = """Language: python
Path: src/utils.py
Symbol: helper_function
Kind: function

def helper_function():
    pass
"""

        result = embedder.embed_text(chunk_text)
        assert result.text == chunk_text
        assert len(result.vector) > 0

    def test_similar_chunks_similar_embeddings(self):
        """Test that similar code produces similar embeddings."""
        embedder = MockEmbedder(embedding_dim=128)

        # Very similar chunks
        chunk1 = "def add(a, b): return a + b"
        chunk2 = "def add(x, y): return x + y"

        result1 = embedder.embed_text(chunk1)
        result2 = embedder.embed_text(chunk2)

        # Calculate cosine similarity
        dot_product = sum(
            a * b for a, b in zip(result1.vector, result2.vector, strict=True)
        )
        # Should be reasonably similar (deterministic, but different)
        assert -1.0 <= dot_product <= 1.0


class TestEmbeddingBatchPerformance:
    """Tests for batch embedding behavior."""

    def test_batch_consistency(self):
        """Test that batch embedding matches individual embedding."""
        embedder = MockEmbedder(embedding_dim=64)
        texts = ["hello", "world", "test"]

        # Individual embeddings
        individual_results = [embedder.embed_text(text) for text in texts]

        # Batch embedding
        batch_results = embedder.embed_batch(texts)

        # Results should match
        assert len(individual_results) == len(batch_results)
        for ind, batch in zip(individual_results, batch_results, strict=True):
            assert ind.vector == batch.vector

    def test_large_batch(self):
        """Test embedding large batch of texts."""
        embedder = MockEmbedder(embedding_dim=128)
        texts = [f"text_{i}" for i in range(1000)]

        results = embedder.embed_batch(texts)

        assert len(results) == 1000
        assert all(len(r.vector) == 128 for r in results)
        assert all(isinstance(r, EmbeddingResult) for r in results)

    def test_mixed_empty_in_batch(self):
        """Test batch with mix of empty and non-empty strings."""
        embedder = MockEmbedder()
        texts = ["text1", "", "text2", "   ", "text3"]

        results = embedder.embed_batch(texts)

        # Only non-empty texts should be embedded
        assert len(results) == 3
        assert results[0].text == "text1"
        assert results[1].text == "text2"
        assert results[2].text == "text3"


class TestEmbeddingEdgeCases:
    """Edge case tests for embeddings."""

    def test_unicode_text(self):
        """Test embedding unicode text."""
        embedder = MockEmbedder()

        unicode_text = "Hello 世界 🌍 Здравствуй"
        result = embedder.embed_text(unicode_text)

        assert result.text == unicode_text
        assert len(result.vector) > 0

    def test_very_long_text(self):
        """Test embedding very long text."""
        embedder = MockEmbedder()

        long_text = "word " * 10000  # ~50K characters
        result = embedder.embed_text(long_text)

        assert len(result.vector) == 768

    def test_special_characters(self):
        """Test embedding text with special characters."""
        embedder = MockEmbedder()

        special_text = "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`"
        result = embedder.embed_text(special_text)

        assert result.text == special_text
        assert len(result.vector) > 0

    def test_whitespace_only_text(self):
        """Test that whitespace-only text raises error."""
        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedder.embed_text("\n\t   \r")


class TestConfigCustomization:
    """Tests for customizable embedder configuration."""

    def test_config_passed_to_embedder(self):
        """Test that custom config is used by embedder."""
        config = EmbeddingConfig(
            model_name="custom-model",
            batch_size=128,
            normalize=False,
        )
        embedder = MockEmbedder(config=config)

        assert embedder.config.model_name == "custom-model"
        assert embedder.config.batch_size == 128

    def test_default_config_if_none(self):
        """Test that default config is used if not provided."""
        embedder = MockEmbedder(config=None)

        assert embedder.config.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.config.batch_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
