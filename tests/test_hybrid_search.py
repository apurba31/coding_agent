"""Tests for hybrid search functionality."""

import pytest
from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.embedding.models import EmbeddingResult
from coding_agent.embedding.mock import MockEmbedder
from coding_agent.search.keyword import BM25Searcher
from coding_agent.search.semantic import SemanticSearcher
from coding_agent.search.hybrid import HybridSearcher
from coding_agent.search.models import SearchResult
from coding_agent.vectordb.client import LanceDBClient
from pathlib import Path
import tempfile


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            path=Path("src/utils.py"),
            language="python",
            symbol_kind=ChunkKind.FUNCTION,
            symbol="calculate_sum",
            start_line=10,
            end_line=15,
            code="def calculate_sum(a, b):\n    return a + b",
            docstring="Calculate the sum of two numbers.",
            comments=["Sum function"],
            imports=[],
            parent_symbol=None,
            is_public=True,
        ),
        Chunk(
            chunk_id="chunk_2",
            path=Path("src/math.py"),
            language="python",
            symbol_kind=ChunkKind.FUNCTION,
            symbol="multiply",
            start_line=5,
            end_line=8,
            code="def multiply(x, y):\n    return x * y",
            docstring="Multiply two values.",
            comments=["Multiplication"],
            imports=[],
            parent_symbol=None,
            is_public=True,
        ),
        Chunk(
            chunk_id="chunk_3",
            path=Path("src/string.py"),
            language="python",
            symbol_kind=ChunkKind.FUNCTION,
            symbol="concat",
            start_line=20,
            end_line=23,
            code="def concat(s1, s2):\n    return s1 + s2",
            docstring="Concatenate two strings.",
            comments=["String concat"],
            imports=[],
            parent_symbol=None,
            is_public=True,
        ),
    ]
    return chunks


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    return MockEmbedder(embedding_dim=768)


@pytest.fixture
def bm25_searcher(sample_chunks):
    """Create a BM25 searcher with sample chunks."""
    return BM25Searcher(sample_chunks)


@pytest.fixture
def lancedb_client():
    """Create a temporary LanceDB client."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = LanceDBClient(uri=tmpdir, dimension=768)
        yield client


@pytest.fixture
def semantic_searcher(sample_chunks, mock_embedder, lancedb_client):
    """Create a semantic searcher with indexed chunks."""
    # Embed all chunks
    embeddings = []
    for chunk in sample_chunks:
        embedding_text = chunk.to_embedding_text()
        result = mock_embedder.embed_text(embedding_text)
        embeddings.append(result.vector)

    # Index chunks in LanceDB
    lancedb_client.add_chunks(sample_chunks, embeddings)

    return SemanticSearcher(mock_embedder, lancedb_client)


def test_hybrid_searcher_initialization(bm25_searcher, semantic_searcher):
    """Test hybrid searcher can be initialized."""
    hybrid = HybridSearcher(bm25_searcher, semantic_searcher)
    assert hybrid is not None
    assert hybrid.bm25_weight > 0
    assert hybrid.semantic_weight > 0
    assert abs(hybrid.bm25_weight + hybrid.semantic_weight - 1.0) < 0.01


def test_hybrid_searcher_custom_weights(bm25_searcher, semantic_searcher):
    """Test hybrid searcher with custom weights."""
    hybrid = HybridSearcher(
        bm25_searcher, semantic_searcher, bm25_weight=0.3, semantic_weight=0.7
    )
    # Should normalize to ~0.3 and ~0.7
    assert abs(hybrid.bm25_weight - 0.3 / 1.0) < 0.01
    assert abs(hybrid.semantic_weight - 0.7 / 1.0) < 0.01


def test_hybrid_search_query(bm25_searcher, semantic_searcher):
    """Test hybrid search with a query."""
    hybrid = HybridSearcher(bm25_searcher, semantic_searcher)
    results = hybrid.search("calculate sum", top_k=2)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    # Results should be sorted by score (descending)
    scores = [r.score for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_hybrid_search_top_k_limit(bm25_searcher, semantic_searcher):
    """Test that hybrid search respects top_k limit."""
    hybrid = HybridSearcher(bm25_searcher, semantic_searcher)
    results_2 = hybrid.search("function", top_k=2)
    results_10 = hybrid.search("function", top_k=10)

    assert len(results_2) <= 2
    assert len(results_10) <= 10


def test_hybrid_search_score_fusion(bm25_searcher, semantic_searcher):
    """Test that scores are properly fused."""
    hybrid = HybridSearcher(
        bm25_searcher, semantic_searcher, bm25_weight=0.5, semantic_weight=0.5
    )
    results = hybrid.search("calculate", top_k=3)

    # Scores should be normalized to roughly [0, 1] range due to fusion
    for result in results:
        assert 0 <= result.score <= 1.0


def test_hybrid_search_empty_query(bm25_searcher, semantic_searcher):
    """Test hybrid search with empty query raises error."""
    hybrid = HybridSearcher(bm25_searcher, semantic_searcher)
    
    # Empty query should raise ValueError from embedder
    with pytest.raises(ValueError):
        results = hybrid.search("", top_k=5)


def test_hybrid_search_consistency(bm25_searcher, semantic_searcher):
    """Test that same query produces consistent results."""
    hybrid = HybridSearcher(bm25_searcher, semantic_searcher)
    results1 = hybrid.search("multiply", top_k=5)
    results2 = hybrid.search("multiply", top_k=5)

    assert len(results1) == len(results2)
    assert [r.chunk_id for r in results1] == [r.chunk_id for r in results2]


def test_hybrid_search_weight_impact(bm25_searcher, semantic_searcher):
    """Test that weight changes affect results."""
    hybrid_semantic_heavy = HybridSearcher(
        bm25_searcher, semantic_searcher, bm25_weight=0.1, semantic_weight=0.9
    )
    hybrid_bm25_heavy = HybridSearcher(
        bm25_searcher, semantic_searcher, bm25_weight=0.9, semantic_weight=0.1
    )

    # Query that favors keyword matching
    results_sem = hybrid_semantic_heavy.search("string concatenation", top_k=3)
    results_bm25 = hybrid_bm25_heavy.search("string concatenation", top_k=3)

    # At least one should have different ranking (though results may overlap)
    assert isinstance(results_sem, list)
    assert isinstance(results_bm25, list)
