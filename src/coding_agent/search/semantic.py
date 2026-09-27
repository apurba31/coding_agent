"""Semantic search implementation."""

from coding_agent.embedding.embedder import Embedder
from coding_agent.observability import MetricsCollector, get_metrics_collector
from coding_agent.vectordb.client import LanceDBClient
from coding_agent.vectordb.schema import record_to_chunk

from .models import SearchResult


class SemanticSearcher:
    """Semantic searcher using an Embedder and LanceDB vector store."""

    def __init__(
        self,
        embedder: Embedder,
        db_client: LanceDBClient,
        metrics: MetricsCollector | None = None,
    ):
        self.embedder = embedder
        self.db_client = db_client
        self.metrics = metrics or getattr(db_client, "metrics", None) or get_metrics_collector()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Rank documents based on semantic vector similarity and return the top_k results.
        
        Args:
            query: The search query string.
            top_k: Number of results to return.
            
        Returns:
            List of SearchResult objects containing the chunk and distance score.
        """
        # 1. Embed Query
        with self.metrics.measure("embedding.query"):
            query_result = self.embedder.embed_text(query)
        
        # 2. Vector Search
        records = self.db_client.search(query_result.vector, limit=top_k)
        
        results = []
        for record in records:
            chunk = record_to_chunk(record)
            # LanceDB returns distance as `_distance`.
            # Lower LanceDB distance implies higher similarity; fusion normalizes it later.
            score = record.get("_distance", 0.0)

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    chunk_id=chunk.chunk_id,
                    bm25_score=None,
                    semantic_score=score,
                    final_score=None,
                )
            )
        self.metrics.observe("search.semantic.results", len(results))
        return results

