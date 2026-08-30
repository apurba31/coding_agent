"""Semantic search implementation."""

from typing import List

from coding_agent.embedding.embedder import Embedder
from coding_agent.vectordb.client import LanceDBClient
from coding_agent.vectordb.schema import record_to_chunk
from .models import SearchResult


class SemanticSearcher:
    """Semantic searcher using an Embedder and LanceDB vector store."""

    def __init__(self, embedder: Embedder, db_client: LanceDBClient):
        self.embedder = embedder
        self.db_client = db_client

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        """Rank documents based on semantic vector similarity and return the top_k results.
        
        Args:
            query: The search query string.
            top_k: Number of results to return.
            
        Returns:
            List of SearchResult objects containing the chunk and distance score.
        """
        # 1. Embed Query
        query_result = self.embedder.embed_text(query)
        
        # 2. Vector Search
        records = self.db_client.search(query_result.vector, limit=top_k)
        
        results = []
        for record in records:
            chunk = record_to_chunk(record)
            # LanceDB returns distance as `_distance`.
            # We preserve this score. Note that for LanceDB, a lower _distance implies higher similarity.
            # In Phase 10 (Hybrid Fusion), we will normalize this.
            score = record.get("_distance", 0.0)
            
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                chunk_id=chunk.chunk_id
            ))
            
        return results

