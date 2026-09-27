"""Reusable reranking abstraction for search result refinement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from coding_agent.search.models import SearchResult


class Reranker(ABC):
    """Base class for ranking search results after retrieval."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return a reordered subset of results for a query."""
        raise NotImplementedError
