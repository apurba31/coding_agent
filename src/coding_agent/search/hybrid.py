"""Hybrid search implementation combining BM25 and semantic search."""


from coding_agent.observability import MetricsCollector

from .keyword import BM25Searcher
from .models import SearchResult
from .semantic import SemanticSearcher


class HybridSearcher:
    """Combines BM25 keyword search and semantic vector search with score fusion."""

    def __init__(
        self,
        bm25_searcher: BM25Searcher,
        semantic_searcher: SemanticSearcher,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
        metrics: MetricsCollector | None = None,
    ):
        """Initialize hybrid searcher.

        Args:
            bm25_searcher: BM25 keyword searcher instance.
            semantic_searcher: Semantic vector searcher instance.
            bm25_weight: Weight for BM25 scores in fusion (0-1).
            semantic_weight: Weight for semantic scores in fusion (0-1).
        """
        self.bm25_searcher = bm25_searcher
        self.semantic_searcher = semantic_searcher
        self.metrics = metrics or bm25_searcher.metrics
        
        # Normalize weights
        total = bm25_weight + semantic_weight
        self.bm25_weight = bm25_weight / total if total > 0 else 0.5
        self.semantic_weight = semantic_weight / total if total > 0 else 0.5

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Perform hybrid search combining BM25 and semantic results.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects ranked by fused score.
        """
        with self.metrics.measure("search.hybrid"):
            return self._search(query, top_k)

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        """Fuse independently measured lexical and vector search results."""
        metrics = self.metrics
        bm25_results = self.bm25_searcher.search(query, top_k=top_k * 2)
        semantic_results = self.semantic_searcher.search(query, top_k=top_k * 2)

        # 2. Build a map of chunk_id -> SearchResult and normalize scores
        result_map: dict[str, SearchResult] = {}

        # Normalize BM25 scores (max score -> 1.0)
        bm25_max_score = max([r.score for r in bm25_results], default=1.0)
        if bm25_max_score == 0:
            bm25_max_score = 1.0

        for result in bm25_results:
            normalized_score = result.score / bm25_max_score
            fused_score = normalized_score * self.bm25_weight
            result_map[result.chunk_id] = SearchResult(
                chunk=result.chunk,
                score=fused_score,
                chunk_id=result.chunk_id,
                bm25_score=normalized_score,
                semantic_score=0.0,
                final_score=fused_score,
            )

        # Normalize semantic scores (min distance -> max similarity, so invert & normalize)
        # LanceDB returns _distance where lower = more similar
        semantic_min_score = min([r.score for r in semantic_results], default=0.0)
        semantic_max_score = max([r.score for r in semantic_results], default=1.0)
        
        # Invert: similarity = 1 - (normalized_distance)
        score_range = semantic_max_score - semantic_min_score
        if score_range == 0:
            score_range = 1.0

        for result in semantic_results:
            # Normalize distance to [0, 1], then invert to similarity
            normalized_distance = (result.score - semantic_min_score) / score_range
            similarity = 1.0 - normalized_distance
            normalized_score = similarity
            fused_score = normalized_score * self.semantic_weight

            chunk_id = result.chunk_id
            if chunk_id in result_map:
                # Chunk appears in both results: fuse scores
                prior = result_map[chunk_id]
                final_score = prior.final_score + fused_score
                result_map[chunk_id] = SearchResult(
                    chunk=prior.chunk,
                    score=final_score,
                    chunk_id=chunk_id,
                    bm25_score=prior.bm25_score if prior.bm25_score is not None else 0.0,
                    semantic_score=max(
                        prior.semantic_score if prior.semantic_score is not None else 0.0,
                        normalized_score,
                    ),
                    final_score=final_score,
                )
            else:
                # Chunk only in semantic results
                result_map[chunk_id] = SearchResult(
                    chunk=result.chunk,
                    score=fused_score,
                    chunk_id=chunk_id,
                    bm25_score=0.0,
                    semantic_score=normalized_score,
                    final_score=fused_score,
                )

        # 3. Sort by fused score (descending) and return top_k
        sorted_results = sorted(
            result_map.values(),
            key=lambda r: r.score,
            reverse=True,
        )

        results = sorted_results[:top_k]
        metrics.observe("search.hybrid.results", len(results))
        return results
