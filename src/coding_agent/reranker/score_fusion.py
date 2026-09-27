"""Simple score-based reranking that fuses BM25 and semantic signals."""

from __future__ import annotations

from dataclasses import replace

from coding_agent.search.models import SearchResult

from .reranker import Reranker


class ScoreFusionReranker(Reranker):
    """Fuse stored component scores into a final ranking."""

    def __init__(self, alpha: float = 0.7) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Sort results by a weighted score fusion of the component signals."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        scored = []
        for result in results:
            bm25_score = float(result.bm25_score if result.bm25_score is not None else 0.0)
            semantic_score = float(
                result.semantic_score if result.semantic_score is not None else 0.0
            )
            if bm25_score == 0.0 and semantic_score == 0.0:
                final_score = float(result.score)
            else:
                final_score = self.alpha * semantic_score + (1.0 - self.alpha) * bm25_score

            scored.append(
                replace(
                    result,
                    score=final_score,
                    final_score=final_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                )
            )

        ranked = sorted(scored, key=lambda item: item.final_score, reverse=True)
        return ranked[:top_k]
