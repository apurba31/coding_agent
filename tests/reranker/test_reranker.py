"""Tests for re-ranking behavior."""

from pathlib import Path

from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.reranker import ScoreFusionReranker
from coding_agent.search.models import SearchResult


def _chunk(chunk_id: str, symbol: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        path=Path(f"src/{symbol}.py"),
        language="python",
        symbol_kind=ChunkKind.FUNCTION,
        symbol=symbol,
        start_line=1,
        end_line=5,
        code=f"def {symbol}():\n    return 1",
        docstring="",
        comments=[],
        imports=[],
        parent_symbol=None,
        is_public=True,
    )


def test_score_fusion_reranker_uses_component_scores_for_priority():
    results = [
        SearchResult(
            chunk=_chunk("a", "alpha"),
            score=0.1,
            chunk_id="a",
            bm25_score=0.8,
            semantic_score=0.2,
        ),
        SearchResult(
            chunk=_chunk("b", "beta"),
            score=0.9,
            chunk_id="b",
            bm25_score=0.1,
            semantic_score=0.95,
        ),
    ]

    reranker = ScoreFusionReranker(alpha=0.7)
    ranked = reranker.rerank("beta", results, top_k=2)

    assert [item.chunk_id for item in ranked] == ["b", "a"]
    assert ranked[0].final_score > ranked[1].final_score
    assert ranked[0].bm25_score == 0.1
    assert ranked[0].semantic_score == 0.95


def test_score_fusion_reranker_respects_top_k_limit():
    results = [
        SearchResult(chunk=_chunk(f"c{i}", f"fn{i}"), score=float(i), chunk_id=f"c{i}")
        for i in range(5)
    ]

    ranked = ScoreFusionReranker().rerank("query", results, top_k=3)

    assert len(ranked) == 3
    assert [item.chunk_id for item in ranked] == ["c4", "c3", "c2"]
