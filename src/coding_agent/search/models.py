"""Data models for search functionality."""

from dataclasses import dataclass

from coding_agent.chunker.models import Chunk


@dataclass
class SearchResult:
    """Represents a single search result and its ranking metadata."""

    chunk: Chunk
    score: float
    chunk_id: str
    bm25_score: float | None = None
    semantic_score: float | None = None
    final_score: float | None = None
