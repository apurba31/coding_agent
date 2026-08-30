"""Data models for search functionality."""

from dataclasses import dataclass
from coding_agent.chunker.models import Chunk

@dataclass
class SearchResult:
    """Represents a single search result with score."""
    chunk: Chunk
    score: float
    chunk_id: str
