"""BM25 Keyword Search implementation."""

import re

from rank_bm25 import BM25Plus

from coding_agent.chunker.models import Chunk
from coding_agent.observability import MetricsCollector, get_metrics_collector

from .models import SearchResult


class CodeTokenizer:
    """A sensible code-aware tokenizer."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split code identifiers while retaining original identifier forms."""
        if not text:
            return []

        # Find all words that consist of word characters
        words = re.findall(r"[a-zA-Z0-9_]+", text)

        tokens = set()

        for word in words:
            # Always keep the original identifier
            tokens.add(word)
            tokens.add(word.lower())

            # Split by underscores
            parts_underscore = word.split("_")
            if len(parts_underscore) > 1:
                for part in parts_underscore:
                    if part:
                        tokens.add(part)
                        tokens.add(part.lower())

            # Split CamelCase and PascalCase
            # Match lower-to-upper and acronym-to-word CamelCase boundaries.
            camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", word)
            if len(camel_parts) > 1:
                for part in camel_parts:
                    if part:
                        tokens.add(part)
                        tokens.add(part.lower())

        return list(tokens)


class BM25Searcher:
    """Keyword searcher using rank-bm25."""

    def __init__(
        self,
        chunks: list[Chunk],
        metrics: MetricsCollector | None = None,
    ):
        self.chunks = chunks
        self.metrics = metrics or get_metrics_collector()
        self.tokenized_corpus = [self._tokenize_chunk(chunk) for chunk in chunks]

        if self.tokenized_corpus:
            self.bm25 = BM25Plus(self.tokenized_corpus)
        else:
            self.bm25 = None

    def _tokenize_chunk(self, chunk: Chunk) -> list[str]:
        """Convert a chunk into a list of tokens for the BM25 corpus."""
        corpus_text = [chunk.symbol, chunk.path.name, str(chunk.path), chunk.code]

        if chunk.parent_symbol:
            corpus_text.append(chunk.parent_symbol)

        if chunk.docstring:
            corpus_text.append(chunk.docstring)

        for comment in chunk.comments:
            corpus_text.append(comment)

        for imp in chunk.imports:
            corpus_text.append(imp)

        full_text = " ".join(corpus_text)
        return CodeTokenizer.tokenize(full_text)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Rank documents based on BM25 and return the top_k results."""
        if not self.bm25 or not self.chunks:
            return []

        with self.metrics.measure("search.bm25"):
            tokenized_query = CodeTokenizer.tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            # BM25Plus scores are strictly positive if there is any match
            if score > 0:
                results.append(
                    SearchResult(
                        chunk=self.chunks[i],
                        score=score,
                        chunk_id=self.chunks[i].chunk_id,
                        bm25_score=score,
                        semantic_score=None,
                        final_score=None,
                    )
                )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        self.metrics.observe("search.bm25.results", len(results[:top_k]))
        return results[:top_k]
