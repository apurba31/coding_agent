# Retrieval

Retrieval combines lexical and semantic signals, then optionally reranks.

## Hybrid search

`HybridSearcher` runs BM25 and vector search independently, min-max normalizes scores, fuses with configurable weights, deduplicates by `chunk_id`, and returns top candidates.

## Reranking

When `MINI_AGENT_USE_RERANKER=true` (default), fused hits pass through `ScoreFusionReranker`, which recomputes:

```text
final = alpha * semantic + (1 - alpha) * bm25
```

Tune `MINI_AGENT_RERANK_ALPHA` (default `0.7`). Disable reranking to compare raw fusion only.

## Observability

Timings: `search.bm25`, `search.hybrid`, `search.rerank`, `vector.search`, `embedding.query`.
