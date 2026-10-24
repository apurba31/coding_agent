# Reranking

`Reranker` is an abstract refinement step after hybrid fusion.

Shipped implementation: **ScoreFusionReranker** — reweights stored `bm25_score` and `semantic_score` with `alpha`.

Extension points (not shipped): cross-encoder, BGE reranker, LLM reranker.

Wired by default in `_make_searcher()` when `use_reranker` is true.
