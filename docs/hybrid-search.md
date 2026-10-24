# Hybrid search

See [retrieval.md](retrieval.md) for the fusion pipeline.

Key modules:

- `coding_agent/search/keyword.py` — BM25 + code-aware tokenization
- `coding_agent/search/semantic.py` — query embed + LanceDB search
- `coding_agent/search/hybrid.py` — normalization, fusion, optional rerank

CLI: `mini-agent search "query" --repo .`
