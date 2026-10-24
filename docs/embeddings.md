# Embeddings

Default model: `BAAI/bge-small-en-v1.5` via `SentenceTransformerEmbedder`.

The embedder interface (`Embedder`) keeps Sentence Transformers out of retrieval and indexing callers. Tests use `MockEmbedder` to avoid model downloads.

Batch size and device come from `AgentConfig` / `MINI_AGENT_*` env vars.
