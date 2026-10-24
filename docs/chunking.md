# Chunking

Chunks follow AST boundaries (classes, methods, functions), not fixed character windows.

Each chunk carries stable IDs, symbol metadata, imports, comments, and code text for both BM25 and embedding formatting (`Chunk.to_embedding_text()`).

Large symbols may split at nested AST boundaries when over the token budget.
