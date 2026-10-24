# Indexing

Indexing turns a repository into searchable chunks and vectors.

## Pipeline

```text
CLI index → RepositoryScanner → SourceParser → Chunker → Embedder → LanceDB
                                      ↓
                              BM25 corpus (in-memory at query time)
```

## Incremental behavior

Each indexed file stores a **content hash** in `index_manifest.json`. Unchanged files are skipped; changed files are re-chunked and upserted; deleted paths remove stale vector rows.

Use `mini-agent index --overwrite` or `mini-agent reindex` for a full rebuild.

## Why repository-relative paths

Chunks and manifests store paths relative to the repository root so indexes remain portable across machines.
