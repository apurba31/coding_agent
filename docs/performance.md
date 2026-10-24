# Performance

Current focus: correctness and modularity, with measurement hooks first.

`MetricsCollector` records phase timings (scan, parse, embed, search, rerank, LLM, tools). Use `mini-agent stats` for process-local telemetry.

The CLI wraps its provider in a bounded in-memory LRU response cache (128 entries by
default). It caches exact request messages/options only when no tools are supplied and
temperature is zero. Requests with tools, non-zero or unspecified provider temperature,
or values that cannot be serialized for an exact cache key bypass the cache. Entries are
process-local and are never written to disk. `llm.cache.hits`, `llm.cache.misses`, and
`llm.cache.bypasses` appear in process-local stats.

Interactive chat streams text deltas from the Groq and OpenAI-compatible clients. Tool-call
deltas are reassembled before the executor runs tools. Other `LLMClient` implementations use
the base fallback, which delivers the completed response as one chunk; cache hits are likewise
replayed as one chunk.

## Measured optimization

Profiling `RepositoryIndexer.collect_chunks` on `src/coding_agent` visited 86 files, parsed
78 files, and produced 622 chunks. In a paired local benchmark of 250 repeated BM25 queries
over those chunks (median of four rounds), selecting only the requested top five results
reduced elapsed time from 331.78 ms to 242.86 ms (26.8%). The optimized search avoids
materializing and sorting a `SearchResult` for every positive-scoring chunk while preserving
score order and input order for ties. These timings are local measurements, not a performance
guarantee.

Other extension points include larger embedding batches, persistent BM25 index, parallel
scan/parse, and async LLM execution.
