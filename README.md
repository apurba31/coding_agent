# Mini Code Agent

An educational AI coding assistant built completely from scratch.

The goal is **not** to build another Cursor or Continue.

The goal is to understand every subsystem that powers modern AI coding assistants.

---

# Features

- Repository indexing
- Tree-sitter parsing
- Intelligent code chunking
- Embeddings
- LanceDB vector storage
- Semantic search
- BM25 keyword search
- Hybrid retrieval
- Prompt construction
- Groq LLM integration
- Tool calling
- Conversation memory
- Deterministic query planner (direct / retrieve / tool)
- LangGraph-backed request routing
- Structured per-process latency, token, batch, tool, and retrieval metrics
- Interactive repository chat
- Optional researcher-to-synthesizer multi-agent workflow
- Read-only `read_file` and `search_code` tools
- AST-backed `find_definition` and `search_symbols`, plus lexical `find_references`
- Modular architecture

Indexing and AST chunk extraction currently support Python, Java, JavaScript, and TypeScript.
Chat requires a Groq API key and uses the local Sentence Transformers model. The agent is
read-only: it can inspect and search repository files, but does not edit them.

---

# Architecture

```
Repository
      │
      ▼
Repository Scanner
      │
      ▼
Tree-sitter Parser
      │
      ▼
Chunker
      │
      ▼
Embeddings
      │
      ▼
LanceDB
      │
      ▼
LangGraph Planner / Retrieval Routing
      │
      ▼
Hybrid Retrieval
      │
      ▼
Prompt Builder
      │
      ▼
Groq
      │
      ▼
Tool Calling
      │
      ▼
Answer
```

---

# Installation

Requires

- Python 3.12+
- Git

Clone

```bash
git clone <repo>
cd mini-code-agent
```

Create virtual environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -e .
```

Create

```
.env
```

Copy

```
.env.example
```

Fill in your Groq API key.

---

# Run

Run tests with `pytest`; isolated subsystem tests remain at `tests/test_<subsystem>.py`, and
cross-subsystem fixture tests live under `tests/integration/`. Select integration tests with
`pytest -m integration`. They use local mock embeddings and a fake LLM, so they need no network or
provider credentials.

Index repository

```bash
mini-agent index
```

Indexing is incremental by default: unchanged files are skipped, changed files are re-embedded,
and deleted-file chunks are removed. Use `--overwrite` for a full rebuild or `--watch` to keep
indexing supported source-file changes.

Index a specific repository and replace the existing vector table:

```bash
mini-agent index --repo . --overwrite
```

The repository path may also be positional (`mini-agent index ./path/to/repo`). Use
`mini-agent reindex ./path/to/repo` to force a full rebuild. `mini-agent inspect ./path` lists
repository files and index-manifest status; `mini-agent stats ./path` reports language and vector
counts, along with metrics collected by the current CLI process, without loading the embedding
model. Metrics are currently in-memory and are not persisted across CLI invocations.

Search

```bash
mini-agent search "Where is UserService?"
```

Search a specific repository:

```bash
mini-agent search "Where is UserService?" --repo .
```

Code navigation works directly from Tree-sitter without loading an embedding model:

```bash
mini-agent find-definition UserService.find_user --repo .
mini-agent find-references find_user --repo .
mini-agent search-symbols "user service" --repo .
```

Reference results use whole-identifier lexical matching; they are not compiler- or LSP-accurate.

Interactive chat

```bash
mini-agent chat
```

Use `mini-agent chat --multi-agent` to run a tool-enabled research agent followed by a separate
answer-synthesis agent. This makes two sequential model passes per request.

Chat execution uses a LangGraph workflow to route requests through direct-answer, retrieval, or
tool-enabled paths. The bounded LLM/tool iteration remains inside the executor for now.

Resume a saved chat with `--conversation-id <id>`. Enter `/exit` to leave the interactive
session; use `--memory-path` to select a different conversation directory.

---

# Project Structure

```
src/coding_agent/
├── agent/       # bounded orchestration loop and prompts
├── chunker/     # AST-aware chunk extraction
├── embedding/   # sentence-transformer and mock embedders
├── indexing/    # repository pipeline and hash manifest
├── llm/         # Groq client
├── memory/      # persisted conversation history
├── navigation/  # AST definitions and lexical references
├── observability/ # shared timings, counters, and distributions
├── parser/      # Tree-sitter engine
├── planner/     # deterministic direct/retrieve/tool routing
├── scanner/     # repository walking and metadata
├── search/      # BM25, semantic, and hybrid retrieval
├── tools/       # tool schemas, registry, and executor
└── vectordb/    # LanceDB schema and client
```

---

# Roadmap

- [x] Repository indexing and Tree-sitter chunk extraction
- [x] Embeddings and LanceDB vector storage
- [x] Semantic, BM25, and hybrid search
- [x] Groq agent loop, tool calling, and persistent multi-turn memory
- [x] Index, search, and chat CLI
- [x] Positional repository indexing, `inspect`, `stats`, and full `reindex` commands
- [x] Hash-based incremental indexing, deterministic upserts, deletion cleanup, and watch mode
- [x] Optional two-role multi-agent handoff
- [x] AST-backed definition lookup, symbol search, and bounded lexical references
- [x] Rule-based planner with explicit retrieval and tool gating
- [x] LangGraph routing for direct, retrieval, and tool-enabled execution paths
- [x] Structured timings/counters for indexing, search, prompt/LLM, token usage, and tools
- [x] Phase timings and counters surfaced by `index` and `stats`
- [x] Offline fixture-based end-to-end indexing, navigation, retrieval, and agent-answer test
- [ ] Move legacy flat subsystem tests into dedicated package directories
- [ ] File-editing tools and test execution
- [ ] Reranking
- [ ] Graph-native iterative tool nodes and checkpointing
- [ ] Parallel/dynamic multi-agent delegation
- [ ] MCP
- [ ] Streaming, caching, and performance work