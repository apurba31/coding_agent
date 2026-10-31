# Mini Code Agent

Mini Code Agent is an educational, local-first coding assistant. It indexes a
repository, finds relevant code with keyword and semantic search, navigates
symbols, and answers questions through a configured language-model provider.
Interactive chat can also read files, inspect Git state, and—when explicitly
enabled—write repository files or run bounded commands.

The project is designed to make the subsystems of a coding agent understandable:
repository scanning, syntax parsing, chunking, embeddings, vector storage,
retrieval, planning, tool calls, conversation memory, and observability.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Provider configuration](#provider-configuration)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Chat tools](#chat-tools)
- [Safety notes](#safety-notes)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Tests and quality checks](#tests-and-quality-checks)
- [Configuration reference](#configuration-reference)
- [Limitations](#limitations)

## Features

- Incremental repository indexing with SHA-256 file manifests and deletion cleanup.
- Tree-sitter parsing and syntax-aware code chunks.
- Local Sentence Transformers embeddings, stored in LanceDB.
- Hybrid retrieval that combines BM25 keyword search and semantic vector search,
  with configurable score fusion.
- AST-backed definition and symbol lookup, plus lexical whole-identifier
  reference search.
- Interactive chat, persistent conversation history, and SQLite-backed
  LangGraph checkpoints.
- A deterministic planner that routes requests to direct answers, repository
  retrieval, or explicitly permitted tools.
- Groq and NVIDIA NIM language-model providers through compatible chat APIs.
- Optional multi-agent research and answer synthesis.
- Optional Model Context Protocol (MCP) stdio servers.
- Per-process timings, counters, token metrics, and retrieval/tool observations.
- Command-line tools for indexing, searching, repository inspection, and chat.

The scanner recognizes source files including Python, Java, JavaScript,
TypeScript, Go, and Rust. AST chunking and symbol navigation are implemented
for Python, Java, JavaScript, and TypeScript. Other recognized files may be
included in repository inventory, but do not necessarily produce code chunks
or AST navigation results.

## Requirements

- Python 3.12 or newer.
- Git is optional for scanning and indexing. Git-backed chat tools require a
  Git working tree.
- A Groq or NVIDIA NIM API key for live chat.
- Internet access the first time the configured Sentence Transformers model is
  downloaded. Embeddings then run locally by default on CPU.

The model, vector index, conversation data, and graph checkpoints are stored
locally. Do not commit `.env` or other credentials.

## Installation

From the repository root, create and activate a virtual environment.

### Windows PowerShell

```powershell
Set-Location D:\coding_agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

### macOS or Linux

```bash
cd /path/to/coding_agent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp -n .env.example .env
```

The editable install creates the `mini-agent` command and installs the project
dependencies plus development tools. If you only need to run the application,
install with `pip install -e .` instead of `pip install -e ".[dev]"`.

## Provider configuration

Edit `.env` and configure the provider you intend to use. The API key value
belongs only in your local `.env`; never paste it into source code, chat logs,
or a commit.

### Groq

```dotenv
MINI_AGENT_LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
# Optional. The application supplies a default if omitted.
# MINI_AGENT_LLM_MODEL=your-groq-model
```

### NVIDIA NIM

```dotenv
MINI_AGENT_LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your-nvidia-api-key
NVIDIA_MODEL=your-nvidia-model-id
# Optional; this is the default NVIDIA NIM OpenAI-compatible endpoint.
# NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

The chat command loads `.env` when it starts. Provider and model may also be
selected with `--provider` and `--model`; these command-line options take
precedence over the corresponding defaults.

## Quick start

Index a repository before using semantic search or chat. Replace the sample path
with the repository you want to explore.

### Windows PowerShell

```powershell
$repo = "D:\coding_agent\tests\fixtures\sample_repo"
$db = "$env:TEMP\mini-agent-sample-index"

mini-agent index $repo --db-path $db
mini-agent inspect $repo --db-path $db
mini-agent search "Where is the user service implemented?" --repo $repo --db-path $db
mini-agent chat --repo $repo --db-path $db
```

### macOS or Linux

```bash
repo=/path/to/coding_agent/tests/fixtures/sample_repo
db=/tmp/mini-agent-sample-index

mini-agent index "$repo" --db-path "$db"
mini-agent inspect "$repo" --db-path "$db"
mini-agent search "Where is the user service implemented?" --repo "$repo" --db-path "$db"
mini-agent chat --repo "$repo" --db-path "$db"
```

At the interactive `You:` prompt, enter a question or task. Use `/exit` or
`/quit` to end the chat. Chat is read-only by default. To allow edits, start
chat with `--allow-write`; see [Chat tools](#chat-tools) and
[Safety notes](#safety-notes).

If the command is not on `PATH`, run the installed script directly:

```powershell
.\.venv\Scripts\mini-agent.exe --help
```

```bash
.venv/bin/mini-agent --help
```

## CLI reference

Run `mini-agent --help` or `mini-agent <command> --help` for the live option
list.

| Command | Purpose |
| --- | --- |
| `mini-agent index [REPO]` | Incrementally scan, parse, embed, and index a repository. The repository can also be supplied with `--repo`. |
| `mini-agent reindex [REPO]` | Rebuild the vector index from scratch. |
| `mini-agent inspect [REPO]` | List repository files and show index-manifest status. |
| `mini-agent stats [REPO]` | Report file/language, chunk, vector, manifest, and current-process metric counts. |
| `mini-agent search QUERY` | Run hybrid keyword and semantic search and display ranked chunks. |
| `mini-agent find-definition SYMBOL` | Find AST-indexed definitions by exact or qualified name. |
| `mini-agent find-references SYMBOL` | Find whole-identifier occurrences in supported source files. |
| `mini-agent search-symbols QUERY` | Search AST-indexed functions, methods, classes, and interfaces. |
| `mini-agent chat` | Start an interactive repository chat session. |

### Indexing options

Indexing is incremental by default: unchanged files are skipped, changed files
are re-embedded, and chunks for deleted files are removed.

```powershell
mini-agent index D:\path\to\repo
mini-agent index --repo D:\path\to\repo --overwrite
mini-agent index D:\path\to\repo --watch
mini-agent reindex D:\path\to\repo
```

- `--overwrite` replaces the existing index during an `index` run.
- `--watch` keeps watching supported source-file changes until interrupted
  with Ctrl+C.
- `reindex` forces a full rebuild.
- `--db-path PATH` selects a vector database location. The default is
  `.mini-agent/lancedb`.

### Search and navigation examples

```powershell
mini-agent search "user lookup by id" --repo D:\path\to\repo
mini-agent find-definition UserService.find_user --repo D:\path\to\repo
mini-agent find-references find_user --repo D:\path\to\repo
mini-agent search-symbols "user service" --repo D:\path\to\repo
```

Search and chat use the vector index at `--db-path`; index the same repository
to that path first. Definition, symbol, and reference navigation work directly
from source parsing and do not load the embedding model.

### Chat options

```powershell
mini-agent chat --repo D:\path\to\repo
mini-agent chat --repo D:\path\to\repo --allow-write
mini-agent chat --repo D:\path\to\repo --allow-terminal
mini-agent chat --repo D:\path\to\repo --multi-agent
mini-agent chat --repo D:\path\to\repo --provider nvidia --model your-model-id
mini-agent chat --repo D:\path\to\repo --conversation-id <id>
```

- `--repo` scopes repository operations to the selected directory.
- `--db-path` selects the index used for retrieval.
- `--memory-path` selects where conversation history is stored.
- `--conversation-id` resumes a saved conversation.
- `--allow-write` enables `write_file`; it is off by default.
- `--allow-terminal` enables `run_terminal` and `run_tests`; both are off by
  default.
- `--multi-agent` uses a researcher stage and a separate synthesis stage. It
  may make multiple sequential or parallel model requests.
- `--provider` accepts `groq` or `nvidia`; `--model` overrides the configured
  model.

## Chat tools

The default chat registers read-only repository tools. The planner decides
whether a request needs repository retrieval or tool access; for example, an
explicit “read this file” or “add a line to this file” request is routed toward
tools. Merely mentioning a repository does not automatically grant write or
terminal access.

| Tool | Enabled by | What it does |
| --- | --- | --- |
| `read_file` | Default | Reads a UTF-8 repository file. Secret paths and paths outside the selected repository are rejected. |
| `list_directory` | Default | Lists entries in a repository directory and omits secret files. |
| `search_code` | Default | Searches indexed code using hybrid keyword and semantic retrieval. |
| `find_definition` | Default | Finds a definition by symbol or qualified name using the AST navigation index. |
| `find_references` | Default | Finds whole-identifier source-line occurrences; results are lexical rather than compiler/LSP accurate. |
| `search_symbols` | Default | Searches AST-indexed functions, methods, classes, and interfaces. |
| `git_status` | Default | Reads Git working-tree status through GitPython. |
| `git_diff` | Default | Reads staged and unstaged diffs; it does not change the working tree. |
| `write_file` | `--allow-write` | Writes complete UTF-8 content to a path inside the selected repository. Secret files are blocked. |
| `run_terminal` | `--allow-terminal` | Runs a shell command from the selected repository root with a timeout, output cap, and denylist for known unsafe command patterns. |
| `run_tests` | `--allow-terminal` | Runs pytest for a selected repository-relative test file or directory; defaults to `tests`. |
| `mcp_<server>_<tool>` | MCP server configured | Makes a tool advertised by a configured MCP stdio server available to the chat agent. |

Built-in file tools resolve paths against the `--repo` root. To let the agent
edit a project, set `--repo` to that project and explicitly pass
`--allow-write`. For example:

```powershell
mini-agent chat --repo D:\my-project --allow-write
```

The agent receives only the tools enabled for that session. Tool results are
logged; the application should not claim a file was changed unless
`write_file` returns successfully.

### MCP servers

MCP server definitions are read from `MINI_AGENT_MCP_SERVERS` as JSON. Each
server entry specifies a local command, optional arguments, and an optional
explicit environment:

```dotenv
MINI_AGENT_MCP_SERVERS={"filesystem":{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","D:/my-project"]}}
```

The MCP Python SDK is only needed when MCP servers are configured. Install the
optional dependency into the active environment with:

```powershell
python -m pip install -e ".[mcp]"
```

MCP server processes and their tools are external to this project's built-in
repository tools; configure only servers you trust. Their descriptions and
results are treated as untrusted content by the agent.

## Safety notes

- Chat starts with file writing and terminal execution disabled. Opt in using
  `--allow-write` and/or `--allow-terminal` only when appropriate.
- The built-in filesystem policy rejects paths outside the selected repository
  and blocks common credential file names and key/certificate suffixes.
- Repository scanning also skips common generated directories, binary files,
  secret files, and basic `.gitignore` matches.
- The terminal tool has timeout/output limits and a denylist, but it is **not
  an operating-system sandbox**. Do not give it access to an untrusted
  repository or rely on it as a security boundary.
- MCP servers run as local processes and have their own permissions. The
  repository path restrictions of built-in file tools do not sandbox MCP
  servers.
- Keep API keys in `.env`; the repository scanner and built-in file tools block
  `.env` from indexing/reading.

## How it works

```text
Repository
   |
   v
Scanner --> parser --> syntax-aware chunks --> local embeddings
   |                                             |
   |                                             v
   +------------------------------------------> LanceDB
                                                 |
User prompt --> deterministic planner --> direct / retrieve / tool route
                                           |         |
                                           |         +--> repository tools
                                           v
                                  BM25 + vector search
                                           |
                                           v
                                  prompt + conversation
                                           |
                                           v
                                  Groq or NVIDIA NIM
                                           |
                                           v
                                  answer / tool cycle
```

1. **Scan:** Walks the selected repository, identifies files, calculates hashes,
   honors common ignore rules, and filters secrets and binary files.
2. **Parse and chunk:** Tree-sitter syntax trees are used to extract code
   chunks and symbols for supported languages.
3. **Embed and store:** Sentence Transformers encodes chunks locally; LanceDB
   stores their vectors and metadata. A manifest tracks file hashes for
   incremental indexing.
4. **Plan:** A small deterministic planner routes a request to a direct answer,
   retrieval, or tool-enabled path. Tool execution is gated by this plan as
   well as by session options.
5. **Retrieve:** BM25 keyword search and vector similarity search contribute
   candidates to hybrid retrieval; optional score fusion reranking can refine
   the order.
6. **Answer and tools:** A configured provider receives the prompt and
   available tool schemas. The bounded executor processes structured tool calls
   and returns results to the model.
7. **Persist and measure:** Conversations and LangGraph checkpoints are stored
   locally. Runtime metrics are collected in memory for the current process.

## Project structure

```text
src/coding_agent/
├── agent/          # Executor, prompts, LangGraph routing, multi-agent handoff
├── chunker/        # Syntax-aware code chunk models and chunking
├── embedding/      # Sentence Transformers and test/mock embeddings
├── harness/        # Development and test harness utilities
├── indexing/       # Index pipeline, incremental manifest, watch support
├── llm/            # Provider clients, caching, and stream collection
├── memory/         # Conversation models and local persistence
├── models/         # Repository, file, and language models
├── navigation/     # Definition, symbol, and reference lookup
├── observability/  # Timings, counters, and distributions
├── parser/         # Tree-sitter grammar loading and parsing
├── planner/        # Direct/retrieve/tool routing
├── prompt/         # Prompt utilities
├── reranker/       # Search score fusion/reranking
├── scanner/        # Repository walking and ignore handling
├── search/         # BM25, semantic, and hybrid retrieval
├── security/       # Repository path, secret, and command policies
├── tools/          # Filesystem, Git, terminal, MCP, registry and schemas
├── utils/          # Shared helpers
└── vectordb/       # LanceDB client and chunk schema

tests/
├── agent/          # Agent orchestration and workflow tests
├── fixtures/       # Small sample repositories
├── integration/    # Offline end-to-end coverage
├── indexing/       # Index pipeline tests
├── llm/            # Provider adapter and streaming tests
├── navigation/     # Symbol and reference tests
├── planner/        # Routing and tool-gating tests
└── ...             # Other subsystem tests
```

## Tests and quality checks

Install the development dependencies with `pip install -e ".[dev]"`, then run:

```powershell
python -m pytest -q
ruff check .
ruff format --check .
```

On macOS or Linux, the same commands work in the activated virtual environment.
Integration tests are offline and use mock embeddings and a fake LLM:

```powershell
python -m pytest -m integration -q
```

Tests requiring a live language model or a real embedding model are marked
separately. Do not run them unless their credentials/network/model requirements
are available.

## Configuration reference

`AgentConfig` reads `MINI_AGENT_`-prefixed settings from environment variables
and `.env`. Chat also reads provider-specific API-key variables.

| Setting | Default | Purpose |
| --- | --- | --- |
| `MINI_AGENT_LLM_PROVIDER` | `groq` | Provider: `groq` or `nvidia`. |
| `MINI_AGENT_LLM_MODEL` | Provider default | Optional model override. |
| `MINI_AGENT_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence embedding model. |
| `MINI_AGENT_EMBEDDING_DEVICE` | `cpu` | Embedding device, for example `cpu` or a supported accelerator. |
| `MINI_AGENT_EMBEDDING_BATCH_SIZE` | `32` | Batch size for local embeddings. |
| `MINI_AGENT_USE_RERANKER` | `true` | Enable the configured score-fusion reranker. |
| `MINI_AGENT_RERANK_ALPHA` | `0.7` | Score-fusion parameter. |
| `MINI_AGENT_TERMINAL_TIMEOUT_SECONDS` | `15` | Maximum terminal command duration. |
| `MINI_AGENT_TERMINAL_MAX_OUTPUT_BYTES` | `32000` | Maximum captured command output. |
| `MINI_AGENT_MCP_SERVERS` | `{}` | JSON map of local stdio MCP server configurations. |

Chat flags such as `--repo`, `--db-path`, and `--memory-path` let a user select
paths for a particular command without changing these settings.

## Limitations

- AST chunking and navigation currently support Python, Java, JavaScript, and
  TypeScript; scanner language recognition is broader than AST feature support.
- Reference search is lexical and whole-identifier based. It is not
  compiler-aware and may include comments or strings.
- Indexing source is limited to the supported file types and repositories can
  contain ignored, binary, or unsupported files.
- The planner uses deterministic heuristics. It may route an ambiguous request
  imperfectly; explicitly describe whether you want a file read, search, or
  edit.
- The terminal denylist is intentionally conservative and is not a complete
  shell security mechanism.
- Metrics are process-local and are not persisted between CLI invocations.
- Multi-agent mode can use more model calls and incur additional latency or
  provider usage.
- MCP support depends on a separately installed MCP SDK and trusted local
  servers.
