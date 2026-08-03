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
- Modular architecture

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

Index repository

```bash
mini-agent index
```

Search

```bash
mini-agent search "Where is UserService?"
```

Interactive chat

```bash
mini-agent chat
```

---

# Project Structure

```
src/

parser/

chunking/

embeddings/

vectordb/

retrieval/

planner/

memory/

tools/

llm/

agent.py
```

---

# Roadmap

- [ ] Repository indexing
- [ ] Tree-sitter parser
- [ ] Intelligent chunking
- [ ] Embeddings
- [ ] LanceDB
- [ ] Semantic Search
- [ ] BM25
- [ ] Hybrid Search
- [ ] Tool Calling
- [ ] Memory
- [ ] Planner
- [ ] Multi-agent
- [ ] MCP
- [ ] Incremental indexing