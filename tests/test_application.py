"""Application-level tests for indexing, retrieval, and repository tools."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from coding_agent import app as app_module
from coding_agent.embedding.mock import MockEmbedder
from coding_agent.indexing.service import RepositoryIndexer
from coding_agent.navigation import NavigationIndex
from coding_agent.search.hybrid import HybridSearcher
from coding_agent.search.keyword import BM25Searcher
from coding_agent.search.semantic import SemanticSearcher
from coding_agent.tools.models import ToolCall
from coding_agent.vectordb.client import LanceDBClient

runner = CliRunner()


def _write_repository(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(
        "def meaningful_function(value):\n    return value + 42\n",
        encoding="utf-8",
    )
    ignored = root / "node_modules" / "unused-package"
    ignored.mkdir(parents=True)
    (ignored / "ignored.py").write_text("def should_not_be_seen(): pass\n", encoding="utf-8")


def test_repository_index_pipeline_and_hybrid_retrieval(tmp_path):
    repo = tmp_path / "repo"
    _write_repository(repo)
    embedder = MockEmbedder(embedding_dim=32)
    db = LanceDBClient(tmp_path / "vectors", dimension=embedder.embedding_dimension)

    summary = RepositoryIndexer(embedder, db).index(repo, overwrite=True)
    chunks, _, _, _ = RepositoryIndexer(embedder, db).collect_chunks(repo)
    retriever = HybridSearcher(
        BM25Searcher(chunks),
        SemanticSearcher(embedder, db),
    )
    results = retriever.search("meaningful function", top_k=5)

    assert summary.chunks >= 1
    assert summary.parsed_files == 1
    assert summary.failed_files == 0
    assert db.count() == summary.chunks
    assert results
    assert results[0].chunk.path == Path("main.py")
    assert all("should_not_be_seen" not in chunk.code for chunk in chunks)


def test_index_and_search_cli_commands_use_repository_pipeline(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repository(repo)
    db_path = tmp_path / "vectors"
    monkeypatch.setattr(app_module, "_make_embedder", lambda: MockEmbedder(embedding_dim=32))

    index_result = runner.invoke(
        app_module.app,
        ["index", "--repo", str(repo), "--db-path", str(db_path), "--overwrite"],
    )
    search_result = runner.invoke(
        app_module.app,
        ["search", "meaningful function", "--repo", str(repo), "--db-path", str(db_path)],
    )

    assert index_result.exit_code == 0, index_result.output
    assert "indexed" in index_result.output
    assert search_result.exit_code == 0, search_result.output
    assert "meaningful_function" in search_result.output
    assert "Semantic" in search_result.output
    assert "BM25" in search_result.output
    assert "Final" in search_result.output


def test_coding_tools_read_repo_files_and_block_path_traversal(tmp_path):
    repo = tmp_path / "repo"
    _write_repository(repo)
    outside = tmp_path / "secret.txt"
    outside.write_text("not for the agent", encoding="utf-8")
    embedder = MockEmbedder(embedding_dim=32)
    db = LanceDBClient(tmp_path / "vectors", dimension=32)
    chunks, _, _, _ = RepositoryIndexer(embedder, db).collect_chunks(repo)
    retriever = HybridSearcher(BM25Searcher(chunks), SemanticSearcher(embedder, db))
    registry = app_module.create_coding_tools(repo, retriever)

    safe = registry.call(ToolCall("read_file", {"path": "main.py"}))
    blocked = registry.call(ToolCall("read_file", {"path": "../secret.txt"}))
    search = registry.call(ToolCall("search_code", {"query": "meaningful_function"}))

    assert safe.success and "meaningful_function" in safe.result
    assert not blocked.success
    assert "outside the repository" in blocked.error
    assert search.success and "meaningful_function" in search.result


def test_cli_help_lists_production_commands():
    result = runner.invoke(app_module.app, ["--help"])

    assert result.exit_code == 0
    assert all(
        command in result.output
        for command in ("index", "reindex", "inspect", "stats", "search", "chat")
    )


def test_watch_filter_ignores_non_source_and_generated_paths():
    assert app_module._watch_filter(None, "repo/src/module.py")
    assert not app_module._watch_filter(None, "repo/README.md")
    assert not app_module._watch_filter(None, "repo/.mini-agent/index_manifest.json")


def test_navigation_commands_find_symbols_without_embedding_model(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repository(repo)
    source = repo / "main.py"
    source.write_text(
        "def meaningful_function(value):\n    return helper(value)\n\n"
        "def helper(value):\n    return value + 42\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        app_module,
        "_make_navigation_index",
        lambda path: NavigationIndex.from_repository(path),
    )

    definition = runner.invoke(
        app_module.app,
        ["find-definition", "helper", "--repo", str(repo)],
    )
    references = runner.invoke(
        app_module.app,
        ["find-references", "helper", "--repo", str(repo)],
    )
    symbols = runner.invoke(
        app_module.app,
        ["search-symbols", "meaningful function", "--repo", str(repo)],
    )

    assert definition.exit_code == 0, definition.output
    assert "main.py:4-5" in definition.output
    assert references.exit_code == 0, references.output
    assert "main.py:2" in references.output
    assert "[definition]" in references.output
    assert symbols.exit_code == 0, symbols.output
    assert "meaningful_function" in symbols.output


def test_index_accepts_positional_repository_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repository(repo)
    monkeypatch.setattr(app_module, "_make_embedder", lambda: MockEmbedder(embedding_dim=32))

    result = runner.invoke(
        app_module.app,
        ["index", str(repo), "--db-path", str(tmp_path / "vectors")],
    )

    assert result.exit_code == 0, result.output
    assert "indexed" in result.output


def test_inspect_and_stats_report_repo_and_index_status(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repository(repo)
    db_path = tmp_path / "vectors"
    monkeypatch.setattr(app_module, "_make_embedder", lambda: MockEmbedder(embedding_dim=32))
    indexed = runner.invoke(
        app_module.app,
        ["index", str(repo), "--db-path", str(db_path)],
    )
    assert indexed.exit_code == 0, indexed.output

    inspected = runner.invoke(
        app_module.app,
        ["inspect", str(repo), "--db-path", str(db_path), "--limit", "5"],
    )
    statistics = runner.invoke(
        app_module.app,
        ["stats", str(repo), "--db-path", str(db_path)],
    )

    assert inspected.exit_code == 0, inspected.output
    assert str(repo) in inspected.output
    assert "main.py" in inspected.output
    assert "files tracked" in inspected.output
    assert statistics.exit_code == 0, statistics.output
    assert "Vector records" in statistics.output
    assert "Chunks in source" in statistics.output
    assert "BM25 records" in statistics.output
    assert "Python files" in statistics.output


def test_reindex_forces_full_rebuild(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repository(repo)
    db_path = tmp_path / "vectors"
    monkeypatch.setattr(app_module, "_make_embedder", lambda: MockEmbedder(embedding_dim=32))
    indexer = runner.invoke(
        app_module.app,
        ["index", str(repo), "--db-path", str(db_path)],
    )
    reindexed = runner.invoke(
        app_module.app,
        ["reindex", str(repo), "--db-path", str(db_path)],
    )

    assert indexer.exit_code == 0, indexer.output
    assert reindexed.exit_code == 0, reindexed.output
    assert "Rebuilt index" in reindexed.output
    assert "created 1 chunks" in reindexed.output


def test_chat_cli_runs_agent_and_persists_conversation(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repository(repo)

    class FakeRetriever:
        def search(self, query, top_k):
            return []

    class FakeLLM:
        def generate_response(self, **kwargs):
            assert kwargs["messages"][-1]["content"] == "Find the definition of meaningful_function"
            assert {tool["function"]["name"] for tool in kwargs["tools"]} == {
                "read_file",
                "search_code",
                "find_definition",
                "find_references",
                "search_symbols",
            }
            message = SimpleNamespace(content="It adds 42 to the value.", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(app_module, "_make_embedder", lambda: MockEmbedder(embedding_dim=32))
    monkeypatch.setattr(app_module, "_make_searcher", lambda *args: FakeRetriever())
    monkeypatch.setattr(app_module, "GroqClient", FakeLLM)
    memory_path = tmp_path / "conversations"

    result = runner.invoke(
        app_module.app,
        [
            "chat",
            "--repo",
            str(repo),
            "--db-path",
            str(tmp_path / "vectors"),
            "--memory-path",
            str(memory_path),
        ],
        input="Find the definition of meaningful_function\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert "It adds 42 to the value." in result.output
    saved_conversations = list(memory_path.glob("*.json"))
    assert len(saved_conversations) == 1


def test_multi_agent_chat_hands_research_to_final_agent(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _write_repository(repo)

    class FakeRetriever:
        def search(self, query, top_k):
            return []

    class FakeLLM:
        def __init__(self):
            self.calls = []
            self.responses = [
                "The function in main.py adds 42 to its input.",
                "It adds 42 to the input value.",
            ]

        def generate_response(self, **kwargs):
            self.calls.append(deepcopy(kwargs))
            message = SimpleNamespace(content=self.responses.pop(0), tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_llm = FakeLLM()
    monkeypatch.setattr(app_module, "_make_embedder", lambda: MockEmbedder(embedding_dim=32))
    monkeypatch.setattr(app_module, "_make_searcher", lambda *args: FakeRetriever())
    monkeypatch.setattr(app_module, "GroqClient", lambda: fake_llm)
    memory_path = tmp_path / "conversations"

    result = runner.invoke(
        app_module.app,
        [
            "chat",
            "--multi-agent",
            "--repo",
            str(repo),
            "--db-path",
            str(tmp_path / "vectors"),
            "--memory-path",
            str(memory_path),
        ],
        input="Explain the function\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert len(fake_llm.calls) == 2
    assert "The function in main.py adds 42" in fake_llm.calls[1]["messages"][0]["content"]
    assert fake_llm.calls[1]["messages"][-1]["content"] == "Explain the function"
    assert "It adds 42 to the input value." in result.output
    assert len(list(memory_path.glob("*.json"))) == 1
