"""Offline end-to-end coverage for repository indexing through an agent answer."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_agent.agent.executor import AgentExecutor
from coding_agent.agent.prompt import PromptBuilder
from coding_agent.agent.workflow import LangGraphAgent
from coding_agent.embedding.mock import MockEmbedder
from coding_agent.indexing.service import RepositoryIndexer
from coding_agent.navigation import NavigationIndex
from coding_agent.planner import SimplePlanner
from coding_agent.search.hybrid import HybridSearcher
from coding_agent.search.keyword import BM25Searcher
from coding_agent.search.semantic import SemanticSearcher
from coding_agent.tools.registry import ToolRegistry
from coding_agent.vectordb.client import LanceDBClient

FIXTURE_REPOSITORY = Path(__file__).parents[1] / "fixtures" / "sample_repo"
pytestmark = pytest.mark.integration


def test_repository_pipeline_finds_user_service_and_answers_with_citations(tmp_path):
    embedder = MockEmbedder(embedding_dim=48)
    db = LanceDBClient(tmp_path / "vectors", dimension=embedder.embedding_dimension)
    indexer = RepositoryIndexer(embedder, db)
    index_summary = indexer.index(FIXTURE_REPOSITORY, overwrite=True)
    chunks, scanned, parsed, failed = indexer.collect_chunks(FIXTURE_REPOSITORY)
    navigation = NavigationIndex(FIXTURE_REPOSITORY, chunks)
    bm25 = BM25Searcher(chunks)
    semantic = SemanticSearcher(embedder, db)
    retriever = HybridSearcher(bm25, semantic)

    definition = navigation.find_definition("UserService")
    assert len(definition) == 1
    assert definition[0].path == Path("src/main/java/com/example/users/UserService.java")
    assert definition[0].start_line == 6

    method_results = navigation.find_definition("UserService.findUser")
    assert len(method_results) == 1
    assert method_results[0].path == definition[0].path
    assert method_results[0].start_line == 14

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate_response(self, **kwargs):
            self.calls.append(kwargs)
            system = kwargs["messages"][0]["content"]
            assert "UserService.java:14-16" in system
            assert "Optional.ofNullable(users.get(id))" in system
            message = SimpleNamespace(
                content=(
                    "`UserService` is defined in "
                    "src/main/java/com/example/users/UserService.java:6-16. "
                    "`findUser` looks up the ID in the users map and wraps the "
                    "result with Optional.ofNullable (lines 14-16)."
                ),
                tool_calls=None,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    llm = FakeLLM()
    executor = AgentExecutor(
        llm_client=llm,
        tool_registry=ToolRegistry(),
        prompt_builder=PromptBuilder("Answer from repository evidence and cite paths/lines."),
        retriever=retriever,
        planner=SimplePlanner(),
    )
    graph_agent = LangGraphAgent(executor)
    answer = graph_agent.execute("How does findUser work in UserService?")

    assert index_summary.chunks > 0
    assert index_summary.failed_files == 0
    assert scanned >= 6
    assert parsed == 4
    assert failed == 0
    assert db.count() == index_summary.chunks
    assert answer["status"] == "SUCCESS"
    assert answer["plan"].needs_repository_context
    assert "UserService.java:6-16" in answer["messages"][-1]["content"]
    assert "lines 14-16" in answer["messages"][-1]["content"]
    assert len(llm.calls) == 1
