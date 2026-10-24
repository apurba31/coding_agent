"""Tests for structured latency, counter, and distribution instrumentation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_agent.agent.executor import AgentExecutor
from coding_agent.agent.prompt import PromptBuilder
from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.embedding.mock import MockEmbedder
from coding_agent.indexing.service import RepositoryIndexer
from coding_agent.observability import MetricsCollector
from coding_agent.search.hybrid import HybridSearcher
from coding_agent.search.keyword import BM25Searcher
from coding_agent.search.semantic import SemanticSearcher
from coding_agent.tools.models import ToolCall
from coding_agent.tools.registry import ToolRegistry
from coding_agent.vectordb.client import LanceDBClient


def test_metrics_collector_aggregates_durations_counters_and_values():
    metrics = MetricsCollector()
    metrics.record_duration("search", 2.0)
    metrics.record_duration("search", 4.0)
    metrics.increment("requests", 2)
    metrics.observe("batch_size", 3)

    snapshot = metrics.snapshot()

    assert snapshot.timings["search"].count == 2
    assert snapshot.timings["search"].total_ms == 6.0
    assert snapshot.timings["search"].average_ms == 3.0
    assert snapshot.timings["search"].max_ms == 4.0
    assert snapshot.counters["requests"] == 2
    assert snapshot.values["batch_size"] == (3.0,)


def test_metrics_collector_measure_records_on_exception():
    metrics = MetricsCollector()

    with pytest.raises(RuntimeError, match="failed"):
        with metrics.measure("operation"):
            raise RuntimeError("failed")

    assert metrics.snapshot().timings["operation"].count == 1


def test_indexing_records_pipeline_timings_and_batch_sizes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("def calculate():\n    return 42\n", encoding="utf-8")
    metrics = MetricsCollector()
    embedder = MockEmbedder(embedding_dim=16)
    db = LanceDBClient(tmp_path / "vectors", dimension=16, metrics=metrics)

    summary = RepositoryIndexer(
        embedder,
        db,
        batch_size=1,
        metrics=metrics,
    ).index(repo)
    snapshot = metrics.snapshot()

    assert summary.chunks == 1
    expected_timings = {
        "scan",
        "parse",
        "chunk",
        "embedding",
        "index.vector_write",
        "vector.upsert",
    }
    assert expected_timings <= set(snapshot.timings)
    assert snapshot.counters["files.parsed"] == 1
    assert snapshot.counters["chunks.created"] == 1
    assert snapshot.values["embedding.batch_size"] == (1.0,)


def test_search_metrics_capture_bm25_hybrid_and_vector_results(tmp_path):
    metrics = MetricsCollector()
    embedder = MockEmbedder(embedding_dim=8)
    chunk = Chunk(
        chunk_id="function-1",
        path=Path("src/example.py"),
        language="Python",
        start_line=1,
        end_line=2,
        symbol="calculate_total",
        symbol_kind=ChunkKind.FUNCTION,
        code="def calculate_total():\n    return 42",
    )
    vector = embedder.embed_text(chunk.to_embedding_text()).vector
    db = LanceDBClient(tmp_path / "vectors", dimension=8, metrics=metrics)
    db.add_chunks([chunk], [vector])
    hybrid = HybridSearcher(
        BM25Searcher([chunk], metrics=metrics),
        SemanticSearcher(embedder, db, metrics=metrics),
        metrics=metrics,
    )

    results = hybrid.search("calculate total", top_k=3)
    snapshot = metrics.snapshot()

    assert results
    expected = {"search.bm25", "search.hybrid", "embedding.query", "vector.search"}
    assert expected <= set(snapshot.timings)
    assert snapshot.values["search.hybrid.results"]
    assert snapshot.values["vector.search.results"]


def test_agent_and_tools_record_latency_and_provider_token_usage():
    metrics = MetricsCollector()
    registry = ToolRegistry(metrics=metrics)
    registry.register("ping", lambda: "pong", "Return pong")
    tool_result = registry.call(ToolCall("ping", {}))
    message = SimpleNamespace(content="Answer", tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    class FakeLLM:
        def generate_response(self, **kwargs):
            return response

    class EmptyRetriever:
        def search(self, query, top_k):
            return []

    state = AgentExecutor(
        llm_client=FakeLLM(),
        tool_registry=registry,
        prompt_builder=PromptBuilder("system"),
        retriever=EmptyRetriever(),
        metrics=metrics,
    ).execute("Explain this repository")
    snapshot = metrics.snapshot()

    assert state["status"] == "SUCCESS"
    assert tool_result.success
    assert {"prompt.build", "llm.request", "tool.execution"} <= set(snapshot.timings)
    assert snapshot.counters["tool.calls"] == 1
    assert snapshot.counters["tokens.prompt"] == 11
    assert snapshot.counters["tokens.completion"] == 7
    assert snapshot.counters["tokens.total"] == 18
    assert snapshot.values["tool.execution_ms"]
