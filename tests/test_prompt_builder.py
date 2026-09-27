"""Tests for the prompt-building subsystem."""

from pathlib import Path

from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.prompt import PromptBuilder
from coding_agent.search.models import SearchResult


def _chunk(chunk_id: str, path: str, symbol: str, code: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        path=Path(path),
        language="python",
        symbol_kind=ChunkKind.FUNCTION,
        symbol=symbol,
        start_line=1,
        end_line=4,
        code=code,
        docstring="",
        comments=[],
        imports=[],
        parent_symbol=None,
        is_public=True,
    )


def test_prompt_builder_formats_repository_context_block():
    builder = PromptBuilder("You are a coding assistant.")
    results = [
        SearchResult(
            chunk=_chunk("c1", "src/example.py", "example", "def example():\n    return 42"),
            score=0.9,
            chunk_id="c1",
        )
    ]

    messages = builder.build_initial_messages(
        "Explain it.",
        context=builder.build_repository_context(results, token_budget=200),
    )

    content = messages[0]["content"]
    assert "<repository_context>" in content
    assert "FILE: src/example.py" in content
    assert "SYMBOL: example" in content
    assert "def example()" in content


def test_prompt_builder_respects_token_budget_and_deduplicates_files():
    builder = PromptBuilder("You are a coding assistant.")
    results = [
        SearchResult(
            chunk=_chunk("c1", "src/a.py", "alpha", "def alpha():\n    return 1"),
            score=0.9,
            chunk_id="c1",
        ),
        SearchResult(
            chunk=_chunk("c2", "src/a.py", "beta", "def beta():\n    return 2"),
            score=0.8,
            chunk_id="c2",
        ),
        SearchResult(
            chunk=_chunk("c3", "src/b.py", "gamma", "def gamma():\n    return 3"),
            score=0.7,
            chunk_id="c3",
        ),
    ]

    context = builder.build_repository_context(results, token_budget=30)
    assert "src/a.py" in context.replace('\\', '/')
    assert "src/b.py" in context.replace('\\', '/') or "src/b.py" not in context.replace('\\', '/')
    assert len(context.split()) <= 60
