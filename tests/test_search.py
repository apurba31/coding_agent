from pathlib import Path

from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.search.keyword import BM25Searcher, CodeTokenizer


def test_code_tokenizer_snake_case():
    tokens = CodeTokenizer.tokenize("find_user_by_id")
    assert "find_user_by_id" in tokens
    assert "find" in tokens
    assert "user" in tokens
    assert "by" in tokens
    assert "id" in tokens


def test_code_tokenizer_camel_case():
    tokens = CodeTokenizer.tokenize("findUserById")
    assert "findUserById" in tokens
    assert "findUserById".lower() in tokens
    assert "find" in tokens
    assert "User" in tokens
    assert "user" in tokens
    assert "By" in tokens
    assert "by" in tokens
    assert "Id" in tokens
    assert "id" in tokens


def test_code_tokenizer_dotted():
    tokens = CodeTokenizer.tokenize("java.util.List")
    assert "java" in tokens
    assert "util" in tokens
    assert "List" in tokens
    assert "list" in tokens


def test_bm25_searcher():
    chunk1 = Chunk(
        chunk_id="chunk_1",
        path=Path("src/user/UserService.java"),
        language="java",
        start_line=10,
        end_line=20,
        symbol="findUser",
        symbol_kind=ChunkKind.METHOD,
        code="public User findUser(Long id) { return repository.findById(id); }",
        parent_symbol="UserService",
        docstring="Finds a user by ID.",
    )

    chunk2 = Chunk(
        chunk_id="chunk_2",
        path=Path("src/user/UserRepository.java"),
        language="java",
        start_line=5,
        end_line=15,
        symbol="findById",
        symbol_kind=ChunkKind.METHOD,
        code="Optional<User> findById(Long id);",
        parent_symbol="UserRepository",
    )

    searcher = BM25Searcher([chunk1, chunk2])

    # Query matching chunk 1
    results = searcher.search("UserService findUser", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "chunk_1"

    # Query matching chunk 2
    results2 = searcher.search("UserRepository findById", top_k=2)
    assert len(results2) > 0
    assert results2[0].chunk.chunk_id == "chunk_2"
