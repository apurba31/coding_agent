"""Tests for AST-aware chunking functionality."""

import pytest
from pathlib import Path
from datetime import datetime

from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.chunker.chunker import (
    Chunker,
    PythonChunker,
    JavaChunker,
    JavaScriptChunker,
)
from coding_agent.models.file import FileMetadata
from coding_agent.models.language import Language
from coding_agent.models.ast import SyntaxTree


class TestChunkModel:
    """Tests for Chunk data model."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            chunk_id="abc123",
            path=Path("test.py"),
            language="python",
            start_line=1,
            end_line=10,
            symbol="my_function",
            symbol_kind=ChunkKind.FUNCTION,
            code="def my_function():\n    pass",
        )

        assert chunk.chunk_id == "abc123"
        assert chunk.path == Path("test.py")
        assert chunk.symbol == "my_function"
        assert chunk.symbol_kind == ChunkKind.FUNCTION

    def test_chunk_with_parent_symbol(self):
        """Test chunk with parent symbol (method in class)."""
        chunk = Chunk(
            chunk_id="abc123",
            path=Path("test.py"),
            language="python",
            start_line=5,
            end_line=10,
            symbol="method",
            symbol_kind=ChunkKind.METHOD,
            parent_symbol="MyClass",
            code="    def method(self):\n        pass",
        )

        assert chunk.parent_symbol == "MyClass"
        assert chunk.display_name == "MyClass.method"

    def test_chunk_display_name_without_parent(self):
        """Test display name for chunks without parent."""
        chunk = Chunk(
            chunk_id="abc123",
            path=Path("test.py"),
            language="python",
            start_line=1,
            end_line=5,
            symbol="standalone_function",
            symbol_kind=ChunkKind.FUNCTION,
            code="def standalone_function():\n    pass",
        )

        assert chunk.display_name == "standalone_function"

    def test_chunk_embedding_text(self):
        """Test conversion to embedding text."""
        chunk = Chunk(
            chunk_id="abc123",
            path=Path("example.py"),
            language="python",
            start_line=1,
            end_line=3,
            symbol="greet",
            symbol_kind=ChunkKind.FUNCTION,
            code="def greet(name):\n    return f'Hello {name}'",
            docstring="Greeting function",
            decorators=["@cached"],
        )

        embedding_text = chunk.to_embedding_text()

        assert "Language: python" in embedding_text
        assert "Path: example.py" in embedding_text
        assert "Symbol: greet" in embedding_text
        assert "Decorators: @cached" in embedding_text
        assert "Docstring:" in embedding_text
        assert "Greeting function" in embedding_text
        assert "def greet" in embedding_text

    def test_chunk_is_public(self):
        """Test public/private visibility."""
        public_chunk = Chunk(
            chunk_id="abc",
            path=Path("test.py"),
            language="python",
            start_line=1,
            end_line=3,
            symbol="public_func",
            symbol_kind=ChunkKind.FUNCTION,
            code="def public_func():\n    pass",
            is_public=True,
        )

        private_chunk = Chunk(
            chunk_id="xyz",
            path=Path("test.py"),
            language="python",
            start_line=5,
            end_line=7,
            symbol="_private_func",
            symbol_kind=ChunkKind.FUNCTION,
            code="def _private_func():\n    pass",
            is_public=False,
        )

        assert public_chunk.is_public
        assert not private_chunk.is_public


class TestPythonChunker:
    """Tests for Python-specific chunking."""

    def test_chunk_id_determinism(self):
        """Test that chunk IDs are stable and deterministic."""
        chunker = PythonChunker()
        path = Path("example.py")

        # Same inputs should produce same ID
        id1 = chunker._generate_chunk_id(path, "my_func", 10)
        id2 = chunker._generate_chunk_id(path, "my_func", 10)
        assert id1 == id2

        # Different inputs should produce different IDs
        id3 = chunker._generate_chunk_id(path, "other_func", 10)
        assert id1 != id3

    def test_token_estimation(self):
        """Test rough token counting."""
        chunker = PythonChunker()

        # Approximately 1 token per 4 characters
        text = "x" * 400  # 400 chars
        tokens = chunker._estimate_tokens(text)
        assert tokens == 100

    def test_line_extraction(self):
        """Test extraction of source lines."""
        chunker = PythonChunker()
        source_lines = [
            "def foo():",
            "    x = 1",
            "    y = 2",
            "    return x + y",
        ]

        # Extract lines 1-4 (1-indexed)
        code = chunker._extract_lines(source_lines, 1, 4)
        assert "def foo():" in code
        assert "return x + y" in code

    def test_line_extraction_boundary(self):
        """Test line extraction at boundaries."""
        chunker = PythonChunker()
        source_lines = ["line1", "line2", "line3"]

        # Request lines beyond file length
        code = chunker._extract_lines(source_lines, 1, 10)
        assert "line1" in code
        assert "line3" in code

        # Request lines before start
        code = chunker._extract_lines(source_lines, 0, 2)
        assert "line1" in code


class TestJavaChunker:
    """Tests for Java-specific chunking."""

    def test_java_chunker_creation(self):
        """Test Java chunker instantiation."""
        chunker = JavaChunker(token_budget=4000)
        assert chunker.token_budget == 4000


class TestJavaScriptChunker:
    """Tests for JavaScript-specific chunking."""

    def test_js_chunker_creation(self):
        """Test JavaScript chunker instantiation."""
        chunker = JavaScriptChunker(token_budget=4000)
        assert chunker.token_budget == 4000


class TestChunkerRouter:
    """Tests for the main Chunker router."""

    def test_chunker_initialization(self):
        """Test Chunker with custom token budget."""
        chunker = Chunker(token_budget=2000)
        assert chunker.token_budget == 2000

    def test_supported_languages(self):
        """Test that chunker has support for required languages."""
        chunker = Chunker()

        assert "python" in chunker._chunkers
        assert "java" in chunker._chunkers
        assert "javascript" in chunker._chunkers
        assert "typescript" in chunker._chunkers

    def test_unsupported_language_raises_error(self):
        """Test that unsupported languages raise ValueError."""
        chunker = Chunker()

        file = FileMetadata(
            path=Path("test.unknown"),
            absolute_path=Path("/test.unknown"),
            extension=".unknown",
            language=Language.UNKNOWN,
            size=100,
            last_modified=datetime.now(),
            is_binary=False,
            sha256="abc123",
        )

        tree = SyntaxTree(language="unknown", root=None, source=b"")

        with pytest.raises(ValueError, match="Unsupported language"):
            chunker.chunk(file, tree, "")


class TestChunkKind:
    """Tests for ChunkKind enum."""

    def test_all_chunk_kinds_defined(self):
        """Verify all symbol kinds are defined."""
        kinds = [
            ChunkKind.CLASS,
            ChunkKind.INTERFACE,
            ChunkKind.FUNCTION,
            ChunkKind.METHOD,
            ChunkKind.CONSTRUCTOR,
            ChunkKind.FIELD,
            ChunkKind.ENUM,
            ChunkKind.MODULE,
            ChunkKind.IMPORT,
            ChunkKind.VARIABLE,
            ChunkKind.CONSTANT,
            ChunkKind.PROPERTY,
        ]

        assert len(kinds) == 12
        assert all(isinstance(k, ChunkKind) for k in kinds)

    def test_chunk_kind_string_values(self):
        """Test ChunkKind string representation."""
        assert ChunkKind.CLASS.value == "class"
        assert ChunkKind.FUNCTION.value == "function"
        assert ChunkKind.METHOD.value == "method"


class TestChunkIntegration:
    """Integration tests combining chunker components."""

    def test_chunk_metadata_preservation(self):
        """Test that chunk metadata is preserved correctly."""
        chunk = Chunk(
            chunk_id="test_id",
            path=Path("module/submodule/file.py"),
            language="python",
            start_line=15,
            end_line=25,
            symbol="calculate",
            symbol_kind=ChunkKind.FUNCTION,
            parent_symbol=None,
            code="def calculate(x, y):\n    return x + y",
            imports=["import math"],
            docstring="Calculates sum",
            comments=["# Helper function"],
            is_public=True,
            decorators=["@staticmethod"],
            type_hints={"x": "int", "y": "int", "return": "int"},
        )

        assert chunk.path == Path("module/submodule/file.py")
        assert chunk.start_line == 15
        assert chunk.end_line == 25
        assert len(chunk.imports) == 1
        assert len(chunk.decorators) == 1
        assert chunk.type_hints["return"] == "int"

    def test_chunk_list_fields_initialization(self):
        """Test that list fields are properly initialized."""
        chunk = Chunk(
            chunk_id="test",
            path=Path("test.py"),
            language="python",
            start_line=1,
            end_line=5,
            symbol="func",
            symbol_kind=ChunkKind.FUNCTION,
            code="def func():\n    pass",
        )

        assert isinstance(chunk.imports, list)
        assert isinstance(chunk.comments, list)
        assert isinstance(chunk.decorators, list)
        assert isinstance(chunk.type_hints, dict)
        assert len(chunk.imports) == 0


class TestChunkEdgeCases:
    """Test edge cases and error handling."""

    def test_chunk_with_empty_code(self):
        """Test chunk with empty code."""
        chunk = Chunk(
            chunk_id="empty",
            path=Path("test.py"),
            language="python",
            start_line=1,
            end_line=1,
            symbol="empty_func",
            symbol_kind=ChunkKind.FUNCTION,
            code="",
        )

        assert chunk.code == ""
        assert chunk.display_name == "empty_func"

    def test_chunk_with_multiline_code(self):
        """Test chunk with complex multiline code."""
        code = """class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
"""
        chunk = Chunk(
            chunk_id="calc",
            path=Path("calc.py"),
            language="python",
            start_line=1,
            end_line=6,
            symbol="Calculator",
            symbol_kind=ChunkKind.CLASS,
            code=code,
        )

        assert "def add" in chunk.code
        assert "def multiply" in chunk.code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
