"""Core chunking logic for AST-aware code segmentation."""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from coding_agent.models.ast import SyntaxTree
from coding_agent.models.file import FileMetadata

from .models import Chunk, ChunkKind


class LanguageChunker(ABC):
    """Abstract base class for language-specific chunking."""

    def __init__(self, token_budget: int = 4000):
        """Initialize chunker.

        Args:
            token_budget: Max tokens per chunk (approximate).
        """
        self.token_budget = token_budget

    @abstractmethod
    def chunk(
        self,
        file: FileMetadata,
        tree: SyntaxTree,
        source_lines: list[str],
    ) -> list[Chunk]:
        """Extract chunks from a parsed file.

        Args:
            file: File metadata.
            tree: Parsed syntax tree.
            source_lines: Original source lines (for line extraction).

        Returns:
            List of chunks.
        """
        raise NotImplementedError

    def _extract_lines(
        self,
        source_lines: list[str],
        start_line: int,
        end_line: int,
    ) -> str:
        """Extract source code lines by line numbers.

        Args:
            source_lines: All source lines (0-indexed).
            start_line: Start line (1-indexed).
            end_line: End line (1-indexed, inclusive).

        Returns:
            Extracted source code.
        """
        start_idx = max(0, start_line - 1)
        end_idx = min(len(source_lines), end_line)
        return "\n".join(source_lines[start_idx:end_idx])

    def _generate_chunk_id(
        self,
        path: Path,
        symbol: str,
        start_line: int,
    ) -> str:
        """Generate deterministic chunk ID.

        Args:
            path: File path.
            symbol: Symbol name.
            start_line: Start line number.

        Returns:
            Stable chunk ID.
        """
        identifier = f"{path}:{symbol}:{start_line}"
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (1 token ≈ 4 chars).

        Args:
            text: Text to estimate.

        Returns:
            Approximate token count.
        """
        return len(text) // 4


class PythonChunker(LanguageChunker):
    """Chunker for Python source code."""

    def chunk(
        self,
        file: FileMetadata,
        tree: SyntaxTree,
        source_lines: list[str],
    ) -> list[Chunk]:
        """Extract chunks from Python AST.

        Args:
            file: File metadata.
            tree: Parsed syntax tree.
            source_lines: Original source lines.

        Returns:
            List of Python code chunks.
        """
        chunks = []
        self._walk_python_tree(tree.root, file, source_lines, chunks)
        return chunks

    def _walk_python_tree(
        self,
        node,
        file: FileMetadata,
        source_lines: list[str],
        chunks: list[Chunk],
        parent_symbol: str | None = None,
    ) -> None:
        """Recursively walk Python AST and extract chunks.

        Args:
            node: AST node.
            file: File metadata.
            source_lines: Original source lines.
            chunks: Accumulator for chunks.
            parent_symbol: Parent symbol name.
        """
        if not hasattr(node, "type"):
            return

        node_type = node.type

        # Extract class definitions
        if node_type == "class_definition":
            class_name = self._get_python_class_name(node)
            if class_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)
                docstring = self._extract_docstring(node)

                chunk_id = self._generate_chunk_id(file.path, class_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=class_name,
                        symbol_kind=ChunkKind.CLASS,
                        parent_symbol=parent_symbol,
                        code=code,
                        docstring=docstring,
                        is_public=not class_name.startswith("_"),
                    )
                )

            # Walk class body
            for child in node.children:
                if child.type == "block":
                    for inner in child.children:
                        self._walk_python_tree(
                            inner,
                            file,
                            source_lines,
                            chunks,
                            parent_symbol=class_name or parent_symbol,
                        )

        # Extract function/method definitions
        elif node_type == "function_definition":
            func_name = self._get_python_func_name(node)
            if func_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)
                docstring = self._extract_docstring(node)
                decorators = self._extract_python_decorators(node)

                is_method = parent_symbol is not None
                kind = ChunkKind.METHOD if is_method else ChunkKind.FUNCTION

                chunk_id = self._generate_chunk_id(file.path, func_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=func_name,
                        symbol_kind=kind,
                        parent_symbol=parent_symbol,
                        code=code,
                        docstring=docstring,
                        decorators=decorators,
                        is_public=not func_name.startswith("_"),
                    )
                )

        # Continue walking
        for child in node.children:
            self._walk_python_tree(child, file, source_lines, chunks, parent_symbol)

    @staticmethod
    def _get_python_class_name(node) -> str | None:
        """Extract class name from class_definition node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _get_python_func_name(node) -> str | None:
        """Extract function name from function_definition node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _extract_docstring(node) -> str | None:
        """Extract docstring from function/class body."""
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for subchild in stmt.children:
                            if subchild.type == "string":
                                text = subchild.text.decode("utf-8")
                                # Remove quotes
                                if text.startswith('"""') or text.startswith("'''"):
                                    return text[3:-3].strip()
                                elif text.startswith('"') or text.startswith("'"):
                                    return text[1:-1].strip()
                                return text.strip()
        return None

    @staticmethod
    def _extract_python_decorators(node) -> list[str]:
        """Extract decorator names from function definition."""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                text = child.text.decode("utf-8").strip()
                if text.startswith("@"):
                    text = text[1:]
                decorators.append(text)
        return decorators


class JavaChunker(LanguageChunker):
    """Chunker for Java source code."""

    def chunk(
        self,
        file: FileMetadata,
        tree: SyntaxTree,
        source_lines: list[str],
    ) -> list[Chunk]:
        """Extract chunks from Java AST.

        Args:
            file: File metadata.
            tree: Parsed syntax tree.
            source_lines: Original source lines.

        Returns:
            List of Java code chunks.
        """
        chunks = []
        self._walk_java_tree(tree.root, file, source_lines, chunks)
        return chunks

    def _walk_java_tree(
        self,
        node,
        file: FileMetadata,
        source_lines: list[str],
        chunks: list[Chunk],
        parent_symbol: str | None = None,
    ) -> None:
        """Recursively walk Java AST and extract chunks.

        Args:
            node: AST node.
            file: File metadata.
            source_lines: Original source lines.
            chunks: Accumulator for chunks.
            parent_symbol: Parent symbol name.
        """
        if not hasattr(node, "type"):
            return

        node_type = node.type

        # Extract class definitions
        if node_type in ("class_declaration", "interface_declaration"):
            class_name = self._get_java_class_name(node)
            if class_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)

                kind = ChunkKind.CLASS if node_type == "class_declaration" else ChunkKind.INTERFACE

                chunk_id = self._generate_chunk_id(file.path, class_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=class_name,
                        symbol_kind=kind,
                        parent_symbol=parent_symbol,
                        code=code,
                        is_public=self._is_java_public(node),
                    )
                )

            # Walk class body
            for child in node.children:
                if child.type == "class_body":
                    for inner in child.children:
                        self._walk_java_tree(
                            inner,
                            file,
                            source_lines,
                            chunks,
                            parent_symbol=class_name or parent_symbol,
                        )

        # Extract method definitions
        elif node_type == "method_declaration":
            method_name = self._get_java_method_name(node)
            if method_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)

                chunk_id = self._generate_chunk_id(file.path, method_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=method_name,
                        symbol_kind=ChunkKind.METHOD,
                        parent_symbol=parent_symbol,
                        code=code,
                        is_public=self._is_java_public(node),
                    )
                )

        # Continue walking
        for child in node.children:
            self._walk_java_tree(child, file, source_lines, chunks, parent_symbol)

    @staticmethod
    def _get_java_class_name(node) -> str | None:
        """Extract class name from class/interface declaration."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _get_java_method_name(node) -> str | None:
        """Extract method name from method_declaration node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _is_java_public(node) -> bool:
        """Check if Java element has public modifier."""
        for child in node.children:
            if child.type == "modifier" and child.text == b"public":
                return True
        return False


class GoChunker(LanguageChunker):
    """Chunker for Go source code."""

    def chunk(
        self,
        file: FileMetadata,
        tree: SyntaxTree,
        source_lines: list[str],
    ) -> list[Chunk]:
        chunks = []
        self._walk_go_tree(tree.root, file, source_lines, chunks)
        return chunks

    def _walk_go_tree(
        self,
        node,
        file: FileMetadata,
        source_lines: list[str],
        chunks: list[Chunk],
        parent_symbol: str | None = None,
    ) -> None:
        if not hasattr(node, "type"):
            return

        node_type = node.type

        if node_type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name = self._get_go_type_name(child)
                    if name:
                        start_line = node.start_point[0] + 1
                        end_line = node.end_point[0] + 1
                        code = self._extract_lines(source_lines, start_line, end_line)
                        chunk_id = self._generate_chunk_id(file.path, name, start_line)
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                path=file.path,
                                language=file.language.value,
                                start_line=start_line,
                                end_line=end_line,
                                symbol=name,
                                symbol_kind=ChunkKind.CLASS,
                                parent_symbol=parent_symbol,
                                code=code,
                            )
                        )
                        parent_symbol = name

        elif node_type == "function_declaration":
            func_name = self._get_go_func_name(node)
            if func_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)
                chunk_id = self._generate_chunk_id(file.path, func_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=func_name,
                        symbol_kind=ChunkKind.FUNCTION,
                        parent_symbol=parent_symbol,
                        code=code,
                    )
                )

        for child in node.children:
            self._walk_go_tree(child, file, source_lines, chunks, parent_symbol)

    @staticmethod
    def _get_go_func_name(node) -> str | None:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _get_go_type_name(node) -> str | None:
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8")
        return None


class RustChunker(LanguageChunker):
    """Chunker for Rust source code."""

    def chunk(
        self,
        file: FileMetadata,
        tree: SyntaxTree,
        source_lines: list[str],
    ) -> list[Chunk]:
        chunks = []
        self._walk_rust_tree(tree.root, file, source_lines, chunks)
        return chunks

    def _walk_rust_tree(
        self,
        node,
        file: FileMetadata,
        source_lines: list[str],
        chunks: list[Chunk],
        parent_symbol: str | None = None,
    ) -> None:
        if not hasattr(node, "type"):
            return

        node_type = node.type

        if node_type == "struct_item":
            struct_name = self._get_rust_type_name(node)
            if struct_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)
                chunk_id = self._generate_chunk_id(file.path, struct_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=struct_name,
                        symbol_kind=ChunkKind.CLASS,
                        parent_symbol=parent_symbol,
                        code=code,
                    )
                )
                parent_symbol = struct_name

        elif node_type == "function_item":
            func_name = self._get_rust_func_name(node)
            if func_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)
                chunk_id = self._generate_chunk_id(file.path, func_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=func_name,
                        symbol_kind=ChunkKind.FUNCTION,
                        parent_symbol=parent_symbol,
                        code=code,
                    )
                )

        for child in node.children:
            self._walk_rust_tree(child, file, source_lines, chunks, parent_symbol)

    @staticmethod
    def _get_rust_func_name(node) -> str | None:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _get_rust_type_name(node) -> str | None:
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8")
        return None


class JavaScriptChunker(LanguageChunker):
    """Chunker for JavaScript source code."""

    def chunk(
        self,
        file: FileMetadata,
        tree: SyntaxTree,
        source_lines: list[str],
    ) -> list[Chunk]:
        """Extract chunks from JavaScript AST.

        Args:
            file: File metadata.
            tree: Parsed syntax tree.
            source_lines: Original source lines.

        Returns:
            List of JavaScript code chunks.
        """
        chunks = []
        self._walk_js_tree(tree.root, file, source_lines, chunks)
        return chunks

    def _walk_js_tree(
        self,
        node,
        file: FileMetadata,
        source_lines: list[str],
        chunks: list[Chunk],
        parent_symbol: str | None = None,
    ) -> None:
        """Recursively walk JavaScript AST and extract chunks.

        Args:
            node: AST node.
            file: File metadata.
            source_lines: Original source lines.
            chunks: Accumulator for chunks.
            parent_symbol: Parent symbol name.
        """
        if not hasattr(node, "type"):
            return

        node_type = node.type

        # Extract class declarations
        if node_type == "class_declaration":
            class_name = self._get_js_class_name(node)
            if class_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)

                chunk_id = self._generate_chunk_id(file.path, class_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=class_name,
                        symbol_kind=ChunkKind.CLASS,
                        parent_symbol=parent_symbol,
                        code=code,
                    )
                )

            # Walk class body
            for child in node.children:
                if child.type == "class_body":
                    for inner in child.children:
                        self._walk_js_tree(
                            inner,
                            file,
                            source_lines,
                            chunks,
                            parent_symbol=class_name or parent_symbol,
                        )

        # Extract function declarations
        elif node_type == "function_declaration":
            func_name = self._get_js_func_name(node)
            if func_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)

                chunk_id = self._generate_chunk_id(file.path, func_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=func_name,
                        symbol_kind=ChunkKind.FUNCTION,
                        parent_symbol=parent_symbol,
                        code=code,
                    )
                )

        # Extract method definitions
        elif node_type == "method_definition":
            method_name = self._get_js_method_name(node)
            if method_name:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                code = self._extract_lines(source_lines, start_line, end_line)

                chunk_id = self._generate_chunk_id(file.path, method_name, start_line)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=file.path,
                        language=file.language.value,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=method_name,
                        symbol_kind=ChunkKind.METHOD,
                        parent_symbol=parent_symbol,
                        code=code,
                    )
                )

        # Continue walking
        for child in node.children:
            self._walk_js_tree(child, file, source_lines, chunks, parent_symbol)

    @staticmethod
    def _get_js_class_name(node) -> str | None:
        """Extract class name from class_declaration."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _get_js_func_name(node) -> str | None:
        """Extract function name from function_declaration."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    @staticmethod
    def _get_js_method_name(node) -> str | None:
        """Extract method name from method_definition."""
        for child in node.children:
            if child.type == "property_identifier":
                return child.text.decode("utf-8")
        return None


class Chunker:
    """Router for language-specific chunking.

    Delegates to appropriate language chunker based on file type.
    """

    def __init__(self, token_budget: int = 4000):
        """Initialize chunker.

        Args:
            token_budget: Max tokens per chunk.
        """
        self.token_budget = token_budget
        self._chunkers = {
            "python": PythonChunker(token_budget),
            "java": JavaChunker(token_budget),
            "javascript": JavaScriptChunker(token_budget),
            "typescript": JavaScriptChunker(token_budget),
            "go": GoChunker(token_budget),
            "rust": RustChunker(token_budget),
        }

    def chunk(
        self,
        file: FileMetadata,
        tree: SyntaxTree,
        source: str,
    ) -> list[Chunk]:
        """Chunk a parsed source file.

        Args:
            file: File metadata.
            tree: Parsed syntax tree.
            source: Original source code.

        Returns:
            List of chunks.

        Raises:
            ValueError: If language is not supported.
        """
        lang = file.language.value.lower()
        if lang not in self._chunkers:
            raise ValueError(f"Unsupported language: {lang}")

        chunker = self._chunkers[lang]
        source_lines = source.splitlines()
        return chunker.chunk(file, tree, source_lines)
