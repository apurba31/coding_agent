"""In-memory symbol definitions and lexical reference lookup."""

import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from coding_agent.chunker.chunker import Chunker
from coding_agent.chunker.models import Chunk
from coding_agent.models.ast import SyntaxTree
from coding_agent.models.language import Language
from coding_agent.parser.engine import TreeSitterEngine
from coding_agent.scanner.scanner import RepositoryScanner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolLocation:
    """A definition location extracted from an AST-backed code chunk."""

    name: str
    qualified_name: str
    kind: str
    path: Path
    start_line: int
    end_line: int
    code: str


@dataclass(frozen=True)
class ReferenceLocation:
    """A source line containing a lexical occurrence of a symbol."""

    symbol: str
    path: Path
    line: int
    code: str
    is_definition: bool = False


class NavigationIndex:
    """Resolve definitions and search references from indexed chunks/source files.

    Definitions come from the AST-aware chunker. References use whole-identifier
    lexical matching: this is fast and language-neutral, but deliberately is not
    presented as compiler-accurate or equivalent to an LSP.
    """

    def __init__(self, root: str | Path, chunks: list[Chunk]) -> None:
        self.root = Path(root).resolve()
        definitions: dict[tuple[str, int, int, str], SymbolLocation] = {}
        for chunk in chunks:
            key = (
                chunk.path.as_posix(),
                chunk.start_line,
                chunk.end_line,
                chunk.symbol,
            )
            candidate = SymbolLocation(
                name=chunk.symbol,
                qualified_name=chunk.display_name,
                kind=chunk.symbol_kind.value,
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                code=chunk.code,
            )
            existing = definitions.get(key)
            if existing is None or (chunk.parent_symbol and "." not in existing.qualified_name):
                definitions[key] = candidate
        # Some language chunkers can visit the same AST node through multiple
        # recursive paths; collapse identical source locations at the boundary.
        self._definitions = list(definitions.values())
        self._definitions.sort(
            key=lambda item: (
                item.qualified_name.casefold(),
                item.path.as_posix().casefold(),
                item.start_line,
            )
        )

    @classmethod
    def from_repository(cls, root: str | Path) -> "NavigationIndex":
        """Build a symbol index without loading an embedding model or vector DB."""
        root_path = Path(root).resolve()
        scanner = RepositoryScanner(root_path)
        parser = TreeSitterEngine()
        chunker = Chunker()
        supported = {
            Language.PYTHON,
            Language.JAVA,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
        }
        chunks: list[Chunk] = []
        for file in scanner.scan(root_path).files:
            if file.language not in supported:
                continue
            try:
                source_bytes = file.absolute_path.read_bytes()
                tree = parser.parse(file.absolute_path, file.language.value.lower())
                syntax_tree = SyntaxTree(
                    language=file.language.value,
                    root=tree.root_node,
                    source=source_bytes,
                )
                chunks.extend(chunker.chunk(file, syntax_tree, source_bytes.decode("utf-8")))
            except (OSError, UnicodeError, ValueError, RuntimeError):
                # Navigation should remain useful if an individual file is unreadable.
                logger.warning("Unable to index navigation symbols in %s", file.path)
                continue
            except Exception:
                logger.exception("Failed to index navigation symbols in %s", file.path)
                continue
        return cls(root_path, chunks)

    def find_definition(self, symbol: str) -> list[SymbolLocation]:
        """Return exact-name or qualified-name definitions, sorted by path/line."""
        query = symbol.strip().casefold()
        if not query:
            return []
        return [
            definition
            for definition in self._definitions
            if definition.name.casefold() == query or definition.qualified_name.casefold() == query
        ]

    def search_symbols(self, query: str, limit: int = 20) -> list[SymbolLocation]:
        """Find definitions by substring or identifier-token matches."""
        if limit < 1:
            return []
        terms = _identifier_terms(query)
        if not terms:
            return []

        ranked: list[tuple[int, SymbolLocation]] = []
        for definition in self._definitions:
            name = definition.name.casefold()
            qualified = definition.qualified_name.casefold()
            tokens = _identifier_terms(qualified)
            overlap = len(terms & tokens)
            if not overlap and not any(term in qualified for term in terms):
                continue
            score = overlap * 10
            if query.casefold() == qualified:
                score += 100
            elif query.casefold() == name:
                score += 80
            elif query.casefold() in qualified:
                score += 30
            ranked.append((score, definition))

        ranked.sort(
            key=lambda pair: (
                -pair[0],
                pair[1].qualified_name.casefold(),
                pair[1].path.as_posix().casefold(),
                pair[1].start_line,
            )
        )
        return [definition for _, definition in ranked[:limit]]

    def find_references(self, symbol: str, limit: int = 100) -> list[ReferenceLocation]:
        """Find whole-identifier occurrences in supported source files.

        Occurrences on a definition's first line are marked as definitions. Files
        are walked under the repository root while pruning common generated dirs.
        """
        name = symbol.rsplit(".", 1)[-1].strip()
        if not name or limit < 1:
            return []
        pattern = re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])")
        definition_lines = {
            (item.path.as_posix(), item.start_line) for item in self.find_definition(symbol)
        }
        results: list[ReferenceLocation] = []
        for source_path in self._source_files():
            relative_path = source_path.relative_to(self.root)
            try:
                lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if pattern.search(line):
                    relative = relative_path.as_posix()
                    results.append(
                        ReferenceLocation(
                            symbol=name,
                            path=relative_path,
                            line=line_number,
                            code=line.strip(),
                            is_definition=(relative, line_number) in definition_lines,
                        )
                    )
                    if len(results) >= limit:
                        return results
        return results

    def _source_files(self) -> Iterator[Path]:
        ignored_directories = {
            ".git",
            ".mini-agent",
            ".venv",
            "venv",
            "node_modules",
            "build",
            "dist",
            "target",
            "__pycache__",
        }
        supported_suffixes = {".py", ".java", ".js", ".ts"}
        for current, directories, files in os.walk(self.root):
            directories[:] = sorted(
                directory for directory in directories if directory not in ignored_directories
            )
            for filename in sorted(files):
                path = Path(current) / filename
                try:
                    path.resolve().relative_to(self.root)
                except ValueError:
                    # A symlink must not let a repository query read outside its root.
                    continue
                if path.suffix.lower() in supported_suffixes:
                    yield path


def _identifier_terms(value: str) -> set[str]:
    """Split a query/symbol into case-folded identifier terms."""
    return {part.casefold() for part in re.findall(r"[A-Za-z0-9]+", value) if part}
