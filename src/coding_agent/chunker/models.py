"""Data models for code chunks."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ChunkKind(StrEnum):
    """Types of code chunks."""

    CLASS = "class"
    INTERFACE = "interface"
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    FIELD = "field"
    ENUM = "enum"
    MODULE = "module"
    IMPORT = "import"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PROPERTY = "property"


@dataclass(frozen=True)
class Chunk:
    """Represents a semantically meaningful chunk of source code."""

    # Identity
    chunk_id: str
    path: Path
    language: str

    # Location
    start_line: int
    end_line: int

    # Symbol information
    symbol: str
    symbol_kind: ChunkKind

    # Content
    code: str

    # Optional fields with defaults
    parent_symbol: str | None = None
    imports: list[str] = None
    docstring: str | None = None
    comments: list[str] = None
    is_public: bool = True
    decorators: list[str] = None
    type_hints: dict[str, str] = None

    def __post_init__(self):
        """Initialize list fields if None."""
        if self.imports is None:
            object.__setattr__(self, "imports", [])
        if self.comments is None:
            object.__setattr__(self, "comments", [])
        if self.decorators is None:
            object.__setattr__(self, "decorators", [])
        if self.type_hints is None:
            object.__setattr__(self, "type_hints", {})

    def to_embedding_text(self) -> str:
        """Convert chunk to text suitable for embedding.

        Combines metadata with code for semantic understanding.
        """
        lines = [
            f"Language: {self.language}",
            f"Path: {self.path}",
            f"Kind: {self.symbol_kind.value}",
            f"Symbol: {self.symbol}",
        ]

        if self.parent_symbol:
            lines.append(f"Parent: {self.parent_symbol}")

        if self.decorators:
            lines.append(f"Decorators: {', '.join(self.decorators)}")

        lines.append("")
        lines.append(self.code)

        if self.docstring:
            lines.append("")
            lines.append("Docstring:")
            lines.append(self.docstring)

        return "\n".join(lines)

    @property
    def display_name(self) -> str:
        """Human-readable identifier for the chunk."""
        if self.parent_symbol:
            return f"{self.parent_symbol}.{self.symbol}"
        return self.symbol
