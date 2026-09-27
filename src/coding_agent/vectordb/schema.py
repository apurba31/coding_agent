"""Schema definitions and converters for storing code chunks in LanceDB."""

from pathlib import Path
from typing import Any

import pyarrow as pa

from ..chunker.models import Chunk, ChunkKind


def get_chunk_schema(dimension: int = 384) -> pa.Schema:
    """Return the PyArrow schema for code chunk vector table.

    Args:
        dimension: Embedding vector length.

    Returns:
        PyArrow Schema for chunk storage.
    """
    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("path", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("start_line", pa.int64(), nullable=False),
            pa.field("end_line", pa.int64(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("symbol_kind", pa.string(), nullable=False),
            pa.field("parent_symbol", pa.string(), nullable=True),
            pa.field("code", pa.string(), nullable=False),
            pa.field("docstring", pa.string(), nullable=True),
            pa.field("embedding_text", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False),
        ]
    )


def chunk_to_record(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
    """Convert a Chunk domain model and its embedding vector to a LanceDB record dict.

    Args:
        chunk: Chunk object from chunker.
        vector: Embedding vector.

    Returns:
        Dictionary matching the chunk PyArrow schema.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "path": chunk.path.as_posix(),
        "language": chunk.language,
        "start_line": int(chunk.start_line),
        "end_line": int(chunk.end_line),
        "symbol": chunk.symbol,
        "symbol_kind": (
            chunk.symbol_kind.value
            if isinstance(chunk.symbol_kind, ChunkKind)
            else str(chunk.symbol_kind)
        ),
        "parent_symbol": chunk.parent_symbol if chunk.parent_symbol else None,
        "code": chunk.code,
        "docstring": chunk.docstring if chunk.docstring else None,
        "embedding_text": chunk.to_embedding_text(),
        "vector": [float(x) for x in vector],
    }


def record_to_chunk(record: dict[str, Any]) -> Chunk:
    """Convert a LanceDB record dict back into a Chunk domain model.

    Args:
        record: Dictionary fetched from LanceDB.

    Returns:
        Chunk object representing the original code chunk.
    """
    try:
        symbol_kind = ChunkKind(record["symbol_kind"])
    except ValueError:
        # Fallback if somehow invalid
        symbol_kind = ChunkKind.FUNCTION

    return Chunk(
        chunk_id=record["chunk_id"],
        path=Path(record["path"]),
        language=record["language"],
        start_line=record["start_line"],
        end_line=record["end_line"],
        symbol=record["symbol"],
        symbol_kind=symbol_kind,
        code=record["code"],
        parent_symbol=record.get("parent_symbol"),
        docstring=record.get("docstring"),
    )
