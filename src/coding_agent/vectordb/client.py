"""LanceDB client for vector storage and similarity querying."""

from pathlib import Path
from typing import Any

import lancedb

from ..chunker.models import Chunk
from ..observability import MetricsCollector, get_metrics_collector
from .schema import chunk_to_record, get_chunk_schema


class LanceDBClient:
    """Manages LanceDB database connections, table schemas, and vector operations."""

    DEFAULT_TABLE_NAME = "chunks"

    def __init__(
        self,
        uri: str | Path = ".mini-agent/lancedb",
        dimension: int = 384,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize LanceDB client.

        Args:
            uri: Database storage path or URI.
            dimension: Dimensionality of the vector embeddings.
        """
        self.uri = str(uri)
        self.dimension = dimension
        self.metrics = metrics or get_metrics_collector()
        self._db: lancedb.DBConnection | None = None

    @property
    def db(self) -> lancedb.DBConnection:
        """Return active connection to LanceDB database."""
        if self._db is None:
            # Ensure parent directory exists for file path
            Path(self.uri).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self.uri)
        return self._db

    def get_table(self, table_name: str = DEFAULT_TABLE_NAME) -> Any:
        """Open existing LanceDB table or create it if missing.

        Args:
            table_name: Name of the vector table.

        Returns:
            LanceDB Table instance.
        """
        schema = get_chunk_schema(dimension=self.dimension)
        if table_name in self.db.table_names():
            return self.db.open_table(table_name)
        return self.db.create_table(table_name, schema=schema)

    def add_records(
        self,
        records: list[dict[str, Any]],
        table_name: str = DEFAULT_TABLE_NAME,
        mode: str = "append",
    ) -> None:
        """Insert records into the vector table.

        Args:
            records: List of dictionaries matching the chunk schema.
            table_name: Name of the destination table.
            mode: Write mode ('append' or 'overwrite').
        """
        if not records:
            return

        schema = get_chunk_schema(dimension=self.dimension)
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(records, mode=mode)
        else:
            self.db.create_table(table_name, data=records, schema=schema)

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        table_name: str = DEFAULT_TABLE_NAME,
        mode: str = "append",
    ) -> None:
        """Convert chunks and vectors into records and insert them into table.

        Args:
            chunks: List of Chunk objects.
            vectors: Corresponding embedding vectors.
            table_name: Destination table name.
            mode: Write mode ('append' or 'overwrite').
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(vectors)} vectors"
            )

        records = [chunk_to_record(c, v) for c, v in zip(chunks, vectors, strict=False)]
        self.add_records(records, table_name=table_name, mode=mode)

    def upsert_records(
        self,
        records: list[dict[str, Any]],
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        """Insert records or replace existing rows by deterministic chunk ID."""
        if not records:
            return

        schema = get_chunk_schema(dimension=self.dimension)
        if table_name not in self.db.table_names():
            with self.metrics.measure("vector.upsert"):
                self.db.create_table(table_name, data=records, schema=schema)
            return

        table = self.db.open_table(table_name)
        with self.metrics.measure("vector.upsert"):
            (
                table.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        """Upsert chunks and embeddings using each deterministic chunk ID."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(vectors)} vectors"
            )
        records = [
            chunk_to_record(chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.upsert_records(records, table_name=table_name)

    def delete_by_path(
        self,
        path: str | Path,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        """Delete every indexed chunk belonging to one repository-relative path."""
        if table_name not in self.db.table_names():
            return
        safe_path = str(path).replace("'", "''")
        self.db.open_table(table_name).delete(f"path = '{safe_path}'")

    def count(self, table_name: str = DEFAULT_TABLE_NAME) -> int:
        """Return total number of rows in the table."""
        with self.metrics.measure("vector.count"):
            if table_name not in self.db.table_names():
                return 0
            table = self.db.open_table(table_name)
            return len(table)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        table_name: str = DEFAULT_TABLE_NAME,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform nearest-neighbor vector search on the table.

        Args:
            query_vector: Embedding vector of the query.
            limit: Maximum number of nearest neighbors to return.
            table_name: Table to search within.
            where: Optional SQL filter condition.

        Returns:
            List of matching records with distance scores.
        """
        if table_name not in self.db.table_names():
            return []

        with self.metrics.measure("vector.search"):
            table = self.db.open_table(table_name)
            query = table.search(query_vector).limit(limit)
            if where:
                query = query.where(where)

            results = query.to_list()
        self.metrics.observe("vector.search.results", len(results))
        return results

    def drop_table(self, table_name: str = DEFAULT_TABLE_NAME) -> None:
        """Drop a table from the database."""
        if table_name in self.db.table_names():
            self.db.drop_table(table_name)
