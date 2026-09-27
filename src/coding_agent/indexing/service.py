"""Build and persist code chunks for a repository."""

import logging
from dataclasses import dataclass
from pathlib import Path

from coding_agent.chunker.chunker import Chunker
from coding_agent.chunker.models import Chunk
from coding_agent.embedding.embedder import Embedder
from coding_agent.models.ast import SyntaxTree
from coding_agent.models.file import FileMetadata
from coding_agent.models.language import Language
from coding_agent.observability import MetricsCollector, get_metrics_collector
from coding_agent.parser.engine import TreeSitterEngine
from coding_agent.scanner.scanner import RepositoryScanner
from coding_agent.vectordb.client import LanceDBClient

from .manifest import IndexManifest, IndexManifestStore

logger = logging.getLogger(__name__)
SUPPORTED_LANGUAGES = {Language.PYTHON, Language.JAVA, Language.JAVASCRIPT, Language.TYPESCRIPT}


@dataclass(frozen=True)
class IndexSummary:
    """Counts from one repository indexing run."""

    scanned_files: int
    parsed_files: int
    chunks: int
    failed_files: int
    unchanged_files: int = 0
    deleted_files: int = 0


class RepositoryIndexer:
    """Scan source files, extract AST chunks, embed them, and store vectors."""

    def __init__(
        self,
        embedder: Embedder,
        db_client: LanceDBClient,
        scanner: RepositoryScanner | None = None,
        parser: TreeSitterEngine | None = None,
        chunker: Chunker | None = None,
        batch_size: int = 32,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.embedder = embedder
        self.db_client = db_client
        self.metrics = metrics or get_metrics_collector()
        self.scanner = scanner or RepositoryScanner(Path("."), metrics=self.metrics)
        self.parser = parser or TreeSitterEngine()
        self.chunker = chunker or Chunker()
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        manifest_path = Path(self.db_client.uri) / "index_manifest.json"
        self.manifest_store = IndexManifestStore(manifest_path)

    def _parse_files(self, files: list[FileMetadata]) -> tuple[list[Chunk], set[str], set[str]]:
        """Parse selected files, returning chunks, successful paths, and failures."""
        chunks: list[Chunk] = []
        parsed_paths: set[str] = set()
        failed_paths: set[str] = set()

        for file in files:
            relative_path = file.path.as_posix()
            try:
                source_bytes = file.absolute_path.read_bytes()
                source = source_bytes.decode("utf-8")
                with self.metrics.measure("parse"):
                    tree = self.parser.parse(file.absolute_path, file.language.value.lower())
                syntax_tree = SyntaxTree(
                    language=file.language.value,
                    root=tree.root_node,
                    source=source_bytes,
                )
                with self.metrics.measure("chunk"):
                    file_chunks = self.chunker.chunk(file, syntax_tree, source)
                parsed_paths.add(relative_path)
                chunks.extend(file_chunks)
                self.metrics.increment("files.parsed")
                self.metrics.increment("chunks.created", len(file_chunks))
            except (OSError, UnicodeError, ValueError, RuntimeError) as error:
                failed_paths.add(relative_path)
                self.metrics.increment("files.failed")
                logger.warning("Skipping %s: %s", file.path, error)
            except Exception:
                failed_paths.add(relative_path)
                self.metrics.increment("files.failed")
                logger.exception("Failed to parse %s", file.path)

        return chunks, parsed_paths, failed_paths

    def collect_chunks(self, root: str | Path) -> tuple[list[Chunk], int, int, int]:
        """Return chunks and counts for scanned, parsed, and failed files."""
        root = Path(root).resolve()
        repository = self.scanner.scan(root)
        supported_files = [
            file for file in repository.files if file.language in SUPPORTED_LANGUAGES
        ]
        chunks, parsed_paths, failed_paths = self._parse_files(supported_files)
        return chunks, repository.indexed_files, len(parsed_paths), len(failed_paths)

    def index(self, root: str | Path, overwrite: bool = False) -> IndexSummary:
        """Incrementally update changed files and remove deleted-file chunks.

        A missing manifest, a different repository root, or ``overwrite=True``
        triggers a clean rebuild. Failed incremental parses retain their last
        successful hash and vector records so the next run retries them.
        """
        root = Path(root).resolve()
        repository = self.scanner.scan(root)
        supported_files = [
            file for file in repository.files if file.language in SUPPORTED_LANGUAGES
        ]
        current_hashes = {file.path.as_posix(): file.sha256 for file in supported_files}
        previous = self.manifest_store.load()
        root_key = str(root)
        same_repository = previous is not None and previous.repository_root == root_key
        full_rebuild = overwrite or not same_repository

        if full_rebuild:
            files_to_parse = supported_files
            deleted_paths: set[str] = set()
        else:
            files_to_parse = [
                file
                for file in supported_files
                if previous.file_hashes.get(file.path.as_posix()) != file.sha256
            ]
            deleted_paths = set(previous.file_hashes) - set(current_hashes)

        chunks, parsed_paths, failed_paths = self._parse_files(files_to_parse)
        changed_paths = parsed_paths
        if full_rebuild:
            self.db_client.drop_table()
        else:
            for relative_path in sorted(deleted_paths | changed_paths):
                self.db_client.delete_by_path(relative_path)

        embedded_chunk_count = 0
        for offset in range(0, len(chunks), self.batch_size):
            batch = chunks[offset : offset + self.batch_size]
            self.metrics.observe("embedding.batch_size", len(batch))
            with self.metrics.measure("embedding"):
                embeddings = self.embedder.embed_batch(
                    [chunk.to_embedding_text() for chunk in batch]
                )
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedder returned a mismatched number of vectors")
            with self.metrics.measure("index.vector_write"):
                self.db_client.upsert_chunks(
                    batch,
                    [embedding.vector for embedding in embeddings],
                )
            embedded_chunk_count += len(batch)

        if full_rebuild:
            next_hashes = {
                path: current_hashes[path] for path in parsed_paths if path in current_hashes
            }
        else:
            next_hashes = {
                path: content_hash
                for path, content_hash in current_hashes.items()
                if path in parsed_paths or path not in failed_paths
            }
            for path in failed_paths:
                if path in previous.file_hashes:
                    next_hashes[path] = previous.file_hashes[path]

        self.manifest_store.save(IndexManifest(repository_root=root_key, file_hashes=next_hashes))
        unchanged_files = 0 if full_rebuild else len(supported_files) - len(files_to_parse)
        return IndexSummary(
            scanned_files=repository.indexed_files,
            parsed_files=len(parsed_paths),
            chunks=embedded_chunk_count,
            failed_files=len(failed_paths),
            unchanged_files=unchanged_files,
            deleted_files=len(deleted_paths),
        )
