"""Tests for hash-based incremental repository indexing."""
import json
from dataclasses import replace
from pathlib import Path

from coding_agent.embedding.mock import MockEmbedder
from coding_agent.indexing.service import RepositoryIndexer
from coding_agent.parser.engine import TreeSitterEngine
from coding_agent.vectordb.client import LanceDBClient


class CountingEmbedder(MockEmbedder):
    def __init__(self, embedding_dim=24):
        super().__init__(embedding_dim=embedding_dim)
        self.embedded_texts = []

    def embed_batch(self, texts):
        self.embedded_texts.extend(texts)
        return super().embed_batch(texts)


def _write_function(path: Path, name: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"def {name}():\n    {body}\n", encoding="utf-8")


def _records(db: LanceDBClient, embedder: CountingEmbedder) -> list[dict]:
    return db.search(embedder.embed_text("list records").vector, limit=20)


def test_incremental_index_skips_unchanged_and_updates_only_changed_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    main_file = repo / "main.py"
    helper_file = repo / "lib" / "helper.py"
    _write_function(main_file, "main", "return 42")
    _write_function(helper_file, "helper", "return 'stable'")

    embedder = CountingEmbedder()
    db = LanceDBClient(tmp_path / "vectors", dimension=embedder.embedding_dimension)
    indexer = RepositoryIndexer(embedder, db)
    first = indexer.index(repo)
    initial_count = db.count()
    initial_embedded = len(embedder.embedded_texts)
    assert first.parsed_files == 2
    assert first.chunks == initial_count == initial_embedded

    no_op = indexer.index(repo)
    assert no_op.parsed_files == 0
    assert no_op.unchanged_files == 2
    assert no_op.chunks == 0
    assert len(embedder.embedded_texts) == initial_embedded
    assert db.count() == initial_count

    _write_function(main_file, "main", "return 43")
    changed = indexer.index(repo)
    assert changed.parsed_files == 1
    assert changed.unchanged_files == 1
    assert changed.chunks == 1
    assert len(embedder.embedded_texts) == initial_embedded + 1
    assert db.count() == initial_count
    records = _records(db, embedder)
    assert any(record["code"].endswith("return 43") for record in records)

    helper_file.unlink()
    deleted = indexer.index(repo)
    assert deleted.parsed_files == 0
    assert deleted.deleted_files == 1
    assert deleted.chunks == 0
    assert db.count() == initial_count - 1
    assert not any(record["path"] == "lib/helper.py" for record in _records(db, embedder))


def test_full_rebuild_resets_index_and_manifest_tracks_sha256(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "module.py"
    _write_function(source, "calculate", "return 1")

    embedder = CountingEmbedder()
    db = LanceDBClient(tmp_path / "vectors", dimension=embedder.embedding_dimension)
    indexer = RepositoryIndexer(embedder, db)
    initial = indexer.index(repo)
    source.write_text("def replacement():\n    return 2\n", encoding="utf-8")
    rebuilt = indexer.index(repo, overwrite=True)

    manifest_path = Path(db.uri) / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _records(db, embedder)
    assert initial.chunks == 1
    assert rebuilt.parsed_files == 1
    assert db.count() == 1
    assert [record["symbol"] for record in records] == ["replacement"]
    assert manifest["file_hashes"]["module.py"]
    assert manifest["repository_root"] == str(repo.resolve())


def test_incremental_index_rebuilds_when_switching_repository_roots(tmp_path):
    first_repo = tmp_path / "first"
    second_repo = tmp_path / "second"
    first_repo.mkdir()
    second_repo.mkdir()
    _write_function(first_repo / "first.py", "first", "return 1")
    _write_function(second_repo / "second.py", "second", "return 2")

    embedder = CountingEmbedder()
    db = LanceDBClient(tmp_path / "vectors", dimension=embedder.embedding_dimension)
    indexer = RepositoryIndexer(embedder, db)
    indexer.index(first_repo)
    switched = indexer.index(second_repo)
    records = _records(db, embedder)

    assert switched.parsed_files == 1
    assert db.count() == 1
    assert [record["path"] for record in records] == ["second.py"]


def test_vector_upsert_replaces_existing_deterministic_chunk_id(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "module.py"
    _write_function(source, "calculate", "return 1")
    embedder = CountingEmbedder()
    db = LanceDBClient(tmp_path / "vectors", dimension=embedder.embedding_dimension)
    indexer = RepositoryIndexer(embedder, db)
    chunks, _, _, _ = indexer.collect_chunks(repo)
    first_vector = embedder.embed_text(chunks[0].to_embedding_text()).vector
    db.upsert_chunks(chunks, [first_vector])

    updated_chunk = replace(chunks[0], code="def calculate():\n    return 99")
    updated_vector = embedder.embed_text(updated_chunk.to_embedding_text()).vector
    db.upsert_chunks([updated_chunk], [updated_vector])
    records = _records(db, embedder)

    assert db.count() == 1
    assert records[0]["chunk_id"] == chunks[0].chunk_id
    assert records[0]["code"] == updated_chunk.code


def test_failed_changed_file_keeps_old_hash_and_is_retried(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "module.py"
    _write_function(source, "calculate", "return 1")

    embedder = CountingEmbedder()
    db = LanceDBClient(tmp_path / "vectors", dimension=embedder.embedding_dimension)
    indexer = RepositoryIndexer(embedder, db)
    indexer.index(repo)
    manifest_path = Path(db.uri) / "index_manifest.json"
    old_hash = json.loads(manifest_path.read_text(encoding="utf-8"))["file_hashes"][
        "module.py"
    ]

    class FailOnceParser:
        def __init__(self):
            self.engine = TreeSitterEngine()
            self.should_fail = True

        def parse(self, file, language):
            if self.should_fail:
                self.should_fail = False
                raise RuntimeError("temporary parser failure")
            return self.engine.parse(file, language)

    source.write_text("def calculate():\n    return 2\n", encoding="utf-8")
    indexer.parser = FailOnceParser()
    failed = indexer.index(repo)
    retained_hash = json.loads(manifest_path.read_text(encoding="utf-8"))["file_hashes"][
        "module.py"
    ]
    retried = indexer.index(repo)

    assert failed.failed_files == 1
    assert failed.chunks == 0
    assert retained_hash == old_hash
    assert retried.failed_files == 0
    assert retried.parsed_files == 1
    assert retried.chunks == 1
    assert db.count() == 1
    assert any(
        record["code"].endswith("return 2") for record in _records(db, embedder)
    )
