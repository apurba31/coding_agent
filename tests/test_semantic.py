from pathlib import Path
from coding_agent.chunker.models import Chunk, ChunkKind
from coding_agent.search.semantic import SemanticSearcher
from coding_agent.embedding.models import EmbeddingResult
from coding_agent.embedding.embedder import Embedder

class MockEmbedder(Embedder):
    def __init__(self):
        super().__init__()
        
    def embed_text(self, text: str) -> EmbeddingResult:
        # Mock embedding, always returns a dummy vector
        return EmbeddingResult(vector=[0.1, 0.2, 0.3], text=text, dimension=3)
        
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(t) for t in texts]

    @property
    def embedding_dimension(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "mock"


class MockLanceDBClient:
    def __init__(self, records):
        self.records = records
        self.last_query_vector = None
        self.last_limit = None

    def search(self, query_vector: list[float], limit: int = 5, table_name: str = "chunks", where: str | None = None):
        self.last_query_vector = query_vector
        self.last_limit = limit
        return self.records


def test_semantic_searcher():
    embedder = MockEmbedder()
    
    mock_records = [
        {
            "chunk_id": "chunk_1",
            "path": "src/user/UserService.java",
            "language": "java",
            "start_line": 10,
            "end_line": 20,
            "symbol": "findUser",
            "symbol_kind": "method",
            "parent_symbol": "UserService",
            "code": "public User findUser(Long id) { return repository.findById(id); }",
            "docstring": "Finds a user by ID.",
            "embedding_text": "Language: java...",
            "vector": [0.1, 0.2, 0.3],
            "_distance": 0.05
        }
    ]
    db_client = MockLanceDBClient(mock_records)
    
    searcher = SemanticSearcher(embedder=embedder, db_client=db_client)
    
    results = searcher.search("UserService findUser", top_k=2)
    
    # Assert query was embedded
    assert db_client.last_query_vector == [0.1, 0.2, 0.3]
    assert db_client.last_limit == 2
    
    # Assert result
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk_1"
    assert results[0].score == 0.05
    assert results[0].chunk.symbol == "findUser"
    assert results[0].chunk.path == Path("src/user/UserService.java")
