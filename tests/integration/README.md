# Integration tests

Tests in this directory exercise multiple real local subsystems together while replacing external
LLM calls with fakes. They use temporary LanceDB storage and deterministic mock embeddings, so they
need neither network access nor model downloads.

Run only these tests with `pytest -m integration`. Root-level `test_*.py` files remain the current
isolated subsystem suite; future refactoring can migrate those modules into subsystem directories
without coupling unit tests to these end-to-end fixtures.
