"""Tests for AST definitions and lexical code references."""
from datetime import datetime
from pathlib import Path

from coding_agent.chunker.chunker import Chunker
from coding_agent.models.ast import SyntaxTree
from coding_agent.models.file import FileMetadata
from coding_agent.models.language import Language
from coding_agent.navigation import NavigationIndex
from coding_agent.parser.engine import TreeSitterEngine


def _navigation_index(tmp_path):
    root = tmp_path / "repo"
    source_dir = root / "src"
    source_dir.mkdir(parents=True)
    module = source_dir / "helpers.py"
    code = """class UserService:
    def find_user(self, user_id):
        return lookup(user_id)

def find_user(user_id):
    return UserService().find_user(user_id)
"""
    module.write_text(code, encoding="utf-8")
    metadata = FileMetadata(
        path=Path("src/helpers.py"),
        absolute_path=module,
        extension=".py",
        language=Language.PYTHON,
        size=module.stat().st_size,
        last_modified=datetime.now(),
        is_binary=False,
        sha256="test-hash",
    )
    parsed = TreeSitterEngine().parse(module, "python")
    tree = SyntaxTree(language="Python", root=parsed.root_node, source=code.encode())
    chunks = Chunker().chunk(metadata, tree, code)
    return root, chunks


def test_find_definition_supports_exact_and_qualified_method_names(tmp_path):
    root, chunks = _navigation_index(tmp_path)
    index = NavigationIndex(root, chunks)

    exact = index.find_definition("find_user")
    qualified = index.find_definition("UserService.find_user")

    assert len(exact) == 2
    assert len(qualified) == 1
    assert qualified[0].kind == "method"
    assert qualified[0].path == Path("src/helpers.py")
    assert qualified[0].start_line == 2


def test_search_symbols_ranks_exact_name_and_supports_identifier_terms(tmp_path):
    root, chunks = _navigation_index(tmp_path)
    index = NavigationIndex(root, chunks)

    exact = index.search_symbols("find_user")
    terms = index.search_symbols("User Service")

    assert exact[0].name == "find_user"
    assert any(result.qualified_name == "UserService.find_user" for result in terms)


def test_find_references_returns_whole_identifier_occurrences_and_marks_definition(
    tmp_path,
):
    root, chunks = _navigation_index(tmp_path)
    index = NavigationIndex(root, chunks)

    references = index.find_references("find_user")

    assert len(references) == 3
    assert sum(reference.is_definition for reference in references) == 2
    assert all(reference.path == Path("src/helpers.py") for reference in references)
    assert not index.find_references("find")


def test_factory_builds_navigation_from_repository_without_embeddings(tmp_path):
    root, _ = _navigation_index(tmp_path)
    index = NavigationIndex.from_repository(root)

    assert len(index.find_definition("UserService")) == 1
    assert len(index.find_references("lookup")) == 1


def test_reference_search_prunes_ignored_directories_and_obeys_limit(tmp_path):
    root, chunks = _navigation_index(tmp_path)
    vendor = root / "node_modules"
    vendor.mkdir()
    (vendor / "vendor.py").write_text("find_user()\n", encoding="utf-8")
    index = NavigationIndex(root, chunks)

    references = index.find_references("find_user", limit=1)

    assert len(references) == 1
    assert references[0].path == Path("src/helpers.py")
