"""Tests for the tool subsystem phase."""

import pytest

from coding_agent.tools.filesystem import RepositoryToolbox
from coding_agent.tools.terminal import TerminalTool


def test_repository_toolbox_restricts_paths_to_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "alpha.py").write_text("print('safe')\n", encoding="utf-8")

    toolbox = RepositoryToolbox(repo)

    assert toolbox.read_file("src/alpha.py") == "print('safe')\n"
    with pytest.raises(ValueError, match="outside the repository"):
        toolbox.read_file("../outside.py")


def test_repository_toolbox_writes_within_repo_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    toolbox = RepositoryToolbox(repo)

    toolbox.write_file("notes.txt", "hello\n")
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "hello\n"

    with pytest.raises(ValueError, match="outside the repository"):
        toolbox.write_file("../escape.txt", "nope")


def test_run_tests_tool_is_registered_with_terminal_enabled(tmp_path):
    from coding_agent.app import create_coding_tools

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tests").mkdir()
    (repo / "tests" / "sample_test.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    class EmptyRetriever:
        def search(self, query, top_k):
            return []

    registry = create_coding_tools(
        repo,
        EmptyRetriever(),
        allow_terminal=True,
    )
    assert registry.has_tool("run_tests")
    from coding_agent.tools.models import ToolCall

    outcome = registry.call(ToolCall("run_tests", {"target": "tests"}))
    assert outcome.success
    assert "exit_code=" in str(outcome.result)


def test_terminal_tool_blocks_dangerous_shell_metacharacters(tmp_path):
    tool = TerminalTool(cwd=tmp_path)

    result = tool.run("echo hello")
    assert result["success"] is True
    assert "hello" in result["stdout"]

    blocked = tool.run("echo hello; rm -rf /")
    assert blocked["success"] is False
    assert "unsafe" in blocked["error"].lower()
