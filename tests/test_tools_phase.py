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


def test_terminal_tool_blocks_dangerous_shell_metacharacters(tmp_path):
    tool = TerminalTool(cwd=tmp_path)

    result = tool.run("echo hello")
    assert result["success"] is True
    assert "hello" in result["stdout"]

    blocked = tool.run("echo hello; rm -rf /")
    assert blocked["success"] is False
    assert "unsafe" in blocked["error"].lower()
