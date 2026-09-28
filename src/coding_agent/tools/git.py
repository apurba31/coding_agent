"""Read-only Git inspection via GitPython (no shelling out).

Git is optional: ordinary directory scanning does not require a repository.
These tools exist so the agent can inspect status/diff without `git` on PATH
being invoked as a free-form shell command.
"""

from __future__ import annotations

from pathlib import Path

from git import InvalidGitRepositoryError, Repo
from git.exc import GitError


class GitToolbox:
    """Expose `git status` and `git diff` constrained to a working tree."""

    def __init__(self, repo_root: str | Path) -> None:
        self.root = Path(repo_root).resolve()

    def _repo(self) -> Repo:
        try:
            repo = Repo(self.root, search_parent_directories=True)
        except InvalidGitRepositoryError as error:
            raise ValueError(f"Not a Git repository: {self.root}") from error
        working_tree = Path(repo.working_tree_dir or self.root).resolve()
        try:
            self.root.relative_to(working_tree)
        except ValueError as error:
            raise ValueError("Requested path is outside the Git working tree") from error
        return repo

    def status(self) -> str:
        """Return porcelain-ish status text for the working tree."""
        try:
            return self._repo().git.status()
        except GitError as error:
            raise RuntimeError(f"git status failed: {error}") from error

    def diff(self) -> str:
        """Return unstaged + staged diff text (empty string when clean)."""
        try:
            repo = self._repo()
            unstaged = repo.git.diff()
            staged = repo.git.diff("--cached")
            parts = [part for part in (unstaged, staged) if part]
            return "\n\n".join(parts)
        except GitError as error:
            raise RuntimeError(f"git diff failed: {error}") from error

    def changed_files(self) -> list[str]:
        """Working-tree paths that differ from HEAD, plus untracked files."""
        repo = self._repo()
        changed = {item.a_path for item in repo.index.diff(None) if item.a_path}
        changed.update(item.a_path for item in repo.index.diff("HEAD") if item.a_path)
        changed.update(repo.untracked_files)
        return sorted(path for path in changed if path)
