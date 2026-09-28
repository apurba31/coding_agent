"""Repository-scoped filesystem tools for read/write operations."""

from __future__ import annotations

from pathlib import Path

from coding_agent.security import SecurityPolicy


class RepositoryToolbox:
    """Restrict file operations to a specific repository root."""

    def __init__(self, repo_root: str | Path, policy: SecurityPolicy | None = None):
        self.policy = policy or SecurityPolicy(repo_root)
        self.root = self.policy.root

    def _resolve_path(self, relative_or_absolute: str | Path) -> Path:
        return self.policy.resolve_inside_repo(relative_or_absolute)

    def read_file(self, path: str | Path) -> str:
        target = self.policy.allow_read(path)
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str | Path, content: str) -> str:
        target = self.policy.allow_write(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target.relative_to(self.root))

    def list_dir(self, path: str | Path = ".") -> list[str]:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        names: list[str] = []
        for child in sorted(target.iterdir(), key=lambda item: item.name):
            if self.policy.is_secret_path(child):
                continue
            names.append(child.name)
        return names
