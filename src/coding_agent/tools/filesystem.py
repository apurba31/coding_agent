"""Repository-scoped filesystem tools for read/write operations."""

from __future__ import annotations

from pathlib import Path


class RepositoryToolbox:
    """Restrict file operations to a specific repository root."""

    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root).resolve()

    def _resolve_path(self, relative_or_absolute: str | Path) -> Path:
        candidate = Path(relative_or_absolute)
        if not candidate.is_absolute():
            target = (self.root / candidate).resolve()
        else:
            target = candidate.resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"Requested path is outside the repository: {relative_or_absolute}"
            ) from exc
        return target

    def read_file(self, path: str | Path) -> str:
        target = self._resolve_path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str | Path, content: str) -> str:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target.relative_to(self.root))

    def list_dir(self, path: str | Path = ".") -> list[str]:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        return sorted(p.name for p in target.iterdir())
