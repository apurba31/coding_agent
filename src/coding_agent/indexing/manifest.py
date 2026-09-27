"""Persistent file-hash manifest for incremental repository indexing."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IndexManifest:
    """Hashes for supported source files belonging to one repository root."""

    repository_root: str
    file_hashes: dict[str, str]


class IndexManifestStore:
    """Load and atomically replace the on-disk index manifest."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> IndexManifest | None:
        """Return a valid manifest, or None when it is absent or malformed."""
        try:
            data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            root = data["repository_root"]
            file_hashes = data["file_hashes"]
            if not isinstance(root, str) or not isinstance(file_hashes, dict):
                return None
            if not all(
                isinstance(path, str) and isinstance(content_hash, str)
                for path, content_hash in file_hashes.items()
            ):
                return None
            return IndexManifest(repository_root=root, file_hashes=file_hashes)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, manifest: IndexManifest) -> None:
        """Write the manifest through a temporary file to avoid partial state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "repository_root": manifest.repository_root,
                    "file_hashes": manifest.file_hashes,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
