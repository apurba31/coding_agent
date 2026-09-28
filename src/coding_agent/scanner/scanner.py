from datetime import datetime
from pathlib import Path

from coding_agent.observability import MetricsCollector, get_metrics_collector
from coding_agent.security import SecurityPolicy

from ..models.file import FileMetadata
from ..models.repository import Repository
from ..utils.hashing import calculate_sha256
from .ignore import (
    IGNORE_DIRECTORIES,
    IGNORE_EXTENSIONS,
    IGNORE_FILENAMES,
    load_gitignore_patterns,
    looks_binary,
    matches_gitignore,
)
from .language import detect_language
from .walker import DirectoryWalker


class RepositoryScanner:
    def __init__(
        self,
        root: Path,
        metrics: MetricsCollector | None = None,
    ):
        self.walker = DirectoryWalker()
        self.metrics = metrics or get_metrics_collector()

    def scan(self, root: Path) -> Repository:
        with self.metrics.measure("scan"):
            repository = self._scan(root)
        self.metrics.increment("files.scanned", repository.indexed_files)
        self.metrics.increment("files.ignored", repository.ignored_files)
        return repository

    def _scan(self, root: Path) -> Repository:
        files = []
        indexed = 0
        ignored = 0
        directories = 0
        gitignore_patterns = load_gitignore_patterns(root)
        policy = SecurityPolicy(root)

        for path in self.walker.walk(root):
            if path.is_dir():
                directories += 1
                if path.name in IGNORE_DIRECTORIES:
                    ignored += 1
                continue

            relative = path.relative_to(root)
            if (
                path.suffix.lower() in IGNORE_EXTENSIONS
                or path.name in IGNORE_FILENAMES
                or path.name.lower() in {name.lower() for name in IGNORE_FILENAMES}
                or policy.is_secret_path(path)
                or matches_gitignore(relative, gitignore_patterns)
                or looks_binary(path)
            ):
                ignored += 1
                continue

            metadata = FileMetadata(
                path=relative,
                absolute_path=path.resolve(),
                extension=path.suffix,
                language=detect_language(path.suffix),
                size=path.stat().st_size,
                last_modified=datetime.fromtimestamp(path.stat().st_mtime),
                is_binary=False,
                sha256=calculate_sha256(path),
            )
            files.append(metadata)
            indexed += 1

        return Repository(
            root=root,
            files=files,
            indexed_files=indexed,
            ignored_files=ignored,
            directories=directories,
        )
