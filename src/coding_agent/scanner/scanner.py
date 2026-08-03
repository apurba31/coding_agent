from datetime import datetime
from pathlib import Path

from .ignore import IGNORE_DIRECTORIES, IGNORE_EXTENSIONS
from .language import detect_language
from .walker import DirectoryWalker
from ..models.file import FileMetadata
from ..models.repository import Repository
from ..utils.hashing import calculate_sha256

class RepositoryScanner:
    def __init__(self, root: Path):
        self.walker = DirectoryWalker()

    def scan(self, root: Path) -> Repository:
        files= []
        indexed = 0
        ignored = 0
        directories = 0

        for path in self.walker.walk(root):
            if path.is_dir():
                directories += 1
                if path.name in IGNORE_DIRECTORIES:
                    ignored += 1
                    continue

            if path.suffix.lower() in IGNORE_EXTENSIONS:
                ignored += 1
                continue

            metadata = FileMetadata(
                path=path.relative_to(root),
                absolute_path=path.resolve(),
                extension=path.suffix,
                language=detect_language(path.suffix),
                size=path.stat().st_size,
                last_modified=datetime.fromtimestamp
                (
                    path.stat().st_mtime
                    ),
                is_binary=False,
                sha256=calculate_sha256(path),
            )
            files.append(metadata)

            return Repository(
                root=root,
                files=files,
                indexed_files=indexed,
                ignored_files=ignored,
                directories=directories
            )