import os
from pathlib import Path

from .ignore import IGNORE_DIRECTORIES


class DirectoryWalker:
    def walk(self, root: Path):
        """Yield repository paths while pruning common generated directories."""
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            child_directories = list(directories)
            directories[:] = [
                name for name in directories if name not in IGNORE_DIRECTORIES
            ]

            for name in child_directories:
                yield current_path / name
            for name in files:
                yield current_path / name