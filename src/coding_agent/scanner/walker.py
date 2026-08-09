from pathlib import Path


class DirectoryWalker:
    def walk(self, root: Path):
        yield from root.rglob('*')