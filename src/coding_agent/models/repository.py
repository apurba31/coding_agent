from pathlib import Path

from pydantic import BaseModel

from .file import FileMetadata


class Repository(BaseModel):
    root: Path
    files: list[FileMetadata]
    indexed_files: int
    ignored_files: int
    directories: int