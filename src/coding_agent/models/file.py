from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from .language import Language


class FileMetadata(BaseModel):
    path: Path
    absolute_path: Path
    extension: str
    language: Language
    size: int
    last_modified: datetime
    is_binary: bool
    sha256: str