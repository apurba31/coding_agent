from pathlib import Path

from pydantic import BaseModel


class SyntaxTree(BaseModel):
    path: Path
    language: str
    root_type: str