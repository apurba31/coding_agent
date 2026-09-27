from typing import Any

from pydantic import BaseModel


class SyntaxTree(BaseModel):
    language: str
    root: Any
    source: bytes
