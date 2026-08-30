from abc import ABC, abstractmethod
from pathlib import Path

from coding_agent.models.ast import SyntaxTree
from src.coding_agent.models.file import FileMetadata


class SourceParser(ABC):
    @abstractmethod
    def parse(
        self,
        file: FileMetadata,
    ) -> SyntaxTree:
        """Parse a source file into a syntax tree."""
        raise NotImplementedError