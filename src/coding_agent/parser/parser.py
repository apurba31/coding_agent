from abc import ABC, abstractmethod
from pathlib import Path

from coding_agent.models.ast import SyntaxTree


class SourceParser(ABC):
    @abstractmethod
    def parse(
        self,
        file: Path,
    ) -> SyntaxTree:
        pass