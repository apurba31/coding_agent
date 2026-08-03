from abc import ABC
from abc import abstractmethod
from pathlib import Path
from coding_agent.models.ast import SyntaxTree

class SourceParser(ABC):
    @abstractmethod
    def parse(
        self,
        file: Path,
    ) -> SyntaxTree:
        pass