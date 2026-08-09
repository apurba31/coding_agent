from pathlib import Path

from tree_sitter import Parser

from .loader import GrammarLoader


class TreeSitterEngine:
    def __init__(self):
        self.loader = GrammarLoader()

    def parse(
        self,
        file: Path,
        language: str,
    ):
        parser = Parser()
        parser.language = self.loader.load(language)
        source = file.read_bytes()

        return parser.parse(source)