from ..models.language import Language
from .parser import SourceParser


class ParserRegistry:
    def __init__(self):
        self.parsers: dict[Language, SourceParser] = {}

    def register(
            self,
            language: Language,
            parser: SourceParser,
    ) -> None:
        self._parsers[language] = parser

    def get(
            self,
            language: Language,
    ) -> SourceParser:
        try:
            return self._parsers[language]
        except KeyError:
            raise ValueError(
                f"No parser registered for language: {language}"
            )

    def supports(self, language: Language) -> bool:
        return language in self._parsers
