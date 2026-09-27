from ..models.language import Language
from .parser import SourceParser


class ParserRegistry:
    def __init__(self):
        self._parsers: dict[Language, SourceParser] = {}

    @property
    def parsers(self) -> dict[Language, SourceParser]:
        return self._parsers

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
        except KeyError as err:
            raise ValueError(f"No parser registered for language: {language}") from err

    def supports(self, language: Language) -> bool:
        return language in self._parsers
